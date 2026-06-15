#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Linkage for GCC/Linker compatibility
extern "C" {

#include "particles.h"
#include "random.h"
#include "emf.h"
#include "current.h"
#include "zdf.h"
#include "timer.h"

#define FULL_MASK 0xffffffff

/* CUDA Globals & Constants */

__constant__ float c_dt, c_dx, c_mq, c_q;
__constant__ int c_nx;

static double _spec_time = 0.0;
static uint64_t _spec_npush = 0;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

/* Helper Functions */

__device__ __forceinline__ int device_ltrim(float x) {
    return (x >= 1.0f) - (x < 0.0f);
}

// Warp Shuffles
__device__ __forceinline__ double warp_reduce_energy(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// Optimized Deposition with atomic additions
__device__ void k_dep_current_local(int ix0, int di, float x0, float dx,
                                    float qnx, float qvy, float qvz,
                                    float3 *J) {
    float v_x0[2], v_x1[2], v_dx[2], v_qvy[2], v_qvz[2];
    int v_ix[2];
    int vnp = 1;

    v_x0[0] = x0; v_dx[0] = dx; v_x1[0] = x0 + dx;
    v_qvy[0] = qvy * 0.5f; v_qvz[0] = qvz * 0.5f; v_ix[0] = ix0;

    if (di != 0) {
        int ib = (di == 1);
        float delta = (x0 + dx - (float)ib) / dx;
        v_x0[1] = 1.0f - (float)ib; v_x1[1] = (x0 + dx) - (float)di; v_dx[1] = dx * delta;
        v_ix[1] = ix0 + di; v_qvy[1] = v_qvy[0] * delta; v_qvz[1] = v_qvz[0] * delta;
        v_x1[0] = (float)ib; v_dx[0] *= (1.0f - delta); v_qvy[0] *= (1.0f - delta); v_qvz[0] *= (1.0f - delta);
        vnp = 2;
    }

    for (int k = 0; k < vnp; k++) {
        float S0x[2] = {1.0f - v_x0[k], v_x0[k]};
        float S1x[2] = {1.0f - v_x1[k], v_x1[k]};
        float w0 = S0x[0] + S1x[0] + (S0x[0] - S1x[0]) * 0.5f;
        float w1 = S0x[1] + S1x[1] + (S0x[1] - S1x[1]) * 0.5f;

        atomicAdd(&J[v_ix[k]].x, qnx * v_dx[k]);
        atomicAdd(&J[v_ix[k]].y, v_qvy[k] * w0);
        atomicAdd(&J[v_ix[k] + 1].y, v_qvy[k] * w1);
        atomicAdd(&J[v_ix[k]].z, v_qvz[k] * w0);
        atomicAdd(&J[v_ix[k] + 1].z, v_qvz[k] * w1);
    }
}

/* Main Particle Kernel */

__global__ void k_spec_advance_optimized(t_part *part, int np, const float3 *E, const float3 *B, 
                                        float3 *J, double *energy_reduction) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double my_energy = 0.0;

    if (i < np) {
        t_part p = part[i];
        int pix = p.ix; float pw1 = p.x;
        int ih = (pw1 < 0.5f) ? pix - 1 : pix;
        float w1h = pw1 + ((pw1 < 0.5f) ? 0.5f : -0.5f);

        float3 Ep, Bp;
        Ep.x = E[ih].x * (1.0f - w1h) + E[ih+1].x * w1h;
        Ep.y = E[pix].y * (1.0f - pw1) + E[pix+1].y * pw1;
        Ep.z = E[pix].z * (1.0f - pw1) + E[pix+1].z * pw1;
        Bp.x = B[pix].x * (1.0f - pw1) + B[pix+1].x * pw1;
        Bp.y = B[ih].y * (1.0f - w1h) + B[ih+1].y * w1h;
        Bp.z = B[ih].z * (1.0f - w1h) + B[ih+1].z * w1h;

        const float tem = 0.5f * c_dt / c_mq;
        float3 u_half = { p.ux + Ep.x * tem, p.uy + Ep.y * tem, p.uz + Ep.z * tem };

        double u2 = (double)u_half.x * u_half.x + (double)u_half.y * u_half.y + (double)u_half.z * u_half.z;
        double gamma = sqrt(1.0 + u2);
        my_energy = u2 / (1.0 + gamma);

        float gtem = tem / (float)gamma;
        Bp.x *= gtem; Bp.y *= gtem; Bp.z *= gtem;
        float otsq = 2.0f / (1.0f + Bp.x*Bp.x + Bp.y*Bp.y + Bp.z*Bp.z);
        float vx = u_half.x + u_half.y*Bp.z - u_half.z*Bp.y;
        float vy = u_half.y + u_half.z*Bp.x - u_half.x*Bp.z;
        float vz = u_half.z + u_half.x*Bp.y - u_half.y*Bp.x;

        p.ux = u_half.x + (vy*Bp.z - vz*Bp.y) * otsq + Ep.x * tem;
        p.uy = u_half.y + (vz*Bp.x - vx*Bp.z) * otsq + Ep.y * tem;
        p.uz = u_half.z + (vx*Bp.y - vy*Bp.x) * otsq + Ep.z * tem;

        float rg = 1.0f / (float)gamma;
        float dx_move = (c_dt / c_dx) * rg * p.ux;
        float x_new = p.x + dx_move;
        int di = device_ltrim(x_new);

        float qnx = c_q * c_dx / c_dt;
        k_dep_current_local(p.ix, di, p.x, dx_move, qnx, (c_q * p.uy * rg), (c_q * p.uz * rg), J);

        p.x = x_new - (float)di; p.ix += di;
        if (p.ix < 0) p.ix += c_nx; else if (p.ix >= c_nx) p.ix -= c_nx;
        part[i] = p;
    }

    // Warp-Level Reduction
    my_energy = warp_reduce_energy(my_energy);
    static __shared__ double s_eng[32];
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32;
    if (lane == 0) s_eng[wid] = my_energy;
    __syncthreads();

    if (wid == 0) {
        double block_sum = (threadIdx.x < blockDim.x / 32) ? s_eng[lane] : 0.0;
        block_sum = warp_reduce_energy(block_sum);
        if (lane == 0) atomicAdd(energy_reduction, block_sum);
    }
}

/* Host Functions */

double spec_time(void) { return _spec_time; }
uint64_t spec_npush(void) { return _spec_npush; }
double spec_perf(void) { return (_spec_npush > 0) ? _spec_time / _spec_npush : -1.0; }

void spec_grow_buffer(t_species* spec, const int size) {
    if (size > spec->np_max) {
        spec->np_max = (size / 1024 + 1) * 1024;
        spec->part = (t_part*)realloc(spec->part, spec->np_max * sizeof(t_part));
    }
}

void spec_set_x(t_species* spec, const int range[]) {
    int i, k, ip;
    int npc = spec->ppc;
    float *poscell = (float*)malloc(npc * sizeof(float));
    for (i = 0; i < npc; i++) poscell[i] = (i + 0.5f) / npc;
    ip = spec->np;
    for (i = range[0]; i <= range[1]; i++) {
        for (k = 0; k < npc; k++) {
            spec->part[ip].ix = i;
            spec->part[ip].x = poscell[k];
            ip++;
        }
    }
    spec->np = ip;
    free(poscell);
}

void spec_set_u(t_species* spec, const int start, const int end) {
    for (int i = start; i <= end; i++) {
        spec->part[i].ux = spec->uth[0] * rand_norm() + spec->ufl[0];
        spec->part[i].uy = spec->uth[1] * rand_norm() + spec->ufl[1];
        spec->part[i].uz = spec->uth[2] * rand_norm() + spec->ufl[2];
    }
}

int spec_np_inj(t_species* spec, const int range[]) {
    return (range[1] - range[0] + 1) * spec->ppc;
}

void spec_inject_particles(t_species* spec, const int range[]) {
    int start = spec->np;
    int np_inj = spec_np_inj(spec, range);
    spec_grow_buffer(spec, spec->np + np_inj);
    spec_set_x(spec, range);
    spec_set_u(spec, start, spec->np - 1);
}

void spec_new(t_species* spec, char name[], const float m_q, const int ppc,
            const float *ufl, const float *uth, const int nx, float box, 
            const float dt, t_density* density) {
    strncpy(spec->name, name, MAX_SPNAME_LEN);
    spec->nx = nx; spec->ppc = ppc; spec->box = box;
    spec->dx = box / nx; spec->m_q = m_q;
    spec->q = copysignf(1.0f, m_q) / ppc;
    spec->dt = dt; spec->np_max = 0; spec->part = NULL;

    if (density) spec->density = *density;
    else { spec->density.n = 1.0; spec->density.type = UNIFORM; }
    spec->q *= fabsf(spec->density.n);

    for (int i = 0; i < 3; i++) spec->ufl[i] = ufl ? ufl[i] : 0;
    for (int i = 0; i < 3; i++) spec->uth[i] = uth ? uth[i] : 0;

    spec->iter = 0; spec->moving_window = 0; spec->n_move = 0; spec->np = 0;
    const int range[2] = {0, nx - 1};
    spec_inject_particles(spec, range);

    spec->n_sort = 100000; spec->bc_type = PART_BC_PERIODIC;
    spec->d_part = NULL; spec->d_E = NULL; spec->d_B = NULL; spec->d_J = NULL;
    spec->d_np_max = 0;
}

void spec_delete(t_species* spec) {
    if (spec->part) free(spec->part);
    if (spec->d_part) cudaFree(spec->d_part);
    if (spec->d_E) cudaFree(spec->d_E);
    if (spec->d_B) cudaFree(spec->d_B);
    if (spec->d_J) cudaFree(spec->d_J);
}

void spec_advance(t_species* spec, t_emf* emf, t_current* current) {
    uint64_t t0 = timer_ticks();
    static bool c_init = false;
    if (!c_init) {
        cudaMemcpyToSymbol(c_dt, &spec->dt, sizeof(float));
        cudaMemcpyToSymbol(c_dx, &spec->dx, sizeof(float));
        cudaMemcpyToSymbol(c_mq, &spec->m_q, sizeof(float));
        cudaMemcpyToSymbol(c_q, &spec->q, sizeof(float));
        cudaMemcpyToSymbol(c_nx, &spec->nx, sizeof(int));
        c_init = true;
    }

    if (spec->np > spec->d_np_max) {
        if (spec->d_part) cudaFree(spec->d_part);
        spec->d_np_max = (spec->np / 1024 + 1) * 1024;
        cudaMalloc(&spec->d_part, spec->d_np_max * sizeof(t_part));
    }

    int fld_size = (emf->gc[0] + emf->nx + emf->gc[1]) * sizeof(float3);
    if (!spec->d_E) {
        cudaMalloc(&spec->d_E, fld_size); cudaMalloc(&spec->d_B, fld_size);
        cudaMalloc(&spec->d_J, fld_size);
    }

    cudaMemcpy(spec->d_part, spec->part, spec->np * sizeof(t_part), cudaMemcpyHostToDevice);
    cudaMemcpy(spec->d_E, emf->E_buf, fld_size, cudaMemcpyHostToDevice);
    cudaMemcpy(spec->d_B, emf->B_buf, fld_size, cudaMemcpyHostToDevice);
    cudaMemset(spec->d_J, 0, fld_size);

    double *d_energy, h_energy = 0;
    cudaMalloc(&d_energy, sizeof(double)); cudaMemset(d_energy, 0, sizeof(double));

    int blockSize = 256; int numBlocks = (spec->np + blockSize - 1) / blockSize;
    k_spec_advance_optimized<<<numBlocks, blockSize>>>(spec->d_part, spec->np, 
        spec->d_E + emf->gc[0], spec->d_B + emf->gc[0], spec->d_J + current->gc[0], d_energy);
    cudaDeviceSynchronize();

    cudaMemcpy(spec->part, spec->d_part, spec->np * sizeof(t_part), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);

    float3 *temp_J = (float3*)malloc(fld_size);
    cudaMemcpy(temp_J, spec->d_J, fld_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < (current->gc[0] + current->nx + current->gc[1]); i++) {
        current->J_buf[i].x += temp_J[i].x; current->J_buf[i].y += temp_J[i].y; current->J_buf[i].z += temp_J[i].z;
    }

    spec->energy = spec->q * spec->m_q * h_energy * spec->dx;
    spec->iter++; _spec_npush += spec->np;
    _spec_time += timer_interval_seconds(t0, timer_ticks());
    free(temp_J); cudaFree(d_energy);
}

void spec_report(const t_species *spec, const int rep_type, const int pha_nx[], const float pha_range[][2]) {
}

} // End of extern "C"