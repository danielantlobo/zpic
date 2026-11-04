# ZPIC on Fujitsu A64FX and NVIDIA A100

Two ports of [ZPIC](https://github.com/ricardo-fonseca/zpic) (a 1D electromagnetic particle-in-cell plasma code) done for the Parallel Computing course at the University of Minho (2025/26), both run on the Deucalion supercomputer:

- **WA1, CPU (C + OpenMP + SVE) on a Fujitsu A64FX:** 76.30 s → 13.86 s, **5.51x over the serial code**, with 12 threads on one Core Memory Group.
- **WA2, GPU (CUDA C++) on an NVIDIA A100:** **123.86 Mpart/s** at 4,000 particles per cell, **27.4x over the serial code on one CPU core**, energy error below 2.1%.

The two use different input decks (see each section), so their speedups are not comparable with each other. Every number below links to the table or log it comes from; see [Limitations](#limitations) before quoting them.

## Results

### WA1: OpenMP on A64FX ([`WA1/`](WA1))

Deucalion `normal-arm` partition, one Fujitsu A64FX (48 cores in 4 Core Memory Groups of 12, HBM2), GCC 13, `-Ofast -mcpu=a64fx -flto`. Input deck `laser_particles` (20,000 cells, 10 particles per cell, t_max = 40). Baseline: the unmodified serial code on the same partition.

| Phase | Serial (s) | 12 threads (s) | Speedup |
| --- | --- | --- | --- |
| Particle advance (`spec_advance`) | 46.77 | 8.72 | 5.36x |
| Field advance (`emf_advance`) | 11.13 | 3.35 | 3.32x |
| **Total simulation** | **76.30** | **13.86** | **5.51x** |

Source: [WA1/report.pdf](WA1/report.pdf), Section IV.A. Threads: `OMP_NUM_THREADS=12`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` (one CMG). 5.51x on 12 threads is about **46% parallel efficiency**. Running on all 48 cores was slower (field solver ~8.1 s, slower than serial) because data then crosses CMGs.

What changed in the code: OpenMP particle push with `omp atomic` current deposition (compiled to A64FX LSE atomics), particle sorting every 10 steps, parallel first-touch initialisation so pages land on the CMG that uses them, `omp for simd` on the Yee field solvers, and a buffer-swap parallel current smoother. Profiled with Score-P/CubeGUI.

### WA2: CUDA on A100 ([`WA2/`](WA2))

Deucalion `normal-a100-40` partition, one NVIDIA A100 40 GB (PCIe vs SXM4 [unverified](#limitations)), CUDA 12.9.1, `nvcc -O3 -arch=sm_80`, host code `gcc -Ofast`. Input deck `twostream` (120 cells, 2 species, 501 time steps); particles per cell (PPC) varied. Time = particle-advance timer, which includes the per-step host↔device copies.

| PPC | Serial CPU, 1 core (s) | A100 (s) | Speedup | A100 throughput (Mpart/s) | ns/particle | Energy error |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 2.65 | 0.572 | 4.6x | 21.01 | 47.59 | 1.58% |
| 500 | 13.25 (estimated) | 0.895 | 14.8x | 67.20 | 14.88 | 1.86% |
| 1000 | 26.49 | 1.288 | 20.5x | 93.38 | 10.71 | 1.88% |
| 2000 | 53.05 | 2.239 | 23.7x | 107.42 | 9.31 | 2.02% |
| **4000** | **106.23** | **3.883** | **27.4x** | **123.86** | **8.07** | **1.78%** |

Source: [WA2/report.pdf](WA2/report.pdf), Tables I and II. The serial baseline runs at ~4.53 Mpart/s at every size; the 500-PPC baseline was extrapolated from that, not measured. Check: 120 cells × 4,000 PPC × 2 species × 501 steps / 3.883 s = 123.9 Mpart/s.

The log committed with the code, [WA2/slurm-844957.out](WA2/slurm-844957.out), is a separate run of the default configuration (500 PPC): 0.985 s, 61.0 Mpart/s, energy error 1.88%, about 10% slower than the report's 500-PPC row.

What changed in the code: particle push and current deposition fused into one kernel (`k_spec_advance_optimized`), current deposition with `atomicAdd`, kinetic energy reduced with `__shfl_down_sync` warp shuffles, then shared memory, then one atomic per block, and simulation constants in `__constant__` memory.

## Build and run (Deucalion, Slurm)

**WA1 (A64FX):**
```bash
cd WA1
sbatch run_zpic_a64fx.sh    # builds with GCC 13, runs on 12 cores of one CMG, exclusive node
```

**WA2 (A100), default validated configuration (500 PPC):**
```bash
cd WA2
sbatch run_zpic.sh          # builds and runs the default twostream deck
```

**WA2 at another density** (default stays 500):
```bash
make clean && make PPC=4000   # sets ZPIC_PPC at compile time
sbatch run_zpic_ppc4000.sh    # same, plus nvidia-smi and a saved log
```

Edit the `--account` line in each script to your own allocation.

## Limitations

- **The baselines are serial, single-core code.** 27.4x (WA2) and 5.51x (WA1) are against the unoptimised serial program, not an optimised multicore CPU version. Against a well-parallelised CPU code the GPU speedup would be much smaller.
- **WA2 copies all particles to the GPU and back on every step** (`spec_advance` in [`WA2/src/particles.cu`](WA2/src/particles.cu)), and allocates and frees a small buffer each step. The report says transfers happen only at initialisation and at the end; the code does not do that. The measured times include these copies, so keeping particles resident on the GPU is the obvious next optimisation.
- **WA1 uses 12 of the A64FX's 48 cores.** 12 threads in one CMG was the fastest setting for this problem size; the report's 64-byte `posix_memalign` alignment is not in the submitted code.
- **The exact A100 variant is unverified.** The report says A100-PCIE-40GB, while another job on the same partition reported A100-SXM4-40GB; `run_zpic_ppc4000.sh` logs `nvidia-smi` to settle it. The report also lists `-O2`; the Makefile used `-O3`.
- **One configuration per number.** Runs were not repeated, so there is no run-to-run variance; the committed 500-PPC log is ~10% off the report's 500-PPC row.

## Layout

```
WA1/   OpenMP + SVE version for A64FX (src/, Makefile, run_zpic_a64fx.sh, report.pdf)
WA2/   CUDA version for A100 (src/particles.cu, Makefile, run_zpic*.sh, slurm-844957.out, report.pdf)
```

Original ZPIC by Ricardo Fonseca (AGPL-3.0, see [`WA2/src/LICENSE`](WA2/src/LICENSE)). Ports by Daniel Lobo and course partner (student IDs pg60241, pg61541).
