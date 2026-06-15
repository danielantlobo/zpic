/**
 * @file particles.c
 * @author Ricardo Fonseca
 * @brief Particle species
 * @version 0.9 (Direct Atomic Optimization)
 * @date 2022-02-04
 * @copyright Copyright (c) 2022
 */

 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 #include <stdio.h>
 #include <omp.h> // Added for OpenMP
 
 #include "particles.h"
 
 #include "random.h"
 #include "emf.h"
 #include "current.h"
 
 #include "zdf.h"
 #include "timer.h"
 
 static double _spec_time = 0.0;
 static uint64_t _spec_npush = 0;
 
 void spec_sort( t_species *spec );
 void spec_move_window( t_species *spec );
 
 
 /**
  * @brief Returns the total time spent pushing particles (includes boundaries and moving window)
  * @return  Total time in seconds
  */
 double spec_time( void )
 {
     return _spec_time;
 }
 
 /**
  * @brief Returns the total number of particle pushes
  * @return  Number of particle pushes
  */
 uint64_t spec_npush( void )
 {
     return _spec_npush;
 }
 
 /**
  * @brief Returns the performance achieved by the code (push time)
  * @return  Performance in seconds per particle
  */
 double spec_perf( void )
 {
     return (_spec_npush > 0 )? _spec_time / _spec_npush: -1.0;
 }
 
 /*********************************************************************************************
  Initialization
  *********************************************************************************************/
 
 void spec_set_u( t_species* spec, const int start, const int end )
 {
 #if 0
     for (int i = start; i <= end; i++) {
         spec->part[i].ux = spec -> ufl[0] + spec -> uth[0] * rand_norm();
         spec->part[i].uy = spec -> ufl[1] + spec -> uth[1] * rand_norm();
         spec->part[i].uz = spec -> ufl[2] + spec -> uth[2] * rand_norm();
     }
 #else
     // Initialize thermal component
     for (int i = start; i <= end; i++) {
         spec->part[i].ux = spec -> uth[0] * rand_norm();
         spec->part[i].uy = spec -> uth[1] * rand_norm();
         spec->part[i].uz = spec -> uth[2] * rand_norm();
     }
 
     float3 * restrict net_u = (float3 *) malloc( spec->nx * sizeof(float3));
     int * restrict    npc   = (int *) malloc( spec->nx * sizeof(int));
 
     memset(net_u, 0, spec->nx * sizeof(float3) );
     memset(npc, 0, (spec->nx) * sizeof(int) );
 
     for (int i = start; i <= end; i++) {
         const int idx  = spec -> part[i].ix;
         net_u[ idx ].x += spec->part[i].ux;
         net_u[ idx ].y += spec->part[i].uy;
         net_u[ idx ].z += spec->part[i].uz;
         npc[ idx ] += 1;
     }
 
     for(int i =0; i< spec->nx; i++ ) {
         const float norm = (npc[ i ] > 0) ? 1.0f/npc[i] : 0;
         net_u[ i ].x *= norm;
         net_u[ i ].y *= norm;
         net_u[ i ].z *= norm;
     }
 
     for (int i = start; i <= end; i++) {
         const int idx  = spec -> part[i].ix;
         spec->part[i].ux += spec -> ufl[0] - net_u[ idx ].x;
         spec->part[i].uy += spec -> ufl[1] - net_u[ idx ].y;
         spec->part[i].uz += spec -> ufl[2] - net_u[ idx ].z;
     }
 
     free( npc );
     free( net_u );
 #endif
 }
 
 int spec_np_inj( t_species* spec, const int range[] )
 {
     int np_inj;
 
     switch ( spec -> density.type ) {
     case STEP: 
         {
             int i0 = spec -> density.start / spec -> dx - spec -> n_move;
             if ( i0 > range[1] ) {
                 np_inj = 0;
             } else {
                 if ( i0 < range[0] ) i0 = range[0];
                 np_inj = ( range[1] - i0 + 1 ) * spec -> ppc;
             }
         }
         break;
     case SLAB: 
         {
             int i0 = spec -> density.start / spec -> dx - spec -> n_move;
             int i1 = spec -> density.end / spec -> dx - spec -> n_move;
             if ( (i0 > range[1]) || (i1 < range[0]) ) {
                 np_inj = 0;
             } else {
                 if ( i0 < range[0] ) i0 = range[0];
                 if ( i1 > range[1] ) i1 = range[1];
                 np_inj = ( i1 - i0 + 1 ) * spec -> ppc;
             }
         }
         break;
     case RAMP: 
         {
             float x0 = spec -> density.start;
             float x1 = spec -> density.end;
             float a = (range[0] + spec -> n_move) * spec->dx;
             float b = (range[1] + 1 + spec -> n_move) * spec->dx;
             if ( (x1 <= x0) || (a > x1) || (b < x0) ) {
                 np_inj = 0;
             } else {
                 if ( a < x0 ) a = x0;
                 if ( b > x1 ) b = x1;
                 float n0 = spec -> density.ramp[0];
                 float n1 = spec -> density.ramp[1];
                 float q = (b-a)*( n0 + 0.5 * (a+b-2*x0)*(n1-n0)/(x1-x0));
                 np_inj = q * spec -> ppc / spec -> dx;
             }
         }
         break;
     case CUSTOM: 
         {
             double q = 0.5 * ( (*spec -> density.custom)((range[0] + spec -> n_move) * spec->dx, spec -> density.custom_data) +
                                (*spec -> density.custom)((range[1] + 1 + spec -> n_move) * spec->dx, spec -> density.custom_data) );
             for( int i = range[0]+1; i <= range[1]; i++) {
                 q += (*spec -> density.custom)((i + spec -> n_move) * spec->dx, spec -> density.custom_data);
             }
             np_inj = ceil(q * spec -> ppc);
         }
         break;
     case EMPTY: np_inj = 0; break;
     default: np_inj = ( range[1] - range[0] + 1 ) * spec -> ppc;
     }
     return np_inj;
 }
 
 void spec_set_x( t_species* spec, const int range[] )
 {
     int i, k, ip;
     float start, end;
     const int npc = spec->ppc;
     float poscell[npc];
 
     for (i=0; i<spec->ppc; i++) poscell[i] = ( i + 0.5 ) / npc;
     ip = spec -> np;
 
     switch ( spec -> density.type ) {
     case STEP: 
         start = spec -> density.start / spec -> dx - spec -> n_move;
         for (i = range[0]; i <= range[1]; i++) {
             for (k=0; k<npc; k++) {
                 if ( i + poscell[k] > start ) {
                     spec->part[ip].ix = i;
                     spec->part[ip].x = poscell[k];
                     ip++;
                 }
             }
         }
         break;
     case SLAB: 
         start = spec -> density.start / spec -> dx - spec -> n_move;
         end   = spec -> density.end / spec -> dx - spec -> n_move;
         for (i = range[0]; i <= range[1]; i++) {
             for (k=0; k<npc; k++) {
                 if ( i + poscell[k] > start &&  i + poscell[k] < end ) {
                     spec->part[ip].ix = i;
                     spec->part[ip].x = poscell[k];
                     ip++;
                 }
             }
         }
         break;
     case RAMP: 
         {
             double r0 = spec -> density.start / spec -> dx;
             double r1 = spec -> density.end / spec -> dx;
             if (((range[0] + spec -> n_move) > r1 ) || ((range[1] + spec -> n_move) < r0 )) break;
             double n0 = spec -> density.ramp[0];
             double n1 = spec -> density.ramp[1];
             if ( r0 < 0 ) {
                 n0 += - r0 * (n1-n0) / (r1-r0);
                 r0 = 0;
             }
             double cpp = 1.0 / spec->ppc;
             for( k = spec -> density.total_np_inj; ; k++ ) {
                 double Rs = (k+0.5) * cpp / (r1 - r0);
                 double pos = 2 * Rs / (sqrt( n0*n0 + 2 * (n1-n0) * Rs ) + n0);
                 if ( pos > 1 ) break;
                 pos = r0 + (r1-r0) * pos;
                 int ix = pos;
                 if ( ix - spec -> n_move < range[0] ) break; 
                 if ( ix - spec -> n_move > range[1] ) break;
                 spec->part[ip].ix = ix - spec -> n_move;
                 spec->part[ip].x = pos - ix;
                 ip++;
             }
         }
         break;
     case CUSTOM: 
         {
             const double dx = spec -> dx;
             const double cpp = 1.0 / spec->ppc;
             k = spec -> density.total_np_inj;
             int ix = range[0];
             double n0;
             double n1 = (*spec -> density.custom)((ix + spec -> n_move) * dx, spec -> density.custom_data);
             double d0;
             double d1 = spec -> density.custom_q_inj;
             double Rs;
             while( ix <= range[1] ){
                 n0 = n1;
                 n1 = (*spec -> density.custom)((ix + 1 + spec -> n_move)*dx, spec -> density.custom_data);
                 d0 = d1;
                 d1 += 0.5 * (n0+n1);
                 while( ( Rs =  (k+0.5) * cpp ) < d1 ) {
                     double pos = 2 * (Rs-d0) /( sqrt( n0*n0 + 2 * (n1-n0) * (Rs-d0) ) + n0 );
                     spec->part[ip].ix = ix;
                     spec->part[ip].x = pos;
                     ip++;
                     k++;
                 }
                 ix++;
             }
             spec -> density.custom_q_inj = d1;
         }
         break;
     case EMPTY: break;
     default: 
         for (i = range[0]; i <= range[1]; i++) {
             for (k=0; k<npc; k++) {
                 spec->part[ip].ix = i;
                 spec->part[ip].x = poscell[k];
                 ip++;
             }
         }
     }
     spec -> density.total_np_inj += ip - spec -> np;
     spec -> np = ip;
 }
 
 void spec_grow_buffer( t_species* spec, const int size ) {
     if ( size > spec -> np_max ) {
         spec -> np_max = ( size/1024 + 1) * 1024;
         spec -> part = realloc( (void*) spec -> part, spec -> np_max * sizeof(t_part) );
     }
 }
 
 void spec_inject_particles( t_species* spec, const int range[] )
 {
     int start = spec -> np;
     int np_inj = spec_np_inj( spec, range );
     spec_grow_buffer( spec, spec -> np + np_inj );
     spec_set_x( spec, range );
     spec_set_u( spec, start, spec -> np - 1 );
 }
 
 void spec_new( t_species* spec, char name[], const float m_q, const int ppc,
               const float *ufl, const float * uth,
               const int nx, float box, const float dt, t_density* density )
 {
     int i, npc;
     strncpy( spec -> name, name, MAX_SPNAME_LEN );
     spec->nx = nx;
     spec->ppc = ppc;
     npc = ppc;
     spec->box = box;
     spec->dx = box / nx;
     spec -> m_q = m_q;
     spec -> q = copysign( 1.0f, m_q ) / npc;
     spec -> dt = dt;
     spec->np_max = 0;
     spec->part = NULL;
 
     if ( density ) {
         spec -> density = *density;
         if ( spec -> density.n == 0. ) spec -> density.n = 1.0;
     } else {
         spec -> density = (t_density) { .type = UNIFORM, .n = 1.0 };
     }
     spec -> density.total_np_inj = 0;
     spec -> density.custom_q_inj = 0.;
     spec ->q *= fabsf( spec -> density.n );
 
     if ( ufl ) {
         for(i=0; i<3; i++) spec -> ufl[i] = ufl[i];
     } else {
         for(i=0; i<3; i++) spec -> ufl[i] = 0;
     }
 
     if ( uth ) {
         for(i=0; i<3; i++) spec -> uth[i] = uth[i];
     } else {
         for(i=0; i<3; i++) spec -> uth[i] = 0;
     }
 
     spec -> iter = 0;
     spec -> moving_window = 0;
     spec -> n_move = 0;
     spec -> np = 0;
     const int range[2] = {0, nx-1};
     spec_inject_particles( spec, range );
     
     // --- OPTIMIZATION: Increase sorting frequency ---
     // Sorting particles improves locality, making atomics faster, because threads don't fight over the same cache lines.
     spec -> n_sort = 10; // Sort every 10 steps
     
     spec -> bc_type = PART_BC_PERIODIC;
 }
 
 void spec_move_window( t_species *spec ){
     if ((spec->iter * spec->dt ) > (spec->dx * (spec->n_move + 1)))  {
         int i;
         for( i = 0; i < spec->np; i++ ) {
             spec->part[i].ix--;
         }
         spec -> n_move++;
         const int range[2] = {spec->nx-1,spec->nx-1};
         spec_inject_particles( spec, range );
     }
 }
 
 void spec_delete( t_species* spec )
 {
     free(spec->part);
     spec->np = -1;
 }
 
 /*********************************************************************************************
  Cuurent deposition
  *********************************************************************************************/
 
 void dep_current_esk( int ix0, int di,
                         float x0, float x1,
                         float qnx, float qvy, float qvz,
                         t_current *current )
 {

 }
 
 void dep_current_zamb( int ix0, int di,
                         float x0, float dx,
                         float qnx, float qvy, float qvz,
                         t_current *current )
 {
     typedef struct {
         float x0, x1, dx, qvy, qvz;
         int ix;
     } t_vp;
 
     t_vp vp[3];
     int vnp = 1;
 
     vp[0].x0 = x0;
     vp[0].dx = dx;
     vp[0].x1 = x0+dx;
     vp[0].qvy = qvy/2.0;
     vp[0].qvz = qvz/2.0;
     vp[0].ix = ix0;
 
     if ( di != 0 ) {
         int ib = ( di == 1 );
         float delta = (x0+dx-ib)/dx;
         vp[1].x0 = 1-ib;
         vp[1].x1 = (x0 + dx) - di;
         vp[1].dx = dx*delta;
         vp[1].ix = ix0 + di;
         vp[1].qvy = vp[0].qvy*delta;
         vp[1].qvz = vp[0].qvz*delta;
         vp[0].x1 = ib;
         vp[0].dx *= (1.0f-delta);
         vp[0].qvy *= (1.0f-delta);
         vp[0].qvz *= (1.0f-delta);
         vnp++;
     }
 
     float3* restrict const J = current -> J;
 
     // --- OPTIMIZATION: Direct Atomics ---
     // Write directly to the global grid using atomic updates.
     for (int k = 0; k < vnp; k++) {
         float S0x[2], S1x[2];
         S0x[0] = 1.0f - vp[k].x0;
         S0x[1] = vp[k].x0;
         S1x[0] = 1.0f - vp[k].x1;
         S1x[1] = vp[k].x1;
 
         float val_x = qnx * vp[k].dx;
         float val_y1 = vp[k].qvy * (S0x[0]+S1x[0]+(S0x[0]-S1x[0])/2.0f);
         float val_y2 = vp[k].qvy * (S0x[1]+S1x[1]+(S0x[1]-S1x[1])/2.0f);
         float val_z1 = vp[k].qvz * (S0x[0]+S1x[0]+(S0x[0]-S1x[0])/2.0f);
         float val_z2 = vp[k].qvz * (S0x[1]+S1x[1]+(S0x[1]-S1x[1])/2.0f);
 
         #pragma omp atomic update
         J[ vp[k].ix     ].x += val_x;
         
         #pragma omp atomic update
         J[ vp[k].ix     ].y += val_y1;
         
         #pragma omp atomic update
         J[ vp[k].ix + 1 ].y += val_y2;
         
         #pragma omp atomic update
         J[ vp[k].ix     ].z += val_z1;
         
         #pragma omp atomic update
         J[ vp[k].ix  +1 ].z += val_z2;
     }
 }
 
 /*********************************************************************************************
  Sorting
  *********************************************************************************************/
 
 void spec_sort( t_species* spec )
 {
     const int ncell = spec->nx;
     int * restrict idx  = (int *) malloc(spec->np*sizeof(int));
     int * restrict npic = (int *) malloc( ncell * sizeof(int));
     memset( npic, 0, ncell * sizeof(int));
 
     for (int i=0; i<spec->np; i++) {
         idx[i] = spec->part[i].ix;
         npic[idx[i]]++;
     }
 
     int isum = 0;
     for (int i=0; i<ncell; i++) {
         int j = npic[i];
         npic[i] = isum;
         isum += j;
     }
 
     for (int i=0; i< spec->np; i++) {
         int j = idx[i];
         idx[i] = npic[j]++;
     }
     free(npic);
 
     for (int i=0; i < spec->np; i++) {
         int k = idx[i];
         while ( k > i ) {
             t_part tmp = spec->part[k];
             spec->part[k] = spec->part[i];
             spec->part[i] = tmp;
             int t = idx[k];
             idx[k] = -1;
             k = t;
         }
     }
     free(idx);
 }
 
 /*********************************************************************************************
  Particle advance
  *********************************************************************************************/
 
 void interpolate_fld( const float3* restrict const E, const float3* restrict const B,
               const t_part* restrict const part, float3* restrict const Ep, float3* restrict const Bp )
 {
     int i, ih;
     float w1, w1h;
     i = part->ix;
     w1 = part->x;
     ih = (w1 <0.5f)? -1 : 0;
     w1h = w1 + ((w1 <0.5f)?0.5f:-0.5f);
     ih += i;
 
     Ep->x = E[ih].x * (1.0f - w1h) + E[ih+1].x * w1h;
     Ep->y = E[i ].y * (1.0f -  w1) + E[i+1 ].y * w1;
     Ep->z = E[i ].z * (1.0f -  w1) + E[i+1 ].z * w1;
 
     Bp->x = B[i ].x * (1.0f  - w1) + B[i+1 ].x * w1;
     Bp->y = B[ih].y * (1.0f - w1h) + B[ih+1].y * w1h;
     Bp->z = B[ih].z * (1.0f - w1h) + B[ih+1].z * w1h;
 }
 
 int ltrim( float x )
 {
     return ( x >= 1.0f ) - ( x < 0.0f );
 }
 
 void spec_advance( t_species* spec, t_emf* emf, t_current* current )
 {
     uint64_t t0 = timer_ticks();
 
     const float tem   = 0.5 * spec->dt/spec -> m_q;
     const float dt_dx = spec->dt / spec->dx;
     const float qnx = spec -> q * spec->dx / spec->dt;
     const int nx0 = spec -> nx;
 
     double energy = 0;
 
     // --- OPTIMIZATION START ---
     #pragma omp parallel for reduction(+:energy) schedule(static)
     for (int i=0; i<spec->np; i++) {
 
         float3 Ep, Bp;
         float utx, uty, utz;
         float ux, uy, uz, u2;
         float gamma, rg, gtem, otsq;
         float x1;
         int di;
         float dx;
 
         ux = spec -> part[i].ux;
         uy = spec -> part[i].uy;
         uz = spec -> part[i].uz;
 
         interpolate_fld( emf -> E_part, emf -> B_part, &spec -> part[i], &Ep, &Bp );
 
         Ep.x *= tem; Ep.y *= tem; Ep.z *= tem;
 
         utx = ux + Ep.x;
         uty = uy + Ep.y;
         utz = uz + Ep.z;
 
         u2 = utx*utx + uty*uty + utz*utz;
         gamma = sqrtf( 1 + u2 );
         energy += u2 / ( 1 + gamma );
 
         gtem = tem / gamma;
         Bp.x *= gtem; Bp.y *= gtem; Bp.z *= gtem;
 
         otsq = 2.0f / ( 1.0f + Bp.x*Bp.x + Bp.y*Bp.y + Bp.z*Bp.z );
 
         ux = utx + uty*Bp.z - utz*Bp.y;
         uy = uty + utz*Bp.x - ux*Bp.z;
         uz = utz + utx*Bp.y - uy*Bp.x;
 
         Bp.x *= otsq; Bp.y *= otsq; Bp.z *= otsq;
 
         utx += uy*Bp.z - uz*Bp.y;
         uty += uz*Bp.x - ux*Bp.z;
         utz += ux*Bp.y - uy*Bp.x;
 
         ux = utx + Ep.x;
         uy = uty + Ep.y;
         uz = utz + Ep.z;
 
         spec -> part[i].ux = ux;
         spec -> part[i].uy = uy;
         spec -> part[i].uz = uz;
 
         rg = 1.0f / sqrtf(1.0f + ux*ux + uy*uy + uz*uz);
         dx = dt_dx * rg * ux;
         x1 = spec -> part[i].x + dx;
         di = ltrim(x1);
         x1 -= di;
 
         float qvy = spec->q * uy * rg;
         float qvz = spec->q * uz * rg;
 
         // Pass global current pointer. 
         // Thread safety is handled inside by atomic updates.
         dep_current_zamb( spec -> part[i].ix, di,
                         spec -> part[i].x, dx,
                         qnx, qvy, qvz,
                         current ); 
 
         spec -> part[i].x = x1;
         spec -> part[i].ix += di;
     }
     // --- OPTIMIZATION END ---
 
     spec -> energy = spec-> q * spec -> m_q * energy * spec -> dx;
     spec -> iter += 1;
 
     if ( spec -> moving_window || spec -> bc_type == PART_BC_OPEN ){
         if (spec -> moving_window )	spec_move_window( spec );
         int i = 0;
         while ( i < spec -> np ) {
             if (( spec -> part[i].ix < 0 ) || ( spec -> part[i].ix >= nx0 )) {
                 spec -> part[i] = spec -> part[ -- spec -> np ];
                 continue;
             }
             i++;
         }
     } else {
         for (int i=0; i<spec->np; i++) {
             spec -> part[i].ix += (( spec -> part[i].ix < 0 ) ? nx0 : 0 ) - (( spec -> part[i].ix >= nx0 ) ? nx0 : 0);
         }
     }
 
     if ( spec -> n_sort > 0 ) {
         if ( ! (spec -> iter % spec -> n_sort) ) spec_sort( spec );
     }
 
     _spec_npush += spec -> np;
     _spec_time += timer_interval_seconds( t0, timer_ticks() );
 }
 
 /*********************************************************************************************
  Diagnostics
  *********************************************************************************************/
 
 void spec_deposit_charge( const t_species* spec, float* charge )
 {
     const float q = spec -> q;
     for (int i=0; i<spec->np; i++) {
         int idx = spec->part[i].ix;
         float w1 = spec->part[i].x;
         charge[ idx            ] += ( 1.0f - w1 ) * q;
         charge[ idx + 1        ] += (        w1 ) * q;
     }
     if ( ! spec -> moving_window ){
         charge[ 0 ] += charge[ spec -> nx ];
     }
 }
 
 void spec_rep_particles( const t_species *spec )
 {
     t_zdf_file part_file;
     int i;
     const char * quants[] = { "x", "ux","uy","uz" };
     const char * qlabels[] = { "x", "u_x","u_y","u_z" };
     const char * qunits[] = { "c/\\omega_p", "c","c","c" };
 
     t_zdf_iteration iter = {
         .name = "ITERATION",
         .n = spec->iter,
         .t = spec -> iter * spec -> dt,
         .time_units = "1/\\omega_p"
     };
 
     t_zdf_part_info info = {
         .name = (char *) spec -> name,
         .label = (char *) spec -> name,
         .nquants = 4,
         .quants = (char **) quants,
         .qlabels = (char **) qlabels,
         .qunits = (char **) qunits,
         .np = spec ->np
     };
 
     char path[1024];
     snprintf(path, 1024, "PARTICLES/%s", spec -> name );
     zdf_open_part_file( &part_file, &info, &iter, path );
 
     size_t size = ( spec -> np ) * sizeof( float );
     float* data = malloc( size );
 
     for( i = 0; i < spec ->np; i++ )
         data[i] = (spec -> n_move + spec -> part[i].ix + spec -> part[i].x ) * spec -> dx;
     zdf_add_quant_part_file( &part_file, quants[0], data, spec ->np );
 
     for( i = 0; i < spec ->np; i++ ) data[i] = spec -> part[i].ux;
     zdf_add_quant_part_file( &part_file, quants[1], data, spec ->np );
 
     for( i = 0; i < spec ->np; i++ ) data[i] = spec -> part[i].uy;
     zdf_add_quant_part_file( &part_file, quants[2], data, spec ->np );
 
     for( i = 0; i < spec ->np; i++ ) data[i] = spec -> part[i].uz;
     zdf_add_quant_part_file( &part_file, quants[3], data, spec ->np );
 
     free( data );
     zdf_close_file( &part_file );
 }
 
 void spec_rep_charge( const t_species *spec )
 {
     size_t size = ( spec -> nx + 1 ) * sizeof( float );
     float *charge = malloc( size );
     memset( charge, 0, size );
     spec_deposit_charge( spec, charge );
     float buffer[ spec -> nx ];
     for ( int i = 0; i < spec->nx; i++ ) buffer[i] = charge[i];
     free( charge );
 
     t_zdf_grid_axis axis = {
         .min = spec -> n_move * spec -> dx,
         .max = spec->box + spec -> n_move * spec -> dx,
         .name = "x",
         .label = "x",
         .units = "c/\\omega_p"
     };
 
     char name[128], label[128];
     snprintf(name, 128, "%s-charge", spec -> name);
     snprintf(label, 128, "%s \\rho", spec -> name);
 
     t_zdf_grid_info info = {
         .ndims = 1,
         .name = name,
         .label = label,
         .units = "n_e",
         .axis  = &axis
     };
     info.count[0] = spec->nx;
 
     t_zdf_iteration iter = {
         .name = "ITERATION",
         .n = spec->iter,
         .t = spec -> iter * spec -> dt,
         .time_units = "1/\\omega_p"
     };
 
     char path[1024];
     snprintf(path, 1024, "CHARGE/%s", spec -> name );
     zdf_save_grid( (void *) buffer, zdf_float32, &info, &iter, path );
 }
 
 void spec_pha_axis( const t_species *spec, int i0, int np, int quant, float * restrict axis )
 {
     switch (quant) {
         case X1:
             for (int i = 0; i < np; i++) axis[i] = ( spec -> part[i0+i].x + spec -> part[i0+i].ix ) * spec -> dx;
             break;
         case U1:
             for (int i = 0; i < np; i++) axis[i] = spec -> part[i0+i].ux;
             break;
         case U2:
             for (int i = 0; i < np; i++) axis[i] = spec -> part[i0+i].uy;
             break;
         case U3:
             for (int i = 0; i < np; i++) axis[i] = spec -> part[i0+i].uz;
             break;
     }
 }
 
 const char * spec_pha_axis_units( int quant ) {
     switch (quant) {
         case X1: return("c/\\omega_p");
         case U1: case U2: case U3: return("m_e c");
     }
     return("");
 }
 
 void spec_deposit_pha( const t_species *spec, const int rep_type,
               const int pha_nx[], const float pha_range[][2], float* restrict buf )
 {
     const int BUF_SIZE = 1024;
     float pha_x1[BUF_SIZE], pha_x2[BUF_SIZE];
     const int nrow = pha_nx[0];
     const int quant1 = rep_type & 0x000F;
     const int quant2 = (rep_type & 0x00F0)>>4;
     const float x1min = pha_range[0][0];
     const float x2min = pha_range[1][0];
     const float rdx1 = pha_nx[0] / ( pha_range[0][1] - pha_range[0][0] );
     const float rdx2 = pha_nx[1] / ( pha_range[1][1] - pha_range[1][0] );
 
     for ( int i = 0; i<spec->np; i+=BUF_SIZE ) {
         int np = ( i + BUF_SIZE > spec->np )? spec->np - i : BUF_SIZE;
         spec_pha_axis( spec, i, np, quant1, pha_x1 );
         spec_pha_axis( spec, i, np, quant2, pha_x2 );
 
         for ( int k = 0; k < np; k++ ) {
             float nx1 = ( pha_x1[k] - x1min ) * rdx1;
             float nx2 = ( pha_x2[k] - x2min ) * rdx2;
             int i1 = (int)(nx1 + 0.5f);
             int i2 = (int)(nx2 + 0.5f);
             float w1 = nx1 - i1 + 0.5f;
             float w2 = nx2 - i2 + 0.5f;
             int idx = i1 + nrow*i2;
 
             if ( i2 >= 0 && i2 < pha_nx[1] ) {
                 if (i1 >= 0 && i1 < pha_nx[0]) {
                     buf[ idx ] += (1.0f-w1)*(1.0f-w2)*spec->q;
                 }
                 if (i1+1 >= 0 && i1+1 < pha_nx[0] ) {
                     buf[ idx + 1 ] += w1*(1.0f-w2)*spec->q;
                 }
             }
             idx += nrow;
             if ( i2+1 >= 0 && i2+1 < pha_nx[1] ) {
                 if (i1 >= 0 && i1 < pha_nx[0]) {
                     buf[ idx ] += (1.0f-w1)*w2*spec->q;
                 }
                 if (i1+1 >= 0 && i1+1 < pha_nx[0] ) {
                     buf[ idx + 1 ] += w1*w2*spec->q;
                 }
             }
         }
     }
 }
 
 void spec_rep_pha( const t_species *spec, const int rep_type,
               const int pha_nx[], const float pha_range[][2] )
 {
     float* restrict buf = malloc( pha_nx[0] * pha_nx[1] * sizeof( float ));
     memset( buf, 0, pha_nx[0] * pha_nx[1] * sizeof( float ));
     spec_deposit_pha( spec, rep_type, pha_nx, pha_range, buf );
 
     int quant1 = rep_type & 0x000F;
     int quant2 = (rep_type & 0x00F0)>>4;
     const char * pha_ax1_units = spec_pha_axis_units(quant1);
     const char * pha_ax2_units = spec_pha_axis_units(quant2);
     char const * const pha_ax_name[] = {"x1","x2","x3","u1","u2","u3"};
     char const * const pha_ax_label[] = {"x","y","z","u_x","u_y","u_z"};
 
     t_zdf_grid_axis axis[2];
     axis[0] = (t_zdf_grid_axis) {
         .min = pha_range[0][0],
         .max = pha_range[0][1],
         .name  = (char *) pha_ax_name[ quant1 - 1 ],
         .label = (char *) pha_ax_label[ quant1 - 1 ],
         .units = (char *) pha_ax1_units
     };
     axis[1] = (t_zdf_grid_axis) {
         .min = pha_range[1][0],
         .max = pha_range[1][1],
         .name  = (char *) pha_ax_name[ quant2 - 1 ],
         .label = (char *) pha_ax_label[ quant2 - 1 ],
         .units = (char *) pha_ax2_units
     };
 
     char pha_name[64], pha_label[64];
     snprintf( pha_name, 64,"%s-%s%s", spec -> name, pha_ax_name[quant1-1], pha_ax_name[quant2-1] );
     snprintf( pha_label, 64,"%s %s-%s", spec -> name, pha_ax_label[quant1-1], pha_ax_label[quant2-1] );
 
     t_zdf_grid_info info = {
         .ndims = 2,
         .name = pha_name,
         .label = pha_label,
         .units = "a.u.",
         .axis  = axis
     };
     info.count[0] = pha_nx[0];
     info.count[1] = pha_nx[1];
 
     t_zdf_iteration iter = {
         .name = "ITERATION",
         .n = spec->iter,
         .t = spec -> iter * spec -> dt,
         .time_units = "1/\\omega_p"
     };
 
     char path[1024];
     snprintf(path, 1024, "PHASESPACE/%s", spec -> name );
     zdf_save_grid( (void *) buf, zdf_float32, &info, &iter, path );
     free( buf );
 }
 
 void spec_report( const t_species *spec, const int rep_type,
                   const int pha_nx[], const float pha_range[][2] )
 {
     switch (rep_type & 0xF000) {
         case CHARGE: spec_rep_charge( spec ); break;
         case PHA: spec_rep_pha( spec, rep_type, pha_nx, pha_range ); break;
         case PARTICLES: spec_rep_particles( spec ); break;
     }
 }