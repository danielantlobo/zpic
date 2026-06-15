/*
 * particles.h
 * zpic - CUDA Compatible Version
 */

 #ifndef __PARTICLES__
 #define __PARTICLES__
 
 #include "zpic.h"
 #include "emf.h"
 #include "current.h"
 
 #include <stdint.h>
 
 /**
  * @brief Maximum species name length
  */
 #define MAX_SPNAME_LEN 32
 
 /**
  * @brief Structure holding single particle data
  */
 typedef struct Particle {
	 int ix;		///< Particle cell index
	 float x;	///< Position inside cell
	 float ux;	///< Generalized velocity along x
	 float uy;	///< Generalized velocity along y
	 float uz;	///< Generalized velocity along z
 } t_part;
 
 /**
  * @brief Types of density profile
  */
 enum density_type {
	 UNIFORM, EMPTY, STEP, SLAB, RAMP, CUSTOM
 };
 
 /**
  * @brief Density profile parameters
  */
 typedef struct Density {
	 float n;
	 enum density_type type;
	 float start;
	 float end;
	 float ramp[2];
	 float (*custom)(float, void*);
	 void *custom_data;
	 unsigned long total_np_inj;
	 double custom_q_inj;
 } t_density;
 
 /**
  * @brief Types of particle boundary conditions
  */
 enum part_boundary { 
	 PART_BC_NONE, PART_BC_PERIODIC, PART_BC_OPEN
 };
 
 /**
  * @brief Set of particles (Updated for CUDA)
  */
 typedef struct Species {
	 char name[MAX_SPNAME_LEN+1];
 
	 // Host Particles
	 t_part *part;	///< Host particle buffer
	 int np;			///< Number of particles in buffer
	 int np_max;		///< Maximum number of particles in buffer
 
	 /* --- CUDA DEVICE POINTERS --- */
	 t_part *d_part;     ///< Device particle buffer
	 float3 *d_E;        ///< Device E-field buffer
	 float3 *d_B;        ///< Device B-field buffer
	 float3 *d_J;        ///< Device Current buffer
	 int    d_np_max;    ///< Track current device buffer capacity
	 /* ---------------------------------------- */
 
	 float m_q;          ///< mass over charge ratio
	 double energy;      ///< total kinetic energy
	 float q;            ///< charge of individual particle
	 int ppc;            ///< Particles per cell
	 t_density density;
	 
	 float ufl[3];	    ///< Initial fluid momentum
	 float uth[3];	    ///< Initial thermal momentum
 
	 int nx;			    ///< Number of grid points
	 float dx;		    ///< Cell size
	 float box;		    ///< Simulation box size
	 float dt;           ///< Time step
	 int iter;           ///< Iteration number
 
	 int moving_window;
	 int n_move;
	 enum part_boundary bc_type;
	 int n_sort;
 
 } t_species;
 
 /* --- Function Prototypes --- */
 
 void spec_new( t_species* spec, char name[], const float m_q, const int ppc,
			   const float ufl[], const float uth[],
			   const int nx, float box, const float dt, t_density* density );
 
 void spec_delete( t_species* spec );
 void spec_grow_buffer( t_species* spec, const int size );
 void spec_advance( t_species* spec, t_emf* emf, t_current* current );
 
 uint64_t spec_npush( void );
 double spec_time( void );
 double spec_perf( void );
 
 /* Diagnostics */
 #define CHARGE 		0x1000
 #define PHA    		0x2000
 #define PARTICLES   0x3000
 #define X1     		0x0001
 #define U1     		0x0004
 #define U2     		0x0005
 #define U3     		0x0006
 
 #define PHASESPACE(a,b) ((a) + (b)*16 + PHA)
 
 void spec_deposit_pha( const t_species *spec, const int rep_type,
			   const int pha_nx[], const float pha_range[][2], float* buf );
 
 void spec_deposit_charge( const t_species* spec, float* charge );
 
 void spec_report( const t_species *spec, const int rep_type,
				   const int pha_nx[], const float pha_range[][2] );
 
 #endif