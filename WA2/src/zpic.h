/*
 * zpic.h
 * zpic
 */

 #ifndef __ZPIC__
 #define __ZPIC__
 
 /**
  * @brief Three component vector
  * * Wrap this in a check to prevent conflicts with CUDA's vector_types.h
  */
 #ifndef __CUDACC__  /* If NOT compiling with NVCC */
	 typedef struct Float3 {
		 float x;	///< x vector component
		 float y;	///< y vector component
		 float z;	///< z vector component
	 } float3;
 #else               /* If compiling with NVCC, use a dummy name or skip */
	 /* * CUDA's built-in float3 will be used automatically. 
	  * We don't redefine it here to avoid the error you saw.
	  */
 #endif
 
 /* ANSI C does not define math constants */
 
 #ifndef M_PI
 #define M_PI        3.14159265358979323846264338327950288   ///< pi
 #endif
 
 #ifndef M_PI_2
 #define M_PI_2      1.57079632679489661923132169163975144   ///< pi/2
 #endif
 
 #ifndef M_PI_4
 #define M_PI_4      0.785398163397448309615660845819875721  ///< pi/4
 #endif
 
 #endif