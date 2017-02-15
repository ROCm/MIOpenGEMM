#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/generatorutil.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <tuple>
#include <fstream>

#include <tinygemm/tentotin.hpp>


namespace tinygemm{
namespace tensilegen{
  

void append_parameter_list(std::stringstream & ss){
  ss << "\n(\n";
  ss << "__global const TFLOAT * restrict A, \n";
  ss << "const unsigned offsetA,\n";
  ss << "__global const TFLOAT * restrict B, \n";
  ss << "const unsigned offsetB,\n";    
  ss << "__global TFLOAT       *          C,\n";
  ss << "const unsigned offsetC,\n";
  ss << "const TFLOAT alpha,\n";
  ss << "const TFLOAT beta )\n \n\n\n";
}

KernelString get_tensile_kernelstring(const tinygemm::TinyGemmGeometry & gg){
  
std::stringstream ss;

ss << "/* CT_SSSSS_Cij_Sk_Aik_Bkj_i16x6f_j16x6s_nl6x1_k16_O2 */\n";

ss <<  "#define strideC1J " << gg.ldc  << "\n";
ss <<  "#define strideAK " << gg.lda  << "\n";
ss <<  "#define strideB1J " << gg.ldb << "\n";
ss <<  "#define sizeK " << gg.k << "\n";

if (gg.n%96 != 0){
  throw tinygemm_error("gg.n%96 != 0 in tentotin\n");
}

if (gg.m%96 != 0){
  throw tinygemm_error("gg.m%96 != 0 in tentotin\n");
}


ss <<  "#define N_GROUPS_HORIZONTALLY " << gg.n / 96 << "\n";

ss <<  R"(
/* tile parameters */
#define WG_0I  16
#define WG_1J  16
#define UT_0I   6
#define UT_1J   6
#define MT_0I  96
#define MT_1J  96
#define UNROLL 16
#define PAD     1

/* load size parallel and perpendicular to coalesced dimension */
#define LS_PARA_A 16
#define LS_PERP_A 16
#define LS_PARA_B 16
#define LS_PERP_B 16

/* global memory indices */
#define GLOBAL_C(IDX0I, IDX1J) ( (IDX0I)*strideC0I + (IDX1J)*strideC1J )
#define GLOBAL_A(IDX0I, IDXK) ( (IDX0I)*strideA0I + (IDXK)*strideAK )
#define GLOBAL_B(IDXK, IDX1J) ( (IDXK)*strideBK + (IDX1J)*strideB1J )


/* data types */
#define TFLOAT     float
#define MAD(A,B,DST) DST += mad(A,B,DST) ; //A*B; //

/* MADs */
#define TYPE_MAD(MULA,MULB,DST) DST = MAD(MULA,MULB,DST);
#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) DST = (ALPHA)*(REG) + (BETA)*(DST);

)";

ss << "/* 6x6 micro-tile */" << "#define MICRO_TILE "
  << "rA[0] = localA[offA + 0*WG_0I];    " 
  << "rA[1] = localA[offA + 1*WG_0I];    " 
  << "rA[2] = localA[offA + 2*WG_0I];    " 
  << "rA[3] = localA[offA + 3*WG_0I];    " 
  << "rA[4] = localA[offA + 4*WG_0I];    " 
  << "rA[5] = localA[offA + 5*WG_0I];    " 
  << "rB[0] = localB[offB + 0*WG_1J];    " 
  << "rB[1] = localB[offB + 1*WG_1J];    " 
  << "rB[2] = localB[offB + 2*WG_1J];    " 
  << "rB[3] = localB[offB + 3*WG_1J];    " 
  << "rB[4] = localB[offB + 4*WG_1J];    " 
  << "rB[5] = localB[offB + 5*WG_1J];    " 
  << "offA += (MT_0I+PAD);    "
  << "offB += (MT_1J+PAD);    "
  << "TYPE_MAD(rA[0],rB[0],rC[0][0]);    "
  << "TYPE_MAD(rA[0],rB[1],rC[0][1]);    "
  << "TYPE_MAD(rA[0],rB[2],rC[0][2]);    "
  << "TYPE_MAD(rA[0],rB[3],rC[0][3]);    "
  << "TYPE_MAD(rA[0],rB[4],rC[0][4]);    "
  << "TYPE_MAD(rA[0],rB[5],rC[0][5]);    "
  << "TYPE_MAD(rA[1],rB[0],rC[1][0]);    "
  << "TYPE_MAD(rA[1],rB[1],rC[1][1]);    "
  << "TYPE_MAD(rA[1],rB[2],rC[1][2]);    "
  << "TYPE_MAD(rA[1],rB[3],rC[1][3]);    "
  << "TYPE_MAD(rA[1],rB[4],rC[1][4]);    "
  << "TYPE_MAD(rA[1],rB[5],rC[1][5]);    "
  << "TYPE_MAD(rA[2],rB[0],rC[2][0]);    "
  << "TYPE_MAD(rA[2],rB[1],rC[2][1]);    "
  << "TYPE_MAD(rA[2],rB[2],rC[2][2]);    "
  << "TYPE_MAD(rA[2],rB[3],rC[2][3]);    "
  << "TYPE_MAD(rA[2],rB[4],rC[2][4]);    "
  << "TYPE_MAD(rA[2],rB[5],rC[2][5]);    "
  << "TYPE_MAD(rA[3],rB[0],rC[3][0]);    "
  << "TYPE_MAD(rA[3],rB[1],rC[3][1]);    "
  << "TYPE_MAD(rA[3],rB[2],rC[3][2]);    "
  << "TYPE_MAD(rA[3],rB[3],rC[3][3]);    "
  << "TYPE_MAD(rA[3],rB[4],rC[3][4]);    "
  << "TYPE_MAD(rA[3],rB[5],rC[3][5]);    "
  << "TYPE_MAD(rA[4],rB[0],rC[4][0]);    "
  << "TYPE_MAD(rA[4],rB[1],rC[4][1]);    "
  << "TYPE_MAD(rA[4],rB[2],rC[4][2]);    "
  << "TYPE_MAD(rA[4],rB[3],rC[4][3]);    "
  << "TYPE_MAD(rA[4],rB[4],rC[4][4]);    "
  << "TYPE_MAD(rA[4],rB[5],rC[4][5]);    "
  << "TYPE_MAD(rA[5],rB[0],rC[5][0]);    "
  << "TYPE_MAD(rA[5],rB[1],rC[5][1]);    "
  << "TYPE_MAD(rA[5],rB[2],rC[5][2]);    "
  << "TYPE_MAD(rA[5],rB[3],rC[5][3]);    "
  << "TYPE_MAD(rA[5],rB[4],rC[5][4]);    "
  << "TYPE_MAD(rA[5],rB[5],rC[5][5]);    "
  << "mem_fence(CLK_LOCAL_MEM_FENCE);    ";

ss << R"(
/* preprocessor definitions of kernel arguments*/
#define strideC0I 1
#define strideA0I 1
#define strideBK 1


/* kernel */
__attribute__((reqd_work_group_size(WG_0I*WG_1J,1, 1)))
__kernel void CT_SSSSS_Cij_Sk_Aik_Bkj_i16x6f_j16x6s_nl6x1_k16_O2 )";

  append_parameter_list(ss);
  
  ss <<  R"(
  
  
  
  {    
  
  /* apply offsets */
  C += offsetC;
  A += offsetA;
  B += offsetB;

  /* allocate registers */
  TFLOAT rC[UT_0I][UT_1J] = {{0}};
  TFLOAT rA[UT_0I];
  TFLOAT rB[UT_1J];

  /* allocate local memory */
  __local TFLOAT localA[UNROLL*(MT_0I+PAD)];
  __local TFLOAT localB[UNROLL*(MT_1J+PAD)];

  /* c indices (group) */
  unsigned int g0I = get_group_id(0) % N_GROUPS_HORIZONTALLY; // d0, tensorA
  unsigned int g1J = get_group_id(0) / N_GROUPS_HORIZONTALLY; // d1, tensorB


  /* c indices (local) */
  unsigned int l0I = get_local_id(0) % (WG_0I);//*WG_1J); // d0
  unsigned int l1J = get_local_id(0) / (WG_0I);//*WG_1J); // d1
  unsigned int loadSerial = l0I + l1J*WG_0I;
  unsigned int a0I = loadSerial%LS_PARA_A;
  unsigned int b1J = loadSerial/LS_PARA_B;

  /* unrolled summation index */
  unsigned int aK = loadSerial/LS_PARA_A;
  unsigned int bK = loadSerial%LS_PARA_B;

  /* other non-unrolled summation indices (all start at zero) */

  /* where will this thread read from global memory */
  A += GLOBAL_A( (unsigned long)a0I+g0I*MT_0I, (unsigned long)aK );
  B += GLOBAL_B( (unsigned long)bK, (unsigned long)b1J+g1J*MT_1J );

  /* where will this thread write to local memory */
  __local TFLOAT *lA = localA + a0I + aK*(MT_0I+PAD);
  __local TFLOAT *lB = localB + b1J + bK*(MT_1J+PAD);

  /* registers used for global -> local loads */
  TFLOAT a_0_0, a_1_0, a_2_0, a_3_0, a_4_0, a_5_0;
  TFLOAT b_0_0, b_0_1, b_0_2, b_0_3, b_0_4, b_0_5;


  /* iterate over summation indice(s) */
  unsigned int sumIterK = sizeK / UNROLL;
  do {

    barrier(CLK_LOCAL_MEM_FENCE);

    /* load A global -> local */
    a_0_0 = A[ 0*LS_PARA_A*strideA0I + 0*LS_PERP_A*strideAK];
    a_1_0 = A[ 1*LS_PARA_A*strideA0I + 0*LS_PERP_A*strideAK];
    a_2_0 = A[ 2*LS_PARA_A*strideA0I + 0*LS_PERP_A*strideAK];
    a_3_0 = A[ 3*LS_PARA_A*strideA0I + 0*LS_PERP_A*strideAK];
    a_4_0 = A[ 4*LS_PARA_A*strideA0I + 0*LS_PERP_A*strideAK];
    a_5_0 = A[ 5*LS_PARA_A*strideA0I + 0*LS_PERP_A*strideAK];

    /* load B global -> local */
    b_0_0 = B[ 0*LS_PARA_B*strideBK + 0*LS_PERP_B*strideB1J];
    b_0_1 = B[ 0*LS_PARA_B*strideBK + 1*LS_PERP_B*strideB1J];
    b_0_2 = B[ 0*LS_PARA_B*strideBK + 2*LS_PERP_B*strideB1J];
    b_0_3 = B[ 0*LS_PARA_B*strideBK + 3*LS_PERP_B*strideB1J];
    b_0_4 = B[ 0*LS_PARA_B*strideBK + 4*LS_PERP_B*strideB1J];
    b_0_5 = B[ 0*LS_PARA_B*strideBK + 5*LS_PERP_B*strideB1J];

    lA[ 0*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a_0_0;
    lA[ 1*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a_1_0;
    lA[ 2*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a_2_0;
    lA[ 3*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a_3_0;
    lA[ 4*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a_4_0;
    lA[ 5*LS_PARA_A + 0*LS_PERP_A*(MT_0I+PAD) ] = a_5_0;

    lB[ 0*LS_PARA_B*(MT_1J+PAD) + 0*LS_PERP_B ] = b_0_0;
    lB[ 0*LS_PARA_B*(MT_1J+PAD) + 1*LS_PERP_B ] = b_0_1;
    lB[ 0*LS_PARA_B*(MT_1J+PAD) + 2*LS_PERP_B ] = b_0_2;
    lB[ 0*LS_PARA_B*(MT_1J+PAD) + 3*LS_PERP_B ] = b_0_3;
    lB[ 0*LS_PARA_B*(MT_1J+PAD) + 4*LS_PERP_B ] = b_0_4;
    lB[ 0*LS_PARA_B*(MT_1J+PAD) + 5*LS_PERP_B ] = b_0_5;

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int offA = l0I; // d0
    unsigned int offB = l1J; // d1

    /* do fmas */
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE
    MICRO_TILE

    A += (long) strideAK*UNROLL;
    B += (long) strideBK*UNROLL;
  } while (--sumIterK > 0);


  
   )";
  
  ss << 
  
  R"(
  







  /* which global Cij index */
  unsigned int globalC1J = g1J*MT_1J + l1J;
  unsigned int globalC0I = g0I*MT_0I + l0I;

  /* write global C */
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 0*WG_1J) ], alpha, rC[0][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 1*WG_1J) ], alpha, rC[0][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 2*WG_1J) ], alpha, rC[0][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 3*WG_1J) ], alpha, rC[0][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 4*WG_1J) ], alpha, rC[0][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 0*WG_0I, (unsigned long) globalC1J + 5*WG_1J) ], alpha, rC[0][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 0*WG_1J) ], alpha, rC[1][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 1*WG_1J) ], alpha, rC[1][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 2*WG_1J) ], alpha, rC[1][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 3*WG_1J) ], alpha, rC[1][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 4*WG_1J) ], alpha, rC[1][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 1*WG_0I, (unsigned long) globalC1J + 5*WG_1J) ], alpha, rC[1][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 0*WG_1J) ], alpha, rC[2][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 1*WG_1J) ], alpha, rC[2][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 2*WG_1J) ], alpha, rC[2][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 3*WG_1J) ], alpha, rC[2][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 4*WG_1J) ], alpha, rC[2][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 2*WG_0I, (unsigned long) globalC1J + 5*WG_1J) ], alpha, rC[2][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 0*WG_1J) ], alpha, rC[3][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 1*WG_1J) ], alpha, rC[3][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 2*WG_1J) ], alpha, rC[3][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 3*WG_1J) ], alpha, rC[3][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 4*WG_1J) ], alpha, rC[3][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 3*WG_0I, (unsigned long) globalC1J + 5*WG_1J) ], alpha, rC[3][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 4*WG_0I, (unsigned long) globalC1J + 0*WG_1J) ], alpha, rC[4][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 4*WG_0I, (unsigned long) globalC1J + 1*WG_1J) ], alpha, rC[4][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 4*WG_0I, (unsigned long) globalC1J + 2*WG_1J) ], alpha, rC[4][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 4*WG_0I, (unsigned long) globalC1J + 3*WG_1J) ], alpha, rC[4][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 4*WG_0I, (unsigned long) globalC1J + 4*WG_1J) ], alpha, rC[4][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 4*WG_0I, (unsigned long) globalC1J + 5*WG_1J) ], alpha, rC[4][5], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 5*WG_0I, (unsigned long) globalC1J + 0*WG_1J) ], alpha, rC[5][0], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 5*WG_0I, (unsigned long) globalC1J + 1*WG_1J) ], alpha, rC[5][1], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 5*WG_0I, (unsigned long) globalC1J + 2*WG_1J) ], alpha, rC[5][2], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 5*WG_0I, (unsigned long) globalC1J + 3*WG_1J) ], alpha, rC[5][3], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 5*WG_0I, (unsigned long) globalC1J + 4*WG_1J) ], alpha, rC[5][4], beta)
  TYPE_MAD_WRITE( C[ GLOBAL_C( (unsigned long) globalC0I + 5*WG_0I, (unsigned long) globalC1J + 5*WG_1J) ], alpha, rC[5][5], beta)

}




)";


  size_t global_work_size = (gg.m*gg.n) / 36;
  size_t local_work_size = 256;
  
  /* return { {uses_a, uses_b, uses_c, uses_workspace, uses_alpha, uses_beta } , 
   * ss.str(), kernelname, dp.global_work_size, dp.n_work_items_per_workgroup}; */

   //return { {true, true, true, false, true, true } , , "tensile_kernel", global_work_size, local_work_size};
  
  return { {true, true, true, false, true, true } , ss.str(), "CT_SSSSS_Cij_Sk_Aik_Bkj_i16x6f_j16x6s_nl6x1_k16_O2", global_work_size, local_work_size};
}

}
}
