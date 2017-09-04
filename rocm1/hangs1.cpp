/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

// this cpp file was generated with function standalone::make of MIOpenGEMM, for
// geometry : tC0_tA0_tB1_colMaj1_m127_n127_k127_lda127_ldb127_ldc127_ws0_f32
// hyperparams : A_MIC1_PAD2_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC4_PAD1_PLU0_LIW1_MIW0_WOS0_VEW2__C_UNR64_GAL3_PUN1_ICE1_IWI0_SZT0_NAW16_UFO0_MAC64_SKW9_AFI1_MIA0
 
 
#include <CL/cl.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>

// print when not CL_SUCCESS
void checkstatus(cl_int x, std::string where)
{
  if (x != CL_SUCCESS)
  {
    std::cout << "\n" << where << " : exit status " << x << '.' << std::endl;
  }
}

int main()
{ 
  size_t a_n_elms = 16129;
  size_t b_n_elms = 16129;
  size_t c_n_elms = 16129;


  std::vector <float> a_init(a_n_elms);
  std::vector <float> b_init(b_n_elms);
  std::vector <float> c_init(c_n_elms);
  std::vector <float> c_final(c_n_elms);


  srand(1011);
  for (auto & x : a_init){ 
    x = 1. - 2. * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  for (auto & x : b_init){ 
    x = 1. - 2. * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }  

 for (auto & x : c_init){ 
    x = 1. - 2. * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }    
  

  std::string source_str = R"(




/* this kernel was generated for starting geometry : */
/* tC0_tA0_tB1_colMaj1_m127_n127_k127_lda127_ldb127_ldc127_ws0_f32*/
#define __K__ 127
#define TFLOAT  float
#define DOES_BETA_C_INC 1
#define DOES_ALPHA_A_B_INC 1

/* A note on how transposes isColMajor  effect the kernel generated: very little. */
/* just via STRIDE_PLL_K_{A,B}, STRIDE_PERP_K_{A,B}, STRIDE_PLL_M_C, STRIDE_PERP_M_C */



/* ********************************** specific to A *************************************** */
/* macro tiles define the pattern of C that workgroups (threads with shared local memory) process */
#define MACRO_TILE_LENGTH_A 16
/* number of elements in load block : MACRO_TILE_LENGTH_A * UNROLL */
#define N_ELEMENTS_IN_A_UNROLL 1024
/* number of groups covering M / MACRO_TILE_LENGTH_A + (PRESHIFT_FINAL_TILE_A != MACRO_TILE_LENGTH_A) */
#define N_GROUPS_A 8
/* 1 + (M - 1) % MACRO_TILE_LENGTH_A. somewhere in 1 ... MACRO_TILE_LENGTH_A  */ 
#define PRESHIFT_FINAL_TILE_A 15
/* strides parallel to k (unroll) in A. MACRO_STRIDE_A is between unroll tiles, STRIDE_A is within unroll tiles  */
#define STRIDE_PLL_K_A 127
#define MACRO_STRIDE_PLL_K_A 127
#define STRIDE_PERP_K_A 1
#define MACRO_STRIDE_PERP_K_A 1
/* vector float type */
#define TVFLOATA float
/* vector width */
#define VEW_A  1
/* micro tiles define the pattern of C that individual threads process */
#define MICRO_TILE_LENGTH_A 1
/* the amount of padding of A in LDS (local) memory, to avoid bank comflicts */
#define PAD_LDS_A  2
/* whether loading of A from global should try to be long in direction of unroll (1) or perpendicular to it (0) */
#define WORK_ITEM_LOAD_A_PLL_TO_UNROLL 0
/* MACRO_TILE_LENGTH_A + PAD_LDS_A : */
#define MACRO_TILE_LENGTH_A_AND_PAD 18
/* MACRO_TILE_LENGTH_A_AND_PAD * UNROLL : */
#define N_ELEMENTS_IN_PADDED_A_UNROLL 1152
/* N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP : */
#define N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM 16
/* MACRO_TILE_LENGTH_A / MICRO_TILE_LENGTH_A : */
#define N_MICRO_IN_MACRO_A  16
/* MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM : */
#define MICRO_A_TILE_PLL_UNROLL 1 
#define MICRO_A_TILE_PERP_UNROLL 16
/* MACRO_TILE_LENGTH_A / MICRO_A_TILE_PLL_UNROLL : */
#define N_MICRO_A_TILES_PLL_UNROLL 64 
/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if the load tiles are truly contiguous tiles (0) */
#define LOAD_TO_LDS_INTERWOVEN_A 0
/* Whether micro tile being processed by a compute item is interwoven with other micro tiles (ala Cobalt, (1)) or if the micro tiles are contiguous in C */
#define C_MICRO_TILES_INTERWOVEN_A 0
/* depending on whether loads to c are interwoven, set as MIW == 0 ? 1 : N_MICRO_IN_MACRO_A */
#define C_INTERWEAVE_STRIDE_A 1


/* ********************************** specific to B *************************************** */
#define MACRO_TILE_LENGTH_B 16
#define N_ELEMENTS_IN_B_UNROLL 1024
#define N_GROUPS_B 8
#define PRESHIFT_FINAL_TILE_B 15
#define STRIDE_PLL_K_B 127
#define MACRO_STRIDE_PLL_K_B 127
#define STRIDE_PERP_K_B 1
#define MACRO_STRIDE_PERP_K_B 1
#define TVFLOATB float2
#define VEW_B  2
#define MICRO_TILE_LENGTH_B 4
#define PAD_LDS_B  1
#define WORK_ITEM_LOAD_B_PLL_TO_UNROLL 0
#define MACRO_TILE_LENGTH_B_AND_PAD 18
#define N_ELEMENTS_IN_PADDED_B_UNROLL 1152
#define N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM 16
#define N_MICRO_IN_MACRO_B  4
#define MICRO_B_TILE_PLL_UNROLL 1 
#define MICRO_B_TILE_PERP_UNROLL 16
#define N_MICRO_B_TILES_PLL_UNROLL 64 
#define LOAD_TO_LDS_INTERWOVEN_B 1
#define C_MICRO_TILES_INTERWOVEN_B 0
#define C_INTERWEAVE_STRIDE_B 1


/* integer types for navigating each of the memory buffers */
#define TINTA ushort
#define TINTB ushort
#define TINTC ushort
#define TINTW ushort

/* type for integer in inner most loops (probably inlined anyway)  */
#define TSHORT ushort

/* type for integers which never exceeds __K__ + UNROLL (for UFO case) */
#define TINTK ushort

/* ********************************** common to A and B *************************************** */
/* whether or not to shimmy the starting k, in an attempt to avoid cache line overuse for cases where lda/ldb are powers of 2 */
/* if 0, no shimmying. if 1, instead of starting at k = 0 workgroups start at some negative offset dependent on work group id */
/* in the same way as the final unroll populates LDS with zeros in k mod UNROLL != 0, the initial negative indices here populate with 0 */
#define UNROLL_FOR_OFFSET 0
/* How much a workgroup loads (global -> LDS) in the k-direction at each iteration of the outer-most loop */
#define UNROLL 64
/* whether or not this kernel uses the edge trick (SC17 submission) */
/* this precompiler defn has no direct influence on the running the kernel, implementation already done in make_kernel.py */
#define EDGETRICK 1
/* the number of work items working on the same c element. if this is 1, there will be just one thread doing all k multiply-adds, */
/* otherwise if it is greater than 1, each thread will be computing ~ k / N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added at the end */ 
#define N_WORK_ITEMS_PER_C_ELM 1
/* define the way in which work groups are assigned to tiles */
/* 1 : column-by-column
 * 2 : row-by-row 
 * 3 : by rows within super-column  */
#define GROUP_ALLOCATION 3
/* this variable is declared because we have GROUP_ALLOCATION type 3. */
/* It should define how many workgroups we expect to have active simulantaneuosly. */
#define N_TARGET_ACTIVE_WORKGROUPS 16
/* Whether to use the unroll pragma to encourage the compiler to unroll certain loops */
/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */
#define PRAGMA_UNROLL_FORLOOPS 1
/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into registers, as compared to doing the math. */
/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */
#define N_PREFETCH_FOR_REGISTER_LOAD 0
/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into LDS, as compared to the unroll loop */
/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */
#define N_PREFETCH_FOR_LDS_LOAD 0
#define MACRO_TILE_AREA 256
#define MICRO_TILE_AREA 4
#define N_WORK_ITEMS_PER_WORKGROUP  64
/* two more parameters, which do dot have an effect the running of this kernel (used in enqueuing) */
/* the total number of work groups this kernel will use (recall m,n,k are fixed) */ 
/* N_WORK_ITEMS_PER_C_ELM * ((M/MACRO_TILE_LENGTH_A) + (M%MACRO_TILE_LENGTH_A != 0)) * ((N/MACRO_TILE_LENGTH_B) + (N%MACRO_TILE_LENGTH_B != 0)) */ 
#define N_WORK_GROUPS 64
/* the global work size, ie the total mumber of work items (threads) which will run */
 /* N_WORK_GROUPS * N_WORK_ITEMS_PER_WORKGROUP */ 
#define GLOBAL_WORK_SIZE 4096
#define STRIDE_PLL_M_C 1
#define STRIDE_PLL_N_C 127


/* This variable defines the width of super-columns (we have GROUP_ALLOCATION 3). It is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) */
#define SUPER_COLUMN_WIDTH 4
/* LAST_SUPER_COLUMN_WIDTH : N_GROUPS_B % SUPER_COLUMN_WIDTH  */
#define LAST_SUPER_COLUMN_WIDTH 0


__attribute__((reqd_work_group_size(64,1, 1)))
__kernel void miog_betac_alphaab
( 
__global const TFLOAT * restrict a, 
const size_t a_offset,
__global const TFLOAT * restrict b, 
const size_t b_offset,
__global TFLOAT       *          c, 
const size_t c_offset,
const TFLOAT alpha,
const TFLOAT beta)

{
  
  
  
  /* In OpenCL, host code does not have access to raw data pointers. */
  /* Host code works with cl_mem objects, which encapsulate and hide raw points. */
  /* For this reason, host code CANNOT simply increment pointers to data, */
  /* as one can do with pointers for CPU gemm, or cublas gemm.*/
  
  c += c_offset;
  const TSHORT local_id = get_local_id(0);
  
  const TINTC group_id_xy = get_group_id(0);
  /* Define which part of the C macro-tile this thread will process: BYA*/
  
  const TSHORT micro_id_a = local_id % N_MICRO_IN_MACRO_A;
  const TSHORT micro_id_b = local_id / N_MICRO_IN_MACRO_A;
  
  
  /* GROUP_ALLOCATION = 3 : allocation examples
  * (if SUPER_COLUMN_WIDTH is 8, m = 3, and N_WORK_ITEMS_PER_C_ELM is 1) is done as follows
  * |0   1  2  3  4  5  6  7| 24 25 26
  * |8   9 10 11 12 13 14 15| 27 28 29
  * |16 17 18 19 20 21 21 23| 30 31 32
  *              
  * if SUPER_COLUMN_WIDTH is 2 and N_WORK_ITEMS_PER_ELM is 3 it is done as follows
  * | (0,   1,  2)  (3,  4,  5 )    |    
  * | (6,   7,  8)  (9,  10, 11)    |    ...
  * | (12, 13, 14)  (15, 16, 17)    |
  *                .
  *                .
  * where the integers are work group numbers
  * */  
  TINTB group_id_b;
  TINTA group_id_a;
  TINTC wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*N_GROUPS_A);
  
  group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
  group_id_a = (group_id_xy / SUPER_COLUMN_WIDTH) % N_GROUPS_A;
  
  int n_unrolls_remaining = __K__ / UNROLL;
  
  /* ************* A setup *************** */
  /* LDS memory */
  __local TVFLOATA localA[N_ELEMENTS_IN_PADDED_A_UNROLL/VEW_A];
  /* jumping pointer to locate the LDS to load into register memory */
  __local const TVFLOATA * lA;
  /* register memory */ 
  TFLOAT rA[MICRO_TILE_LENGTH_A];
  /* Define which part of the C macro-tile this thread will process (% / or / % ? doesn't seem to make much difference) */
  TINTA write_macro_tile_start_a = group_id_a*MACRO_TILE_LENGTH_A; 
  /* tile on edge : pulling it in so no C overflow */
  if (group_id_a == N_GROUPS_A - 1){
    write_macro_tile_start_a -= (MACRO_TILE_LENGTH_A - PRESHIFT_FINAL_TILE_A);
  }
  const TINTA write_start_a = write_macro_tile_start_a + micro_id_a*(MICRO_TILE_LENGTH_A/1);
  
  
  
  a += a_offset;
  /* Define what of A this thread will load from unroll tile in global to LDS (% / or / % ? looks like no difference ) */
  const TINTA pll_unroll_a_load_id = local_id % N_MICRO_A_TILES_PLL_UNROLL;
  const TINTA perp_unroll_a_load_id = local_id / N_MICRO_A_TILES_PLL_UNROLL;
  /* Define which part of A this thread will read from (% / or / % ? doesn't seem to make much difference) */
  TINTA read_macro_tile_start_a = group_id_a*MACRO_TILE_LENGTH_A; 
  /* tile on edge and A is not normal form: pulling in read zone so no C overflow */
  if (group_id_a == N_GROUPS_A - 1){
    read_macro_tile_start_a -= (MACRO_TILE_LENGTH_A - PRESHIFT_FINAL_TILE_A);
  }
  /* move to corner of the region required by the macro tile */
  a += read_macro_tile_start_a*MACRO_STRIDE_PERP_K_A;
  /* make the micro adjustments (A) for the thread, getting ready to load */
  const TINTA a_offset_pll_unroll = MICRO_A_TILE_PLL_UNROLL * pll_unroll_a_load_id;
  const TINTA a_offset_perp_unroll = MICRO_A_TILE_PERP_UNROLL * perp_unroll_a_load_id;
  /* the offset in vector-floats perp to unroll */
  const TINTA a_offset_perp_unroll_v = MICRO_A_TILE_PERP_UNROLL/VEW_A * perp_unroll_a_load_id;
  a += STRIDE_PLL_K_A * a_offset_pll_unroll;
  /* vectorised version of a */
  const __global TVFLOATA * a_vec = (const __global TVFLOATA * )a;
  a_vec += STRIDE_PERP_K_A * a_offset_perp_unroll_v;
  
  
  
  /* ************* B setup *************** */
  __local TVFLOATB localB[N_ELEMENTS_IN_PADDED_B_UNROLL/VEW_B];
  __local const TVFLOATB * lB;
  TFLOAT rB[MICRO_TILE_LENGTH_B];
  TINTB write_macro_tile_start_b = group_id_b*MACRO_TILE_LENGTH_B; 
  if (group_id_b == N_GROUPS_B - 1){
    write_macro_tile_start_b -= (MACRO_TILE_LENGTH_B - PRESHIFT_FINAL_TILE_B);
  }
  const TINTB write_start_b = write_macro_tile_start_b + micro_id_b*(MICRO_TILE_LENGTH_B/1);
  
  
  
  b += b_offset;
  const TINTB pll_unroll_b_load_id = local_id % N_MICRO_B_TILES_PLL_UNROLL;
  const TINTB perp_unroll_b_load_id = local_id / N_MICRO_B_TILES_PLL_UNROLL;
  TINTB read_macro_tile_start_b = group_id_b*MACRO_TILE_LENGTH_B; 
  if (group_id_b == N_GROUPS_B - 1){
    read_macro_tile_start_b -= (MACRO_TILE_LENGTH_B - PRESHIFT_FINAL_TILE_B);
  }
  b += read_macro_tile_start_b*MACRO_STRIDE_PERP_K_B;
  const TINTB b_offset_pll_unroll =  pll_unroll_b_load_id;
  const TINTB b_offset_perp_unroll =  perp_unroll_b_load_id;
  const TINTB b_offset_perp_unroll_v =  perp_unroll_b_load_id;
  b += STRIDE_PLL_K_B * b_offset_pll_unroll;
  const __global TVFLOATB * b_vec = (const __global TVFLOATB * )b;
  b_vec += STRIDE_PERP_K_B * b_offset_perp_unroll_v;
  
  
  
  
  /* register memory for C */
 TFLOAT rC[MICRO_TILE_LENGTH_A][MICRO_TILE_LENGTH_B] = {{0.}};
  
  
  while (n_unrolls_remaining > 0){
    
    /* load data from a into LDS */
    #pragma unroll
    for (TINTA mu_perp_i = 0; mu_perp_i < MICRO_A_TILE_PERP_UNROLL/VEW_A; ++mu_perp_i) {
      #pragma unroll
      for (TINTA mu_pll_i = 0; mu_pll_i < MICRO_A_TILE_PLL_UNROLL; ++mu_pll_i) {
        localA[MACRO_TILE_LENGTH_A_AND_PAD/VEW_A*(a_offset_pll_unroll + mu_pll_i) + a_offset_perp_unroll_v + mu_perp_i] = 
        a_vec[(mu_pll_i*STRIDE_PLL_K_A + VEW_A*mu_perp_i*STRIDE_PERP_K_A)/VEW_A];
      }
    }
    a_vec += (STRIDE_PLL_K_A*UNROLL)/VEW_A;
    
    
    /* load data from b into LDS */
    #pragma unroll
    for (TINTB mu_perp_i = 0; mu_perp_i < MACRO_TILE_LENGTH_B/VEW_B; mu_perp_i += MACRO_TILE_LENGTH_B/MICRO_B_TILE_PERP_UNROLL) {
      #pragma unroll
      for (TINTB mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_B_TILE_PLL_UNROLL) {
        localB[MACRO_TILE_LENGTH_B_AND_PAD/VEW_B*(b_offset_pll_unroll + mu_pll_i) + b_offset_perp_unroll_v + mu_perp_i] = 
        b_vec[(mu_pll_i*STRIDE_PLL_K_B + VEW_B*mu_perp_i*STRIDE_PERP_K_B)/VEW_B];
      }
    }
    b_vec += (STRIDE_PLL_K_B*UNROLL)/VEW_B;
    
    
    /* make sure all loads from LDS memory have completed */
    barrier(CLK_LOCAL_MEM_FENCE); 
    
    lA = localA + micro_id_a*MICRO_TILE_LENGTH_A/VEW_A;
    lB = localB + micro_id_b*MICRO_TILE_LENGTH_B/VEW_B;
    
    for (TSHORT u = 0; u < UNROLL; ++u){
      
      #pragma unroll
      for (TSHORT i = 0; i < MICRO_TILE_LENGTH_A/VEW_A; ++i){
        rA[i] = lA[i*C_INTERWEAVE_STRIDE_A];
      }
      lA += MACRO_TILE_LENGTH_A_AND_PAD/VEW_A;
      
      #pragma unroll
      for (TSHORT i = 0; i < MICRO_TILE_LENGTH_B/VEW_B; ++i){
        rB[VEW_B*i + 0] = lB[i*C_INTERWEAVE_STRIDE_B].s0;
        rB[VEW_B*i + 1] = lB[i*C_INTERWEAVE_STRIDE_B].s1;
      }
      lB += MACRO_TILE_LENGTH_B_AND_PAD/VEW_B;
      
      #pragma unroll
      for (TSHORT dima = 0; dima < MICRO_TILE_LENGTH_A; ++dima){
        #pragma unroll
        for (TSHORT dimb = 0; dimb < MICRO_TILE_LENGTH_B; ++dimb){
          rC[dima][dimb] += rA[dima]*rB[dimb];   
        }
      }
    }
    
    /* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
    barrier(CLK_LOCAL_MEM_FENCE); 
    --n_unrolls_remaining;
  }
  
  /* *********** processing the tail *************** */
  
  TSHORT k_remaining = __K__ % UNROLL;
  
  /* load final bit of data from a into LDS, less than a full unroll (ignoring tail) */
  #pragma unroll
  for (TINTA mu_perp_i = 0; mu_perp_i < MICRO_A_TILE_PERP_UNROLL/VEW_A; ++mu_perp_i) {
    #pragma unroll
    for (TINTA mu_pll_i = 0; mu_pll_i < MICRO_A_TILE_PLL_UNROLL; ++mu_pll_i) {
      localA[MACRO_TILE_LENGTH_A_AND_PAD/VEW_A*(a_offset_pll_unroll + mu_pll_i) + a_offset_perp_unroll_v + mu_perp_i] = 
      (a_offset_pll_unroll + mu_pll_i)  < k_remaining  ? a_vec[(mu_pll_i*STRIDE_PLL_K_A + VEW_A*mu_perp_i*STRIDE_PERP_K_A)/VEW_A] : 0;
    }
  }
  
  
  /* load final bit of data from b into LDS, less than a full unroll (ignoring tail) */
  #pragma unroll
  for (TINTB mu_perp_i = 0; mu_perp_i < MACRO_TILE_LENGTH_B/VEW_B; mu_perp_i += MACRO_TILE_LENGTH_B/MICRO_B_TILE_PERP_UNROLL) {
    #pragma unroll
    for (TINTB mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_B_TILE_PLL_UNROLL) {
      localB[MACRO_TILE_LENGTH_B_AND_PAD/VEW_B*(b_offset_pll_unroll + mu_pll_i) + b_offset_perp_unroll_v + mu_perp_i] = 
      (b_offset_pll_unroll + mu_pll_i)  < k_remaining  ? b_vec[(mu_pll_i*STRIDE_PLL_K_B + VEW_B*mu_perp_i*STRIDE_PERP_K_B)/VEW_B] : 0;
    }
  }
  
  
  /* make sure all loads from LDS memory have completed */
  barrier(CLK_LOCAL_MEM_FENCE); 
  
  lA = localA + micro_id_a*MICRO_TILE_LENGTH_A/VEW_A;
  lB = localB + micro_id_b*MICRO_TILE_LENGTH_B/VEW_B;
  
  for (TSHORT u = 0; u < k_remaining; ++u){
    
    #pragma unroll
    for (TSHORT i = 0; i < MICRO_TILE_LENGTH_A/VEW_A; ++i){
      rA[i] = lA[i*C_INTERWEAVE_STRIDE_A];
    }
    lA += MACRO_TILE_LENGTH_A_AND_PAD/VEW_A;
    
    #pragma unroll
    for (TSHORT i = 0; i < MICRO_TILE_LENGTH_B/VEW_B; ++i){
      rB[VEW_B*i + 0] = lB[i*C_INTERWEAVE_STRIDE_B].s0;
      rB[VEW_B*i + 1] = lB[i*C_INTERWEAVE_STRIDE_B].s1;
    }
    lB += MACRO_TILE_LENGTH_B_AND_PAD/VEW_B;
    
    #pragma unroll
    for (TSHORT dima = 0; dima < MICRO_TILE_LENGTH_A; ++dima){
      #pragma unroll
      for (TSHORT dimb = 0; dimb < MICRO_TILE_LENGTH_B; ++dimb){
        rC[dima][dimb] += rA[dima]*rB[dimb];   
      }
    }
  }
  
  /* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
  barrier(CLK_LOCAL_MEM_FENCE); 
  
  /* *********************************************** */
  
  
  
  TINTC index;
  
  /* the case where this is not an edge tile : will write to all cells */ 
  if ((group_id_b != N_GROUPS_B - 1) && (group_id_a != N_GROUPS_A - 1)){ 
    
    /* loops for writing to c */
    #pragma unroll
    for (TINTA dimai = 0; dimai < MICRO_TILE_LENGTH_A/VEW_A; ++dimai) {
      #pragma unroll
      for (TINTA dimbi = 0; dimbi < MICRO_TILE_LENGTH_B/VEW_B; ++dimbi) {
        #pragma unroll
        for (TINTA dimai_v = 0; dimai_v < VEW_A; ++dimai_v) {
          #pragma unroll
          for (TINTB dimbi_v = 0; dimbi_v < VEW_B; ++dimbi_v) {
            TINTB dimb = dimbi*VEW_B + dimbi_v;
            TINTA dima = dimai*VEW_A + dimai_v;
            
            index =  STRIDE_PLL_M_C*(write_start_a + dima) + STRIDE_PLL_N_C*(write_start_b + dimb) ;
            c[index] *= beta;
            
            c[index] += alpha*rC[dima][dimb];
            
          }
        }
      }
    }
    
  }
  
  else{
    /* loops for writing to c */
    #pragma unroll
    for (TINTA dimai = 0; dimai < MICRO_TILE_LENGTH_A/VEW_A; ++dimai) {
      #pragma unroll
      for (TINTA dimbi = 0; dimbi < MICRO_TILE_LENGTH_B/VEW_B; ++dimbi) {
        #pragma unroll
        for (TINTA dimai_v = 0; dimai_v < VEW_A; ++dimai_v) {
          #pragma unroll
          for (TINTB dimbi_v = 0; dimbi_v < VEW_B; ++dimbi_v) {
            TINTB dimb = dimbi*VEW_B + dimbi_v;
            TINTA dima = dimai*VEW_A + dimai_v;
            
            /* catching the write cases */
            if (
            
            
            
            /* B overflow, but not A edge */
            ((write_start_b + dimb >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1)) && group_id_a  !=  (N_GROUPS_A - 1 )) ||
            
            /* A overflow, but not B edge */
            ((write_start_a + dima >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)) && group_id_b  !=  (N_GROUPS_B - 1 )) ||
            
            
            /* A edge and B edge and A overflow and B overflow */
            (
            group_id_a == (N_GROUPS_A - 1)   && 
            group_id_b == (N_GROUPS_B - 1)   && 
            write_start_b + dimb >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1) && 
            write_start_a + dima >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)
            )){
              
              index =  STRIDE_PLL_M_C*(write_start_a + dima) + STRIDE_PLL_N_C*(write_start_b + dimb) ;
              c[index] *= beta;
              
              c[index] += alpha*rC[dima][dimb];
              
            }
          }
        }
      }
    }
    
  }
}



)";


  cl_platform_id   platform_id   = nullptr;
  cl_device_id     device_id     = nullptr;
  cl_context       context       = nullptr;
  cl_command_queue command_queue = nullptr;

  cl_mem memobj_a = nullptr;
  cl_mem memobj_b = nullptr;
  cl_mem memobj_c = nullptr;
  
  size_t a_offset = 0;
  size_t b_offset = 0;
  size_t c_offset = 0;

  
  cl_program program = nullptr;
  cl_kernel  kernel  = nullptr;
  
  cl_uint    ret_num_devices;
  cl_uint    ret_num_platforms;
  cl_int     ret;

  /* Get platform/device information */
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  checkstatus(ret, "clGetPlatformIDs");

  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  checkstatus(ret, "clGetDeviceIDs");

  /* Create OpenCL Context */
  context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
  checkstatus(ret, "clCreateContext");

  /* Create Command Queue */
  std::cout << "clCreateCommandQueue.." << std::flush;
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  checkstatus(ret, "clCreateCommandQueue");
  
    // create and write to buffers 
  memobj_a = clCreateBuffer(context, CL_MEM_READ_WRITE, a_n_elms  * sizeof(float), nullptr, &ret);
  checkstatus(ret, "clCreateBuffer");
  ret = clEnqueueWriteBuffer(
  command_queue, memobj_a, CL_TRUE, 0, a_n_elms * sizeof(float), a_init.data(), 0, nullptr, nullptr);
  checkstatus(ret, "clEnqueueWriteBuffer");

  memobj_b = clCreateBuffer(context, CL_MEM_READ_WRITE, b_n_elms  * sizeof(float), nullptr, &ret);
  checkstatus(ret, "clCreateBuffer");
  ret = clEnqueueWriteBuffer(
  command_queue, memobj_b, CL_TRUE, 0, b_n_elms * sizeof(float), b_init.data(), 0, nullptr, nullptr);
  checkstatus(ret, "clEnqueueWriteBuffer");

  memobj_c = clCreateBuffer(context, CL_MEM_READ_WRITE, c_n_elms  * sizeof(float), nullptr, &ret);
  checkstatus(ret, "clCreateBuffer");
  ret = clEnqueueWriteBuffer(
  command_queue, memobj_c, CL_TRUE, 0, c_n_elms * sizeof(float), c_init.data(), 0, nullptr, nullptr);
  checkstatus(ret, "clEnqueueWriteBuffer");



  /* Create Kernel program from the read in source */
  auto   source_c_str = source_str.c_str();
  size_t source_size  = source_str.size();
  
  std::cout << "clCreateProgramWithSource.." << std::flush;
  program             = clCreateProgramWithSource(
    context, 1, (const char**)&source_c_str, (const size_t*)&source_size, &ret);
  checkstatus(ret, "clCreateProgramWithSource");

  /* Build Kernel Program */
  std::cout << "clBuildProgram.." << std::flush;
  ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
  checkstatus(ret, "clBuildProgram");

  /* Create OpenCL Kernel */ 
    kernel = clCreateKernel(program, "miog_betac_alphaab", &ret);
  checkstatus(ret, "clCreateKernel");


  /* Set OpenCL kernel argument */


  std::cout << "clSetKernelArg(s).." << std::flush;

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_a);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 1, sizeof(size_t), &a_offset);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_b);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 3, sizeof(size_t), &b_offset);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&memobj_c);
  checkstatus(ret, "clSetKernelArg");

  ret = clSetKernelArg(kernel, 5, sizeof(size_t), &c_offset);
  checkstatus(ret, "clSetKernelArg");
  float alpha = 0.41569302918234590782;
  ret = clSetKernelArg(kernel, 6, sizeof(float), &alpha);
  checkstatus(ret, "clSetKernelArg");
  float beta = 0.27353934093480936074;
  ret = clSetKernelArg(kernel, 7, sizeof(float), &beta);
  checkstatus(ret, "clSetKernelArg");
  size_t local_work_size = 64;
  size_t global_work_size = 4096;

  
  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "clEnqueueNDRangeKernel.." << std::flush;
  ret = clEnqueueNDRangeKernel(
    command_queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
  checkstatus(ret, "clEnqueueNDRangeKernel");

  std::cout << "clFinish.." << std::flush;
  ret = clFinish(command_queue);
  checkstatus(ret, "clFinish");
  

  std::cout << "clEnqueueReadBuffer.." << std::flush;
  ret = clEnqueueReadBuffer(command_queue,
                                   memobj_c,
                                   CL_TRUE,
                                   0,
                                   sizeof(float)*c_n_elms,
                                   c_final.data(),
                                   0,
                                   nullptr,
                                   nullptr);
  
 checkstatus(ret, "clEnqueueReadBuffer");

  std::cout << "clFinish.." << std::flush;
  ret = clFinish(command_queue);
  checkstatus(ret, "clFinish");
 

    
  
  std::cout << "done." << std::endl;
  
  float sum_final = std::accumulate(c_final.begin(), c_final.end(), 0.f);
  float sum_init  = std::accumulate(c_init.begin(), c_init.end(), 0.f);

  // (precomputed in standalone.cpp using OpenBLAS)
   float sum_final_cpu = 20.900096893310546875;
  float error = sum_final_cpu - sum_final;

  
  std::cout << "sum of initial c = " << std::setprecision(20) << sum_init << std::endl;
  std::cout << "sum of final c  gpu = " << std::setprecision(20) << sum_final << std::endl;
  std::cout << "sum of final on cpu = " <<  20.900096893310546875  << std::endl; 
  std::cout << "(cpu - gpu )/cpu = " <<  std::setprecision(10) <<  error << std::endl; 

   
  

  auto                         end             = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms           = end - start;
  float                        elapsed_seconds = fp_ms.count();
  std::cout << "elapsed seconds : " << elapsed_seconds << std::endl;

  /* Finalization */
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj_a);
  ret = clReleaseMemObject(memobj_b);
  ret = clReleaseMemObject(memobj_c);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);


  
  return 0;
}

  
  
  
