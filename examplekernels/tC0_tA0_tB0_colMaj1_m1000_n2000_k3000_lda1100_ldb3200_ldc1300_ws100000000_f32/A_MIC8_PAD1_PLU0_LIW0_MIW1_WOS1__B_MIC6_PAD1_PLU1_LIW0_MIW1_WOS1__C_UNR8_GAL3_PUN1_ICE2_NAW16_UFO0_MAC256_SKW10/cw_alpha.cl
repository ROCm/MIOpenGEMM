
/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
/* ****************************************************************************
* This alphaab kernel string was generated on Thu Jun  8 17:13:31 2017
**************************************************************************** */

/* this kernel was generated for starting geometry : */
/* tC0_tA0_tB0_colMaj1_m1000_n2000_k3000_lda1100_ldb3200_ldc1300_ws100000000_f32*/
#define __K__ 3000
#define TFLOAT  float
#define DOES_BETA_C_INC 0
#define DOES_ALPHA_A_B_INC 1

/* A note on how transposes isColMajor  effect the kernel generated: very little. */
/* just via STRIDE_PLL_K_{A,B}, STRIDE_PERP_K_{A,B}, STRIDE_PLL_M_C, STRIDE_PERP_M_C */



/* ********************************** specific to A *************************************** */
/* macro tiles define the pattern of C that workgroups (threads with shared local memory) process */
#define MACRO_TILE_LENGTH_A 128
/* number of elements in load block : MACRO_TILE_LENGTH_A * UNROLL */
#define N_ELEMENTS_IN_A_UNROLL 1024
/* number of groups covering M / MACRO_TILE_LENGTH_A + (PRESHIFT_FINAL_TILE_A != MACRO_TILE_LENGTH_A) */
#define N_GROUPS_A 8
/* 1 + (M - 1) % MACRO_TILE_LENGTH_A. somewhere in 1 ... MACRO_TILE_LENGTH_A  */ 
#define PRESHIFT_FINAL_TILE_A 104
/* strides parallel to k (unroll) in A. MACRO_STRIDE_A is between unroll tiles, STRIDE_A is within unroll tiles  */
#define STRIDE_PLL_K_A 1011
#define MACRO_STRIDE_PLL_K_A 1011
#define STRIDE_PERP_K_A 1
#define MACRO_STRIDE_PERP_K_A 1
/* micro tiles define the pattern of C that individual threads process */
#define MICRO_TILE_LENGTH_A 8
/* the amount of padding of A in LDS (local) memory, to avoid bank comflicts */
#define PAD_LDS_A  1
/* whether loading of A from global should try to be long in direction of unroll (1) or perpendicular to it (0) */
#define WORK_ITEM_LOAD_A_PLL_TO_UNROLL 0
/* MACRO_TILE_LENGTH_A + PAD_LDS_A : */
#define MACRO_TILE_LENGTH_A_AND_PAD 129
/* MACRO_TILE_LENGTH_A_AND_PAD * UNROLL : */
#define N_ELEMENTS_IN_PADDED_A_UNROLL 1032
/* N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP : */
#define N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM 4
/* MACRO_TILE_LENGTH_A / MICRO_TILE_LENGTH_A : */
#define N_MICRO_IN_MACRO_A  16
/* MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM : */
#define MICRO_A_TILE_PLL_UNROLL 1 
#define MICRO_A_TILE_PERP_UNROLL 4
/* MACRO_TILE_LENGTH_A / MICRO_A_TILE_PLL_UNROLL : */
#define N_MICRO_A_TILES_PLL_UNROLL 8 
/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if the load tiles are truly contiguous tiles (0) */
#define LOAD_TO_LDS_INTERWOVEN_A 0
/* Whether micro tile being processed by a compute item is interwoven with other micro tiles (ala Cobalt, (1)) or if the micro tiles are contiguous in C */
#define C_MICRO_TILES_INTERWOVEN_A 1
/* depending on whether loads to c are interwoven, set as MIW == 0 ? 1 : N_MICRO_IN_MACRO_A */
#define C_INTERWEAVE_STRIDE_A 16
/* global memory offset, depends on type of copy of both a,b */
#define GLOBAL_OFFSET_A 0

/* ********************************** specific to B *************************************** */
#define MACRO_TILE_LENGTH_B 96
#define N_ELEMENTS_IN_B_UNROLL 768
#define N_GROUPS_B 21
#define PRESHIFT_FINAL_TILE_B 80
#define STRIDE_PLL_K_B 1
#define MACRO_STRIDE_PLL_K_B 1
#define STRIDE_PERP_K_B 3014
#define MACRO_STRIDE_PERP_K_B 3014
#define MICRO_TILE_LENGTH_B 6
#define PAD_LDS_B  1
#define WORK_ITEM_LOAD_B_PLL_TO_UNROLL 1
#define MACRO_TILE_LENGTH_B_AND_PAD 97
#define N_ELEMENTS_IN_PADDED_B_UNROLL 776
#define N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM 3
#define N_MICRO_IN_MACRO_B  16
#define MICRO_B_TILE_PLL_UNROLL 1 
#define MICRO_B_TILE_PERP_UNROLL 3
#define N_MICRO_B_TILES_PLL_UNROLL 8 
#define LOAD_TO_LDS_INTERWOVEN_B 0
#define C_MICRO_TILES_INTERWOVEN_B 1
#define C_INTERWEAVE_STRIDE_B 16
#define GLOBAL_OFFSET_B 3033000

/* ********************************** common to A and B *************************************** */
/* whether or not to shimmy the starting k, in an attempt to avoid cache line overuse for cases where lda/ldb are powers of 2 */
/* if 0, no shimmying. if 1, instead of starting at k = 0 workgroups start at some negative offset dependent on work group id */
/* in the same way as the final unroll populates LDS with zeros in k mod UNROLL != 0, the initial negative indices here populate with 0 */
#define UNROLL_FOR_OFFSET 0
/* How much a workgroup loads (global -> LDS) in the k-direction at each iteration of the outer-most loop */
#define UNROLL 8
/* whether or not this kernel uses the edge trick (SC17 submission) */
/* this precompiler defn has no direct influence on the running the kernel, implementation already done in make_kernel.py */
#define EDGETRICK 1
/* the number of work items working on the same c element. if this is 1, there will be just one thread doing all k multiply-adds, */
/* otherwise if it is greater than 1, each thread will be computing ~ k / N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added at the end */ 
#define N_WORK_ITEMS_PER_C_ELM 2
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
#define MACRO_TILE_AREA 12288
#define MICRO_TILE_AREA 48
#define N_WORK_ITEMS_PER_WORKGROUP  256
/* two more parameters, which do dot have an effect the running of this kernel (used in enqueuing) */
/* the total number of work groups this kernel will use (recall m,n,k are fixed) */ 
/* N_WORK_ITEMS_PER_C_ELM * ((M/MACRO_TILE_LENGTH_A) + (M%MACRO_TILE_LENGTH_A != 0)) * ((N/MACRO_TILE_LENGTH_B) + (N%MACRO_TILE_LENGTH_B != 0)) */ 
#define N_WORK_GROUPS 336
/* the global work size, ie the total mumber of work items (threads) which will run */ 
/* N_WORK_GROUPS * N_WORK_ITEMS_PER_WORKGROUP */ 
#define GLOBAL_WORK_SIZE 86016
#define STRIDE_PLL_M_C 1
#define STRIDE_PLL_N_C 1300
/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL 16 // N_WORK_ITEMS_PER_C_ELM*UNROLL

/* This variable defines the width of super-columns (we have GROUP_ALLOCATION 3). It is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) */
#define SUPER_COLUMN_WIDTH 2
/* LAST_SUPER_COLUMN_WIDTH : N_GROUPS_B % SUPER_COLUMN_WIDTH  */
#define LAST_SUPER_COLUMN_WIDTH 1


__attribute__((reqd_work_group_size(256,1, 1)))
__kernel void tg_alphaab
(

__global TFLOAT       *          c, 
const unsigned c_offset,
__global const TFLOAT * restrict w,
const unsigned w_offset,
const TFLOAT alpha)

{
  
  
  
  /* In OpenCL, host code does not have access to raw data pointers. */
  /* Host code works with cl_mem objects, which encapsulate and hide raw points. */
  /* For this reason, host code CANNOT simply increment pointers to data, */
  /* as one can do with pointers for CPU gemm, or cublas gemm.*/
  
  c += c_offset;
  const unsigned local_id = get_local_id(0);
  
  const unsigned group_id = get_group_id(0);
  const unsigned group_id_xy = group_id / N_WORK_ITEMS_PER_C_ELM;
  const unsigned group_id_z = group_id % N_WORK_ITEMS_PER_C_ELM;
  
  /* Define which part of the C macro-tile this thread will process (% / or / % ? doesn't seem to make much difference) */
  const unsigned micro_id_a = local_id % N_MICRO_IN_MACRO_A;
  const unsigned micro_id_b = local_id / N_MICRO_IN_MACRO_A;
  
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
  unsigned group_id_b;
  unsigned group_id_a;
  unsigned wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*N_GROUPS_A);
  
  if (group_id_xy < (N_GROUPS_B - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A){
    group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
    group_id_a = (group_id_xy / SUPER_COLUMN_WIDTH) % N_GROUPS_A;
  }
  else{
    group_id_b = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % LAST_SUPER_COLUMN_WIDTH;
    group_id_a = (group_id_xy  - (N_GROUPS_B - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_A) / LAST_SUPER_COLUMN_WIDTH;
  }
  
  
  /* *************************** A setup ************************** */
  /* LDS memory */
  __local TFLOAT localA[N_ELEMENTS_IN_PADDED_A_UNROLL];
  /* jumping pointer to locate the LDS to load into register memory */
  __local const TFLOAT * lA;
  /* register memory */ 
  TFLOAT rA[MICRO_TILE_LENGTH_A];
  /* Define which part of the C macro-tile this thread will process (% / or / % ? doesn't seem to make much difference) */
  unsigned write_macro_tile_start_a = group_id_a*MACRO_TILE_LENGTH_A; 
  /* tile on edge : pulling it in so no C overflow */
  if (group_id_a == N_GROUPS_A - 1){
    write_macro_tile_start_a -= (MACRO_TILE_LENGTH_A - PRESHIFT_FINAL_TILE_A);
  }
  const unsigned write_start_a = write_macro_tile_start_a + micro_id_a*1;
  
  
  
  /* from workspace */
  const TFLOAT * restrict a = w + w_offset + GLOBAL_OFFSET_A;
  /* Define what of A this thread will load from unroll tile in global to LDS (% / or / % ? looks like no difference ) */
  const unsigned pll_unroll_a_load_id = local_id % N_MICRO_A_TILES_PLL_UNROLL;
  const unsigned perp_unroll_a_load_id = local_id / N_MICRO_A_TILES_PLL_UNROLL;
  /* Define which part of A this thread will read from process (% / or / % ? doesn't seem to make much difference) */
  unsigned read_macro_tile_start_a = group_id_a*MACRO_TILE_LENGTH_A; 
  /* tile on edge and A is not normal form: pulling in read zone so no C overflow */
  if (group_id_a == N_GROUPS_A - 1){
    read_macro_tile_start_a -= (MACRO_TILE_LENGTH_A - PRESHIFT_FINAL_TILE_A);
  }
  /* move to corner of the region required by the macro tile */
  a += read_macro_tile_start_a*MACRO_STRIDE_PERP_K_A;
  /* a points to top left of region required, but this work group  */
  /* might not process the whole of a. So turn 90 and move to the start for this wg */
  a += UNROLL*group_id_z*STRIDE_PLL_K_A;
  /* make the micro adjustments (A) for the thread, getting ready to load */
  const unsigned a_offset_pll_unroll = MICRO_A_TILE_PLL_UNROLL * pll_unroll_a_load_id;
  const unsigned a_offset_perp_unroll = MICRO_A_TILE_PERP_UNROLL * perp_unroll_a_load_id;
  a += STRIDE_PLL_K_A * a_offset_pll_unroll;
  a += STRIDE_PERP_K_A * a_offset_perp_unroll;
  
  
  
  /* *************************** B setup ************************** */
  __local TFLOAT localB[N_ELEMENTS_IN_PADDED_B_UNROLL];
  __local const TFLOAT * lB;
  TFLOAT rB[MICRO_TILE_LENGTH_B];
  unsigned write_macro_tile_start_b = group_id_b*MACRO_TILE_LENGTH_B; 
  if (group_id_b == N_GROUPS_B - 1){
    write_macro_tile_start_b -= (MACRO_TILE_LENGTH_B - PRESHIFT_FINAL_TILE_B);
  }
  const unsigned write_start_b = write_macro_tile_start_b + micro_id_b*1;
  
  
  
  const TFLOAT * restrict b = w + w_offset + GLOBAL_OFFSET_B;
  const unsigned pll_unroll_b_load_id = local_id % N_MICRO_B_TILES_PLL_UNROLL;
  const unsigned perp_unroll_b_load_id = local_id / N_MICRO_B_TILES_PLL_UNROLL;
  unsigned read_macro_tile_start_b = group_id_b*MACRO_TILE_LENGTH_B; 
  if (group_id_b == N_GROUPS_B - 1){
    read_macro_tile_start_b -= (MACRO_TILE_LENGTH_B - PRESHIFT_FINAL_TILE_B);
  }
  b += read_macro_tile_start_b*MACRO_STRIDE_PERP_K_B;
  b += UNROLL*group_id_z*STRIDE_PLL_K_B;
  const unsigned b_offset_pll_unroll = MICRO_B_TILE_PLL_UNROLL * pll_unroll_b_load_id;
  const unsigned b_offset_perp_unroll = MICRO_B_TILE_PERP_UNROLL * perp_unroll_b_load_id;
  b += STRIDE_PLL_K_B * b_offset_pll_unroll;
  b += STRIDE_PERP_K_B * b_offset_perp_unroll;
  
  
  
  
  /* register memory for C */
 TFLOAT rC[MICRO_TILE_LENGTH_A][MICRO_TILE_LENGTH_B] = {{0.}};
  
  /* a certain number of work groups process one more unroll. Note that with UFO = 1, this depends on column */
  const int n_work_groups_with_1_more = (__K__ % G_UNROLL) / UNROLL; 
  
  /* branching between work groups : some wgs have 1 more unroll to process. */
  int n_unrolls_remaining = (__K__ / G_UNROLL) +  (group_id_z < n_work_groups_with_1_more);
  
  while (n_unrolls_remaining > 0){
    
    /* load data from a into LDS */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MICRO_A_TILE_PERP_UNROLL; ++mu_perp_i) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < MICRO_A_TILE_PLL_UNROLL; ++mu_pll_i) {
        localA[MACRO_TILE_LENGTH_A_AND_PAD*(a_offset_pll_unroll + mu_pll_i) + (a_offset_perp_unroll + mu_perp_i)] = 
        a[mu_pll_i*STRIDE_PLL_K_A + mu_perp_i*STRIDE_PERP_K_A];
      }
    }
    a += STRIDE_PLL_K_A*G_UNROLL;
    
    
    /* load data from b into LDS */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MICRO_B_TILE_PERP_UNROLL; ++mu_perp_i) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < MICRO_B_TILE_PLL_UNROLL; ++mu_pll_i) {
        localB[MACRO_TILE_LENGTH_B_AND_PAD*(b_offset_pll_unroll + mu_pll_i) + (b_offset_perp_unroll + mu_perp_i)] = 
        b[mu_pll_i*STRIDE_PLL_K_B + mu_perp_i*STRIDE_PERP_K_B];
      }
    }
    b += STRIDE_PLL_K_B*G_UNROLL;
    
    
    /* make sure all loads from LDS memory have completed */
    barrier(CLK_LOCAL_MEM_FENCE); 
    
    lA = localA + micro_id_a*1;
    lB = localB + micro_id_b*1;
    
    for (unsigned u = 0; u < UNROLL; ++u){
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_LENGTH_A; ++i){
        rA[i] = lA[i*C_INTERWEAVE_STRIDE_A];
      }
      lA += MACRO_TILE_LENGTH_A_AND_PAD;
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_LENGTH_B; ++i){
        rB[i] = lB[i*C_INTERWEAVE_STRIDE_B];
      }
      lB += MACRO_TILE_LENGTH_B_AND_PAD;
      
      #pragma unroll
      for (unsigned row = 0; row < MICRO_TILE_LENGTH_A; ++row){
        #pragma unroll
        for (unsigned col = 0; col < MICRO_TILE_LENGTH_B; ++col){
          rC[row][col] += rA[row]*rB[col];   
        }
      }
    }
    
    /* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
    barrier(CLK_LOCAL_MEM_FENCE); 
    --n_unrolls_remaining;
  }
  
  
  unsigned index;
  
  /* the following variables are used in implementing a basic atomic increment */
  global TFLOAT * ptr_to_c_elm;  // with `restrict' is no faster
  TFLOAT previous_value; 
  uint newVal;
  uint prevVal;
  
  
  /* the case where this is not an edge tile : will write to all cells */ 
  if ((group_id_a != N_GROUPS_A - 1) && (group_id_b != N_GROUPS_B - 1)){ 
    
    /* loops for writing to c */
    #pragma unroll
    for (unsigned row = 0; row < MACRO_TILE_LENGTH_A; row += N_MICRO_IN_MACRO_A) {
      #pragma unroll
      for (unsigned col = 0; col < MACRO_TILE_LENGTH_B; col += N_MICRO_IN_MACRO_B) {
        
        index = STRIDE_PLL_M_C*(write_start_a + row) + STRIDE_PLL_N_C*(write_start_b + col);
        
        ptr_to_c_elm = c + index;
        do {
          previous_value = *ptr_to_c_elm;
          prevVal = as_uint(previous_value);
          newVal = as_uint(alpha*rC[row/C_INTERWEAVE_STRIDE_A][col/C_INTERWEAVE_STRIDE_B] + previous_value);
        } while (atomic_cmpxchg(( __global uint*)(ptr_to_c_elm), prevVal, newVal) != prevVal);
      }
    }
    
  }
  
  else{
    /* loops for writing to c */
    #pragma unroll
    for (unsigned row = 0; row < MACRO_TILE_LENGTH_A; row += N_MICRO_IN_MACRO_A) {
      #pragma unroll
      for (unsigned col = 0; col < MACRO_TILE_LENGTH_B; col += N_MICRO_IN_MACRO_B) {
        
        /* catching the write cases */
        if (
        /* B overflow, but not A edge */
        ((write_start_b + col >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1)) && group_id_a  !=  (N_GROUPS_A - 1 )) ||
        /* A overflow, but not B edge */
        ((write_start_a + row >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)) && group_id_b  !=  (N_GROUPS_B - 1 )) ||
        /* A edge and B edge and A overflow and B overflow */
        (
        group_id_a == (N_GROUPS_A - 1)   && 
        group_id_b == (N_GROUPS_B - 1)   && 
        write_start_b + col >= MACRO_TILE_LENGTH_B*(N_GROUPS_B - 1) && 
        write_start_a + row >= MACRO_TILE_LENGTH_A*(N_GROUPS_A - 1)
        )){
          
          index = STRIDE_PLL_M_C*(write_start_a + row) + STRIDE_PLL_N_C*(write_start_b + col);
          
          ptr_to_c_elm = c + index;
          do {
            previous_value = *ptr_to_c_elm;
            prevVal = as_uint(previous_value);
            newVal = as_uint(alpha*rC[row/C_INTERWEAVE_STRIDE_A][col/C_INTERWEAVE_STRIDE_B] + previous_value);
          } while (atomic_cmpxchg(( __global uint*)(ptr_to_c_elm), prevVal, newVal) != prevVal);
        }
      }
    }
    
  }
}
