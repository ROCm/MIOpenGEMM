/* ******************************************************************
* This gemm kernel string was generated at Wed Feb  1 12:07:26 2017
****************************************************************** */

/* ***********************************************
* These parameters define WHAT this kernel does *
* *********************************************** */
#define __M__ 1234
#define __N__ 2345
#define __K__ 3456
#define __LDA__ 1245
#define __LDB__ 3478
#define __LDC__ 1267
#define IS_COL_MAJOR 1
#define A_TRANSPOSED 0
#define B_TRANSPOSED 0
#define C_TRANSPOSED 0
#define TFLOAT  float
/* certain kernels can only do one or the other of the terms in c <- alpha a*b + beta c. */
/* TODO : if DOES_BETA_C_INC is 0, then alpha should not appear as a kernel parameter */
#define DOES_BETA_C_INC 0
#define DOES_ALPHA_A_B_INC 1

/* TODO : remove final barrier, not nec. Check performance is not mysteriously hit! */
/* TODO : beta = 1 optimisation */
/* TODO : investigate mad. When should one use this instead of standard overloads, += and * ? 


/* ****************************************
 * These parameters define HOW it does it *
 * *****************************************/
/* Defines a tile shape. Each thread will process a tile of this shape from C (if N_WORK_ITEMS_PER_C_ELM > 1 the processing is shared with threads in other WGs)  */
#define MICRO_TILE_WIDTH 6
#define MICRO_TILE_HEIGHT 8
/* Area of C which a workgroup will process. Recall that a workgroup is made of several threads which share LDS memory */
#define MACRO_TILE_WIDTH 96
#define MACRO_TILE_HEIGHT 128
/* How much a workgroup load (global -> LDS) in the k-direction at each iteration of the outer-most loop */
#define UNROLL 8
/* padding in LDS to avoid bank conflicts*/
#define PAD 1
/* whether or not this kernel uses the edge trick (see documentation : (TODO, currently internal AMD document)) */
/* this precompiler defn has no direct influence on the running the kernel, implementation already done in make_kernel.py */
#define EDGETRICK 1
/* the number of work items working on the same c element. if this is 1, there will be just one thread doing all k multiply-adds, */
/* otherwise if it is greater than 1, each thread will be computing ~ k / N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added at the end */ 
#define N_WORK_ITEMS_PER_C_ELM 2
/* whether or not to shimmy the starting k, in an attempt to avoid cache line overuse for cases where lda/ldb are powers of 2 */
/* if 0, no shimmying. if 1, instead of starting at k = 0 workgroups start at some negative offset dependent on work group id */
/* in the same way as the final unroll populates LDS with zeros in k mod UNROLL != 0, the initial negative indices here populate with 0 */
#define UNROLL_FOR_OFFSET 1

/* define the way in which work groups are assigned to tiles */
/* 1 : column-by-column
 * 2 : row-by-row 
 * 3 : by rows within super-column  */
#define GROUP_ALLOCATION 3/* this variable is declared because we have GROUP_ALLOCATION type 3. */
/* It should define how many workgroups we expect to have active simulantaneuosly. */
#define N_TARGET_ACTIVE_WORKGROUPS 64
/* Whether the load tiles are long in the direction of unroll (1) or perpendicular to the direction of unroll (0) */
/* Note : if the load tiles are long in the direction of unroll, the destination tile in LDS is NOT contiguous  */
/* We include these parameters here as pre-processor variables, but the loading micro-tile shapes are set in make_kernel.py */
#define WORK_ITEM_LOAD_A_PLL_TO_UNROLL 0
#define WORK_ITEM_LOAD_B_PLL_TO_UNROLL 1
/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if the load tiles are truly contiguous tiles of A/B (0) */
/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */
#define LOAD_TO_LDS_INTERWOVEN 1
/* Whether the micro tile being processed by a compute item is interwoven with other micro tiles (ala Cobalt, (1)) or if the micro tiles are contiguous in C */
/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */
#define C_MICRO_TILES_INTERWOVEN 1
/* Whether to use the unroll pragma to encourage the compiler to unroll certain loops */
/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */
#define PRAGMA_UNROLL_FORLOOPS 1
/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into registers, as compared to doing the math. */
/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */
#define N_PREFETCH_FOR_REGISTER_LOAD 0
/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into LDS, as compared to the unroll loop */
/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */
#define N_PREFETCH_FOR_LDS_LOAD 0




/* *****************************************************************************
 * The following are all implied by the preceding: these are NOT free parameters!
 * **************************************************************************** */
/*  <+_+>
 *           <+_+>
 *      <+_+>
 * TODO : rerereconsider assignment of workitems to load and math regions, there may be some sweet overlap spot where values automatically in registers for math (?) */

#define MACRO_TILE_AREA 12288  // MACRO_TILE_WIDTH*MACRO_TILE_HEIGHT
#define MICRO_TILE_AREA 48 // MICRO_TILE_WIDTH * MICRO_TILE_HEIGHT
#define N_WORK_ITEMS_PER_WORKGROUP  256 // MACRO_TILE_AREA / MICRO_TILE_AREA
#define MACRO_TILE_HEIGHT_AND_PAD 129 // MACRO_TILE_HEIGHT + PAD
#define MACRO_TILE_WIDTH_AND_PAD 97 // MACRO_TILE_WIDTH + PAD
#define N_ELEMENTS_IN_A_UNROLL 1024 // MACRO_TILE_HEIGHT * UNROLL
#define N_ELEMENTS_IN_B_UNROLL 768 // MACRO_TILE_WIDTH * UNROLL
#define N_ELEMENTS_IN_PADDED_A_UNROLL 1032 // MACRO_TILE_HEIGHT_AND_PAD * UNROLL
#define N_ELEMENTS_IN_PADDED_B_UNROLL 776 // MACRO_TILE_WIDTH_AND_PAD * UNROLL
#define N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM 4 // N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP
#define N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM 3 // N_ELEMENTS_IN_B_UNROLL / N_WORK_ITEMS_PER_WORKGROUP
#define N_MICRO_TILES_VERTICALLY 16 // MACRO_TILE_HEIGHT / MICRO_TILE_HEIGHT
#define N_MICRO_TILES_HORIZONTALLY 16 // MACRO_TILE_WIDTH / MICRO_TILE_WIDTH

/* MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM */
/* The dimensions of a tile in A loaded by a work item.  */
#define MICRO_A_TILE_PLL_UNROLL 1 // size of the loaded tile, pll to unroll
#define MICRO_A_TILE_PERP_UNROLL 4
#define N_MICRO_A_TILES_PLL_UNROLL 8 // UNROLL / MICRO_A_TILE_PLL_UNROLL

/*  MICRO_B_TILE_PLL_UNROLL * MICRO_B_TILE_PERP_UNROLL = N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM */
/* The dimensions of a tile in B read by a work item */
#define MICRO_B_TILE_PLL_UNROLL 1
#define MICRO_B_TILE_PERP_UNROLL 3
#define N_MICRO_B_TILES_PLL_UNROLL 8 // UNROLL / MICRO_B_TILE_PLL_UNROLL
#define N_MICRO_B_TILES_PLL_UNROLL 8 // UNROLL / MICRO_B_TILE_PLL_UNROLL

/* two more parameters, which do dot have an effect the running of this kernel (used in enqueuing) */
/* the total number of work groups this kernel will use (recall m,n,k are fixed) */ 
/* N_WORK_ITEMS_PER_C_ELM * ((__M__/MACRO_TILE_HEIGHT) + (__M__%MACRO_TILE_HEIGHT != 0)) * ((__M__/MACRO_TILE_WIDTH) + (__N__%MACRO_TILE_WIDTH != 0)) */ 
#define N_WORK_GROUPS 500
/* the global work size, ie the total mumber of work items (threads) which will run */ 
/* N_WORK_GROUPS * N_WORK_ITEMS_PER_WORKGROUP */ 
#define GLOBAL_WORK_SIZE 128000


/* To move from A[row][col] to A[row+1][col], how much should the pointer increment? As we have A_TRANSPOSED = 0 and IS_COL_MAJOR = 1, this is 1 */
#define ROW_STRIDE_A 1
/* To move from A[row][col] to A[row][col+1], how much should the pointer increment? As we have A_TRANSPOSED = 0 and IS_COL_MAJOR = 1, this is LDA */
#define COL_STRIDE_A 1245

/* To move from B[row][col] to B[row+1][col], how much should the pointer increment? As we have B_TRANSPOSED = 0 and IS_COL_MAJOR = 1, this is 1 */
#define ROW_STRIDE_B 1
/* To move from B[row][col] to B[row][col+1], how much should the pointer increment? As we have B_TRANSPOSED = 0 and IS_COL_MAJOR = 1, this is LDB */
#define COL_STRIDE_B 3478

/* To move from C[row][col] to C[row+1][col], how much should the pointer increment? As we have C_TRANSPOSED = 0 and IS_COL_MAJOR = 1, this is 1 */
#define ROW_STRIDE_C 1
/* To move from C[row][col] to C[row][col+1], how much should the pointer increment? As we have C_TRANSPOSED = 0 and IS_COL_MAJOR = 1, this is LDC */
#define COL_STRIDE_C 1267

/* 1 + (__M__ - 1) % MACRO_TILE_HEIGHT  */ 
#define PRESHIFT_BOTTOMMOST_TILE_HEIGHT 82 // somewhere in 1 ... MACRO_TILE_HEIGHT
/* 1 + (__N__ - 1) % MACRO_TILE_WIDTH */ 
#define PRESHIFT_RIGHTMOST_TILE_WIDTH 41 // somewhere in 1 ... MACRO_TILE_WIDTH

/* the number of work groups vertically and horizontally. */
/* note that this ignores n_work_items_per_c_elm, so that only one workgroup per c cell is used in computing this */ 
/* number of groups vertically : __M__ / MACRO_TILE_HEIGHT + (PRESHIFT_BOTTOMMOST_TILE_HEIGHT != MACRO_TILE_HEIGHT) */
#define N_GROUPS_VERTICALLY 10
/* number of groups horizontally : __N__ / MACRO_TILE_WIDTH + (PRESHIFT_RIGHTMOST_TILE_WIDTH != MACRO_TILE_WIDTH)*/
#define N_GROUPS_HORIZONTALLY 25

/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL 16 // N_WORK_ITEMS_PER_C_ELM*UNROLL

/* This variable defines the width of super-columns (we have GROUP_ALLOCATION 3). It is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) */
#define SUPER_COLUMN_WIDTH 5
/* LAST_SUPER_COLUMN_WIDTH : N_GROUPS_HORIZONTALLY % SUPER_COLUMN_WIDTH  */
#define LAST_SUPER_COLUMN_WIDTH 0


__attribute__((reqd_work_group_size(256,1, 1)))
__kernel void bolziberb(
__global TFLOAT       *          c,
__global const TFLOAT * restrict a,
__global const TFLOAT * restrict b,
const TFLOAT alpha,
const TFLOAT beta,
const unsigned a_offset,
const unsigned b_offset,
const unsigned c_offset  
)
{
  
  
  /* In OpenCL, host code does not have access to raw data pointers. */
  /* Host code works with cl_mem objects, which encapsulate and hide raw points. */
  /* For this reason, host code CANNOT simply increment pointers to data, */
  /* as one can do with pointers for CPU gemm, or cublas gemm for that matter. */
  a += a_offset;
  b += b_offset;
  c += c_offset;
  
  
  
  
  const unsigned group_id = get_group_id(0);
  const unsigned group_id_xy = group_id / N_WORK_ITEMS_PER_C_ELM;
  const unsigned group_id_z = group_id % N_WORK_ITEMS_PER_C_ELM;
  const unsigned local_id = get_local_id(0);
  
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
  unsigned group_id_horizontal;
  unsigned group_id_vertical;
  unsigned wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*N_GROUPS_VERTICALLY);
  
      
  if (group_id_xy < (N_GROUPS_HORIZONTALLY - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_VERTICALLY){
    group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
    group_id_vertical = (group_id_xy / SUPER_COLUMN_WIDTH) % N_GROUPS_VERTICALLY;
  }
  else{
    group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % LAST_SUPER_COLUMN_WIDTH;
    group_id_vertical = (group_id_xy  - (N_GROUPS_HORIZONTALLY - LAST_SUPER_COLUMN_WIDTH)*N_GROUPS_VERTICALLY) / LAST_SUPER_COLUMN_WIDTH;
  }
  
  unsigned macro_tile_start_row_in_c = group_id_vertical*MACRO_TILE_HEIGHT;
  unsigned macro_tile_start_col_in_c = group_id_horizontal*MACRO_TILE_WIDTH;  
  
  
  /* Special case of the tile being on far right : pull the tile to the left just enough so that it doesn't overflow C */
  if (group_id_horizontal == N_GROUPS_HORIZONTALLY - 1){
    macro_tile_start_col_in_c -= (MACRO_TILE_WIDTH - PRESHIFT_RIGHTMOST_TILE_WIDTH);
  }
    
  /* Special case of the tile being on the bottom : pull the tile up just enough so that it doesn't overflow C */    
  if (group_id_vertical == N_GROUPS_VERTICALLY - 1){
    macro_tile_start_row_in_c -= (MACRO_TILE_HEIGHT - PRESHIFT_BOTTOMMOST_TILE_HEIGHT);
  }
  
  /* move to the top left corner of a (top left corner of b) of region required by the macro tile */
  a += macro_tile_start_row_in_c*ROW_STRIDE_A;   
  b += macro_tile_start_col_in_c*COL_STRIDE_B;
  
  
  /* a,b are pointing to the top left of the region required by the macro tile, but this work group  */
  /* might not process the whole of a and b. We now turn 90 and shift pointers a,b to the start for this wg */
  a += UNROLL*group_id_z*COL_STRIDE_A;
  b += UNROLL*group_id_z*ROW_STRIDE_B;
  
  /* this additional offset of a and b appears because UNROLL_FOR_OFFSET is 1 */
  unsigned unroll_offset = (3*group_id_vertical + 11*group_id_vertical)%UNROLL;
  unsigned k_plus_offset = __K__ + unroll_offset;
  a -= unroll_offset*COL_STRIDE_A;
  b -= unroll_offset*ROW_STRIDE_B;
  
  /* Define which rows or columns of A, B this thread will load from global into LDS */
  const unsigned pll_unroll_a_load_id = local_id % N_MICRO_A_TILES_PLL_UNROLL;
  const unsigned perp_unroll_a_load_id = local_id / N_MICRO_A_TILES_PLL_UNROLL;
  const unsigned pll_unroll_b_load_id = local_id % N_MICRO_B_TILES_PLL_UNROLL;
  const unsigned perp_unroll_b_load_id = local_id / N_MICRO_B_TILES_PLL_UNROLL;
  
  /* make the micro adjustments (A) for the thread, getting ready to load */
  const unsigned a_offset_pll_unroll =  pll_unroll_a_load_id;
  const unsigned a_offset_perp_unroll =  perp_unroll_a_load_id;
  a += COL_STRIDE_A * a_offset_pll_unroll;
  a += ROW_STRIDE_A * a_offset_perp_unroll;
  
  /* make the micro adjustments (B) for the thread, getting ready to load */
  const unsigned b_offset_pll_unroll =  pll_unroll_b_load_id;
  const unsigned b_offset_perp_unroll =  perp_unroll_b_load_id;
  b += ROW_STRIDE_B * b_offset_pll_unroll;
  b += COL_STRIDE_B * b_offset_perp_unroll;
  
  
  /* Define which part of the C macro-tile this thread will process */
  const unsigned micro_id_vertical = local_id % N_MICRO_TILES_VERTICALLY;
  const unsigned micro_id_horizontal = local_id / N_MICRO_TILES_VERTICALLY;
    
  /* LDS memory */
  __local TFLOAT localA[N_ELEMENTS_IN_PADDED_A_UNROLL];
  __local TFLOAT localB[N_ELEMENTS_IN_PADDED_B_UNROLL];
  /* register memory */
    TFLOAT rC[MICRO_TILE_HEIGHT][MICRO_TILE_WIDTH] = {{0.}};
  TFLOAT rA[MICRO_TILE_HEIGHT];
  TFLOAT rB[MICRO_TILE_WIDTH];
  
  /* jumping pointers to locate the LDS to load into register memory */
  __local const TFLOAT * lA;
  __local const TFLOAT * lB;
  
  
  /* a certain number of work groups process one more unroll. Note that with UFO = 1, this depends on column */
  const int n_work_groups_with_1_more = (k_plus_offset % G_UNROLL) / UNROLL; 
  
  // branching between work groups : some wgs have 1 more unroll to process.
  int n_unrolls_remaining = (k_plus_offset / G_UNROLL) +  (group_id_z < n_work_groups_with_1_more);
  
  /* This is where the first unroll will be performed. Identical to what is in the main while, but with zero buffering.  */
  if (group_id_z == 0){
    
    /* load first bit of data from a into LDS, ignoring the prepended values (less than a full unroll)  */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MACRO_TILE_HEIGHT; mu_perp_i += MACRO_TILE_HEIGHT/MICRO_A_TILE_PERP_UNROLL) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_A_TILE_PLL_UNROLL) {
        localA[MACRO_TILE_HEIGHT_AND_PAD*(a_offset_pll_unroll + mu_pll_i) + (a_offset_perp_unroll + mu_perp_i)] = 
        (a_offset_pll_unroll + mu_pll_i) >= unroll_offset ? a[mu_pll_i*COL_STRIDE_A + mu_perp_i*ROW_STRIDE_A] : 0;
      }
    }
    
    /* load first bit of data from b into LDS, ignoring the prepended values (less than a full unroll) */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MACRO_TILE_WIDTH; mu_perp_i += MACRO_TILE_WIDTH/MICRO_B_TILE_PERP_UNROLL) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_B_TILE_PLL_UNROLL) {
        localB[MACRO_TILE_WIDTH_AND_PAD*(b_offset_pll_unroll + mu_pll_i) + (b_offset_perp_unroll + mu_perp_i)] = 
        (b_offset_pll_unroll + mu_pll_i) >= unroll_offset ? b[mu_pll_i*ROW_STRIDE_B + mu_perp_i*COL_STRIDE_B] : 0;
      }
    }
    
    a += COL_STRIDE_A*G_UNROLL;
    b += ROW_STRIDE_B*G_UNROLL;
    
    /* make sure all loads from LDS memory have completed */
    barrier(CLK_LOCAL_MEM_FENCE); 
    lA = localA + micro_id_vertical*1;
    lB = localB + micro_id_horizontal*1;
    
    for (unsigned u = 0; u < UNROLL; ++u){
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_HEIGHT; ++i){
        rA[i] = lA[i*N_MICRO_TILES_VERTICALLY];
      }
      lA += MACRO_TILE_HEIGHT_AND_PAD;
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_WIDTH; ++i){
        rB[i] = lB[i*N_MICRO_TILES_HORIZONTALLY];
      }
      lB += MACRO_TILE_WIDTH_AND_PAD;
      #pragma unroll
      for (unsigned row = 0; row < MICRO_TILE_HEIGHT; ++row){
        #pragma unroll
        for (unsigned col = 0; col < MICRO_TILE_WIDTH; ++col){
          //mad(rA[row],rB[col],rC[row][col]); 
          //can the compiler change these unsigneds to chars? if not, maybe try. 
          //That said, it's going to be unrolled anyway, so not worth it.
          rC[row][col] += rA[row]*rB[col];   
        }
      }
      
    }
    
    /* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
    barrier(CLK_LOCAL_MEM_FENCE); 
    --n_unrolls_remaining;
  }
  
  while (n_unrolls_remaining > 0){
    
    /* load data from a into LDS */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MACRO_TILE_HEIGHT; mu_perp_i += MACRO_TILE_HEIGHT/MICRO_A_TILE_PERP_UNROLL) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_A_TILE_PLL_UNROLL) {
        localA[MACRO_TILE_HEIGHT_AND_PAD*(a_offset_pll_unroll + mu_pll_i) + (a_offset_perp_unroll + mu_perp_i)] = 
        a[mu_pll_i*COL_STRIDE_A + mu_perp_i*ROW_STRIDE_A];
      }
    }
    
    /* load data from b into LDS */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MACRO_TILE_WIDTH; mu_perp_i += MACRO_TILE_WIDTH/MICRO_B_TILE_PERP_UNROLL) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_B_TILE_PLL_UNROLL) {
        localB[MACRO_TILE_WIDTH_AND_PAD*(b_offset_pll_unroll + mu_pll_i) + (b_offset_perp_unroll + mu_perp_i)] = 
        b[mu_pll_i*ROW_STRIDE_B + mu_perp_i*COL_STRIDE_B];
      }
    }
    
    a += COL_STRIDE_A*G_UNROLL;
    b += ROW_STRIDE_B*G_UNROLL;
    
    /* make sure all loads from LDS memory have completed */
    barrier(CLK_LOCAL_MEM_FENCE); 
    lA = localA + micro_id_vertical*1;
    lB = localB + micro_id_horizontal*1;
    
    for (unsigned u = 0; u < UNROLL; ++u){
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_HEIGHT; ++i){
        rA[i] = lA[i*N_MICRO_TILES_VERTICALLY];
      }
      lA += MACRO_TILE_HEIGHT_AND_PAD;
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_WIDTH; ++i){
        rB[i] = lB[i*N_MICRO_TILES_HORIZONTALLY];
      }
      lB += MACRO_TILE_WIDTH_AND_PAD;
      #pragma unroll
      for (unsigned row = 0; row < MICRO_TILE_HEIGHT; ++row){
        #pragma unroll
        for (unsigned col = 0; col < MICRO_TILE_WIDTH; ++col){
          //mad(rA[row],rB[col],rC[row][col]); 
          //can the compiler change these unsigneds to chars? if not, maybe try. 
          //That said, it's going to be unrolled anyway, so not worth it.
          rC[row][col] += rA[row]*rB[col];   
        }
      }
      
    }
    
    /* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
    barrier(CLK_LOCAL_MEM_FENCE); 
    --n_unrolls_remaining;
  }
  
  unsigned k_remaining = k_plus_offset % UNROLL;
  /* There is one workgroup which will process the remainder (less that UNROLL) */
  /* JN 16 Nov 2016 */
  if (group_id_z == n_work_groups_with_1_more && k_remaining > 0){
    
    /* load final bit of data from a into LDS, less than a full unroll */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MACRO_TILE_HEIGHT; mu_perp_i += MACRO_TILE_HEIGHT/MICRO_A_TILE_PERP_UNROLL) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_A_TILE_PLL_UNROLL) {
        localA[MACRO_TILE_HEIGHT_AND_PAD*(a_offset_pll_unroll + mu_pll_i) + (a_offset_perp_unroll + mu_perp_i)] = 
        (a_offset_pll_unroll + mu_pll_i) < k_remaining ? a[mu_pll_i*COL_STRIDE_A + mu_perp_i*ROW_STRIDE_A] : 0;
      }
    }
    
    /* load final bit of data from b into LDS, less than a full unroll */
    #pragma unroll
    for (unsigned mu_perp_i = 0; mu_perp_i < MACRO_TILE_WIDTH; mu_perp_i += MACRO_TILE_WIDTH/MICRO_B_TILE_PERP_UNROLL) {
      #pragma unroll
      for (unsigned mu_pll_i = 0; mu_pll_i < UNROLL; mu_pll_i += UNROLL/MICRO_B_TILE_PLL_UNROLL) {
        localB[MACRO_TILE_WIDTH_AND_PAD*(b_offset_pll_unroll + mu_pll_i) + (b_offset_perp_unroll + mu_perp_i)] = 
        (b_offset_pll_unroll + mu_pll_i) < k_remaining ? b[mu_pll_i*ROW_STRIDE_B + mu_perp_i*COL_STRIDE_B] : 0;
      }
    }
    
    
    /* make sure all loads from LDS memory have completed */
    barrier(CLK_LOCAL_MEM_FENCE); 
    lA = localA + micro_id_vertical*1;
    lB = localB + micro_id_horizontal*1;
    
    for (unsigned u = 0; u < k_remaining; ++u){
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_HEIGHT; ++i){
        rA[i] = lA[i*N_MICRO_TILES_VERTICALLY];
      }
      lA += MACRO_TILE_HEIGHT_AND_PAD;
      
      #pragma unroll
      for (unsigned i = 0; i < MICRO_TILE_WIDTH; ++i){
        rB[i] = lB[i*N_MICRO_TILES_HORIZONTALLY];
      }
      lB += MACRO_TILE_WIDTH_AND_PAD;
      #pragma unroll
      for (unsigned row = 0; row < MICRO_TILE_HEIGHT; ++row){
        #pragma unroll
        for (unsigned col = 0; col < MICRO_TILE_WIDTH; ++col){
          //mad(rA[row],rB[col],rC[row][col]); 
          //can the compiler change these unsigneds to chars? if not, maybe try. 
          //That said, it's going to be unrolled anyway, so not worth it.
          rC[row][col] += rA[row]*rB[col];   
        }
      }
      
    }
    
    /* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one) */
    barrier(CLK_LOCAL_MEM_FENCE); 
  }
  const unsigned write_start_row = macro_tile_start_row_in_c + micro_id_vertical*1;
  const unsigned write_start_col = macro_tile_start_col_in_c + micro_id_horizontal*1;
  unsigned index;
  
  /* the following variables are used in implementing a basic atomic increment */
  global TFLOAT * ptr_to_c_elm;  // with `restrict' is no faster
  TFLOAT previous_value; 
  uint newVal;
  uint prevVal;
  
  
  /* the case where this is not an edge tile : will write to all cells */
  if ((group_id_horizontal != N_GROUPS_HORIZONTALLY - 1 || PRESHIFT_RIGHTMOST_TILE_WIDTH == MACRO_TILE_WIDTH) 
  && (group_id_vertical != N_GROUPS_VERTICALLY - 1 || PRESHIFT_BOTTOMMOST_TILE_HEIGHT == MACRO_TILE_HEIGHT)){
    
    /* loops for writing to c */
    #pragma unroll
    for (unsigned row = 0; row < MACRO_TILE_HEIGHT; row += N_MICRO_TILES_VERTICALLY) {
      #pragma unroll
      for (unsigned col = 0; col < MACRO_TILE_WIDTH; col += N_MICRO_TILES_HORIZONTALLY) {
        
        index = ROW_STRIDE_C*(write_start_row + row) + COL_STRIDE_C*(write_start_col + col);
        
        ptr_to_c_elm = c + index;
        do {
          previous_value = *ptr_to_c_elm;
          prevVal = as_uint(previous_value);
          newVal = as_uint(alpha*rC[row/N_MICRO_TILES_VERTICALLY][col/N_MICRO_TILES_HORIZONTALLY] + previous_value);
        } while (atomic_cmpxchg(( __global uint*)(ptr_to_c_elm), prevVal, newVal) != prevVal);
      }
    }
    
  }
  
  else{
    /* loops for writing to c */
    #pragma unroll
    for (unsigned row = 0; row < MACRO_TILE_HEIGHT; row += N_MICRO_TILES_VERTICALLY) {
      #pragma unroll
      for (unsigned col = 0; col < MACRO_TILE_WIDTH; col += N_MICRO_TILES_HORIZONTALLY) {
        
        /* catching the write cases for lower(l), right(r) and lr-corner tiles */
        if (
        ((write_start_col + col >= MACRO_TILE_WIDTH*(N_GROUPS_HORIZONTALLY - 1)) && group_id_vertical   != (N_GROUPS_VERTICALLY   - 1 )) ||
        ((write_start_row + row >= MACRO_TILE_HEIGHT*(N_GROUPS_VERTICALLY - 1 )) && group_id_horizontal != (N_GROUPS_HORIZONTALLY - 1 )) ||
        (
        group_id_vertical == (N_GROUPS_VERTICALLY-1)     && 
        group_id_horizontal == (N_GROUPS_HORIZONTALLY-1) && 
        write_start_col + col >= MACRO_TILE_WIDTH*(N_GROUPS_HORIZONTALLY - 1) && 
        write_start_row + row >= MACRO_TILE_HEIGHT*(N_GROUPS_VERTICALLY - 1)
        )){
          
          index = ROW_STRIDE_C*(write_start_row + row) + COL_STRIDE_C*(write_start_col + col);
          
          ptr_to_c_elm = c + index;
          do {
            previous_value = *ptr_to_c_elm;
            prevVal = as_uint(previous_value);
            newVal = as_uint(alpha*rC[row/N_MICRO_TILES_VERTICALLY][col/N_MICRO_TILES_HORIZONTALLY] + previous_value);
          } while (atomic_cmpxchg(( __global uint*)(ptr_to_c_elm), prevVal, newVal) != prevVal);
        }
      }
    }
    
  }
}
