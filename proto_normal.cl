/* ***********************************************
* These parameters define WHAT this kernel does *
* *********************************************** */
#define TFLOAT  float

#define LDX 4001

/* less than or equal to LDX, DIM_COAL is size in the contiguous direction */ 
#define DIM_COAL 3999

/* DIM_UNCOAL is the other dimension of the matrix */ 
#define DIM_UNCOAL 4180


#define COAL_IS_PLL_UNROLL 1

#define UNROLL 16

#define MACRO_TILE_LENGTH 96

/* ****************************************
 * These parameters define HOW it does it *
 * *****************************************/

/* Whether the load tiles are long in the direction of unroll (1) or perpendicular to the direction of unroll (0) */
#define WORK_ITEM_LOAD_PLL_TO_UNROLL 0


/* *****************************************************************************
 * The following are all implied by the preceding: these are NOT free parameters!
 * **************************************************************************** */
    
/*          UNROLL * MACRO_TILE_LENGTH */
#define N_ELEMENTS_IN_MACRO_TILE 1536 
/*          micro (load) tile shape. set elsewhere based on WORK_ITEM_LOAD_PLL_TO_UNROLL */
#define MICRO_TILE_PLL_UNROLL 1
#define MICRO_TILE_PERP_UNROLL 6
#define N_ELEMENTS_IN_MICRO_TILE 6

/*        including spillover tiles */
#define N_MACRO_TILES_PLL_UNROLL
/*        the final unroll has fewer lines than UNROLL, how many*/
#define N_LINES_FINAL_PLL_UNROLL


/*        including spillover tiles */
#define N_MACRO_TILES_PERP_UNROLL
/*         the final unroll column spills, by how many lines [0...MACRO_TILE_LENGTH-1] */
#define OVERFLOW_PERP_UNROLL


/*          total number of work groups [1...UNROLL] */
#define N_MACRO_TILES 5808

/*          total number of work items */
#define N_MICRO_TILES 1486848 

/*          jump size to move a line parallel to unroll */
#define STRIDE_PLL_UNROLL

/*          jump size to move a line perpendicular to unroll */
#define STRIDE_PERP_UNROLL

/*          how many micro tiles parallel to unroll in a macro tile */
#define N_MICRO_PLL_UNROLL 

/*          how many micro tiles perpendicular to unroll in a macro tile */
#define N_MICRO_PERP_UNROLL 

#define GLOBAL_OFFSET_Y 0

__attribute__((reqd_work_group_size(256,1, 1)))
__kernel void tg_blablabla
(
__global const TFLOAT * restrict x, 
const unsigned x_offset,
__global const TFLOAT * restrict y, 
const unsigned y_offset) 

{
  x += x_offset;
  y += y_offset;
  y += GLOBAL_OFFSET_Y;
  
  const unsigned group_id = get_group_id(0);
  const unsigned local_id = get_local_id(0);
  
  /* pll to unroll must change fastest for y to be correct, do not switch % and / !*/
  const unsigned group_id_pll_unroll = group_id % N_MACRO_TILES_PLL_UNROLL;
  const unsigned group_id_perp_unroll = group_id / N_MACRO_TILES_PLL_UNROLL;
  
  x += group_id_pll_unroll*UNROLL*STRIDE_PLL_UNROLL;
  x += group_id_perp_unroll*LOAD_TILE_LENGTH*STRIDE_PERP_UNROLL;
  
  y += group_id*N_ELEMENTS_IN_MACRO_TILE;
   
  //move in case of tail perpendicular to unroll:
  if (group_id_perp_unroll == N_MACRO_TILES_PERP_UNROLL - 1){
    x - OVERFLOW_PERP_UNROLL * STRIDE_PERP_UNROLL;
  }
  
  const bool in_pll_tail = (group_id_pll_unroll == N_MACRO_TILES_PLL_UNROLL - 1);
  
  /* which part of the micro tile to load */
  const unsigned pll_unroll_load_id  = local_id / N_MICRO_PERP_UNROLL;
  const unsigned perp_unroll_load_id = local_id % N_MICRO_PERP_UNROLL;  
  
  /* make the micro adjustments for the thread, getting ready to load */
  const unsigned offset_pll_unroll = MICRO_TILE_PLL_UNROLL * pll_unroll_load_id;
  const unsigned offset_perp_unroll = MICRO_TILE_PERP_UNROLL * perp_unroll_load_id;
  
  x += STRIDE_PLL_UNROLL  * offset_pll_unroll;
  x += STRIDE_PERP_UNROLL * offset_perp_unroll;
  
  y += MACRO_TILE_LENGTH * offset_pll_unroll;
  y += offset_perp_unroll;
  

  

  if (in_pll_tail == true){
    #pragma unroll
    for (unsigned mu_pll_i = 0; mu_pll_i < MICRO_TILE_PLL_UNROLL; ++mu_pll_i) {  
      #pragma unroll
      for (unsigned mu_perp_i = 0; mu_perp_i < MICRO_TILE_PERP_UNROLL; ++mu_perp_i) {
        y[mu_pll_i*MACRO_TILE_LENGTH + mu_perp_i] = 
        x[mu_pll_i*STRIDE_PLL_UNROLL + mu_perp_i*STRIDE_PERP_UNROLL];
      }
    }
  }
  
  else{
    #pragma unroll
    for (unsigned mu_pll_i = 0; mu_pll_i < MICRO_TILE_PLL_UNROLL; ++mu_pll_i) {  
      #pragma unroll
      for (unsigned mu_perp_i = 0; mu_perp_i < MICRO_TILE_PERP_UNROLL; ++mu_perp_i) {
        y[mu_pll_i*MACRO_TILE_LENGTH + mu_perp_i] = 
        mu_pll_i < N_LINES_FINAL_PLL_UNROLL ? x[mu_pll_i*STRIDE_PLL_UNROLL + mu_perp_i*STRIDE_PERP_UNROLL] : 0;
      }
    }
  }
}
