"""This is the interesting part of project tinygemm
Go to `def make_kernel' for the entry point """

import math
import textwrap
import os
import sys
#from IPython.core.debugger import Tracer


def get_indented(astr):
  """
  insert two spaces at the beginning of every line
  """
  newstring = ""
  split_on_newline = astr.split("\n") 
  for l in split_on_newline[0:-1]:
    newstring += "  "
    newstring += l
    newstring += "\n"
  newstring += "  " 
  newstring += split_on_newline[-1]
  return newstring


def get_multiples(N):
  multiples = []
  for k in range(1, N+1):
    if N%k == 0:
      multiples.append(k)
  return multiples

  
def get_tile_dimensions(TH, TW, tS):
  """
  given a macro tile TH x TW, 
  and given a micro tile size of tS, 
  find the tallest possible micro tile size (tH x tW)
  to fit the macro tile. Example, macro tile is 6 x 4:
  
  * * * * 
  * * * * 
  * * * * 
  * * * * 
  * * * * 
  * * * * 
  
  tS = 2 return [2, 1]
  tS = 3 return [3, 1]
  tS = 4 return [2, 2]
  tS = 5 raise an error ((TH * TH) % tS != 0)
  tS = 6 return [6, 1]
  tS = 7 raise an error ((TH * TH) % tS != 0) 
  tS = 8 return [2, 4]
  tS = 9 raise an error ((TH * TH) % tS != 0)
  tS = 10 raise an error ((TH * TH) % tS != 0)
  tS = 11 raise an error ((TH * TH) % tS != 0)
  tS = 12 return [6, 2]
  tS = 13 .. 23 raise an error ((TH * TH) % tS != 0)
  tS = 24 return [6, 4]

  """  
  
  base_string = """TH : %d, TW : %d, tS : %d"""%(TH, TW, tS)

  tH, tW = None, None
  
   
  if (TH*TW) % tS != 0:
    raise RuntimeError("Areas of micro and macro tiles are incompatible : %s"%(base_string,))
    

  multiples_of_TH = get_multiples(TH)
  multiples_of_TH.sort()
  #Going through the multiples is descending order, we check if it is also multiple of tS
  for multiple_of_TH in multiples_of_TH[-1::-1]:
    if (tS % multiple_of_TH == 0) and ((tS / multiple_of_TH) <= TW):
      tH = multiple_of_TH
      tW = tS / tH
      break
  
  
  if tH == None:
    raise RuntimeError("Impossible tiling problem in get_tile_dimensions : %s"%(base_string,))
  
  
  base_string = """%s, tH : %d, tW : %d"""%(base_string, tH, tW)

  if tW  > tH : 
    pass
    #print "no `tall' tile. best `wide' one : %s. "%(base_string,),
  
  if TW % tW != 0 or TH % TH != 0 or tW*tH != tS : 
    raise RuntimeError("This is strange : the found micro tile size is not consistent with the macro tile %s"%(base_string,)) 
 
  
  return tH, tW
  
  





def make_kernel(dir_name = "where_to_write_kernel_to___just_the_directory", kernelname = None, t_float = "float", 
#tA1 tB0 tC0 CM1 
a_transposed = 1, b_transposed = 0, c_transposed = 0, is_col_major = 1, 
#x
micro_tile_width = 2, 
#y
micro_tile_height = 2, 
#X
macro_tile_width = 32, 
#Y
macro_tile_height = 32, 
#U
unroll = 16, 
#P
pad = 1, 
#GA
group_allocation = 3, 
#APLU
work_item_load_a_pll_to_unroll = 0, 
#BPLU
work_item_load_b_pll_to_unroll = 1, 
#PU
unroll_pragma = 1, 
#LIW
load_to_lds_interwoven = 0, 
#MIW
c_micro_tiles_interwoven = 1, 
#ET
use_edge_trick = 1,  
#ICE
n_work_items_per_c_elm = 3,
#UFO
unroll_for_offset = 0,
 # haven't optimised over this
n_target_active_workgroups = 64
):
  """
dir_name : string
  The directory to write the kernel to

kernelname : string or None
  if None, a default kernelname will be generated based on the parameters. 
  default name looks something like this (16 Nov 2016, subject to change)
  A0B0C0f32___Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW0_ET1_ICE1.cl
  where:
  A0B0C0f32 is for transpose cases and float types
  Y128_X128_y8_x8 is the macro and micro tile sizes (Y,y for height)
  U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW0_ET1_ICE1
  means unroll 8, padding 1, group_allocation 1, load_to_lds_interwoven  0, 
  c_micro_tiles_interwoven 1, edge trick 1, n_work_items_per_c_elm 1.
  TODO : include is_col_major and n_target_active_workgroups into the kernel name.
  
a_transposed, b_transposed, c_transposed : 0 or 1
  op(C) = alpha * op(A) * op(B) + beta * op(C). We include op(C) which is non-standard for GEMM interfaces. 
  
is_col_major : 0 or 1
  Whether the matrices are column major ala fortran (1) or row major ala C (0)
  
t_float : "float" or "double" 
  currently either "float" or "double"

micro_tile_width, macro_tile_height : positive int
  as per usual, micro tile size in C (not in op(C))
  
macro_tile_width, macro_tile_height : positive int
  as per usual, macro tile size in C (not in op(C))
  
unroll : positive int
  as per usual, how many slivers in the k-direction a workgroup loads at a time fom global
  
pad : non-negative int
  padding in LDS (see a generated kernel for explanation)

group_allocation : 1,2 or 3
  how to allocate a work group to region of C, either 1 (row-by-row) 2 (column-by-column) or 3 (blocked). 
  see a generated kernel for further explanation
  
work_item_load_a_pll_to_unroll, work_item_load_a_pll_to_unroll : 0 or 1
  The values in global which a work item loads can either be parallel to the unroll direction (1) or 
  perpendicular to it (0). This is true for both load_to_lds_interwoven 0 and 1. 

unroll_pragma : 0 or 1
  whether to insert #pragma unroll before certain for loops in the kernel

load_to_lds_interwoven : 0 or 1
  Tensile has interwoven (1) loads.
  The alternative (0) corresponds to a work item loading contiguous values in C.
  Note that by contiguous we don't nec mean contiguous in memory, just in C (row or col)
  
c_micro_tiles_interwoven : 0 or 1
  The cells of C within a macro tile which a work item computes can either be interwoven (1, like Tensile)
  or not (0). The case 0 corresponds to a true micro tile in the sense that the micro tile is contiguous in C

use_edge_trick : 0 or 1
  Whether the pointer shifting trick is used. This should always be 1. 
  
n_work_items_per_c_elm : positive integer
  The splitting-in-k parameter. 1 corresponds the standard approach, where a thread computes all k values.
  If greater than 1, ONLY the ` + alpha A*B ' part of GEMM is performed, the ` beta C ' part needs to be performed
  by a custom kernel. (Currently betackernel_f32.cl or betackernel_f64.cl)

n_target_active_workgroups : positive integer (default is 64)
  only relevant when group allocation is 3, this corresponds the number of work groups per blocks (up-to rounding))

unroll_for_offset : relegated to fixed parameter
   this was used an attempt to help with the case lda/ldb powers of 2 (4096). 
   In retrospect, it could only have helped if the problem were inter-wg which it is not.
   The problem (I currently believe, 16 Nov) is with-in workgroup. The values needed for loading by a workgroup   
    
Further comments:
  This kernel generator is currently part of tinygemm, the end user should not need to run use this script directly. 
  tinygemm provides help in determining the {local, global}_work_size OpenCL parameters for the kernel(s).
  tinygemm does several tests on the kernel before compiling and running, to check that it generated correctly.

  """


   
  split_on_k = 0 if n_work_items_per_c_elm == 1 else 1
  does_alpha_c_inc = not split_on_k

  if not c_micro_tiles_interwoven:
    strided_i_vertical = "i"
    strided_i_horizontal = "i"
    
  else:
    strided_i_vertical = "i*N_MICRO_TILES_VERTICALLY"
    strided_i_horizontal = "i*N_MICRO_TILES_HORIZONTALLY"




  def get_group_allocation_string():
    ga_comments = {1:"""
  /* GROUP_ALLOCATION = 1 :  allocation is done column-by-column */""", 
  
  2:"""
  /* GROUP_ALLOCATION = 2 :  allocation is done row-by-row */""", 
  
  3:"""
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
   * */"""}
  
    group_allocation_string = ga_comments[group_allocation]
  
  
      
    if group_allocation==1:
      group_allocation_string += """
  const unsigned group_id_vertical = group_id_xy % n_groups_vertically;
  const unsigned group_id_horizontal = group_id_xy / n_groups_vertically;"""
        
    elif group_allocation==2:
      group_allocation_string += """
  unsigned group_id_horizontal = group_id_xy % n_groups_horizontally;
  unsigned group_id_vertical = group_id_xy / n_groups_horizontally;"""
        
    elif group_allocation==3:      
      group_allocation_string += """  
  unsigned group_id_horizontal;
  unsigned group_id_vertical;
  unsigned wg_super_column = group_id_xy / (SUPER_COLUMN_WIDTH*n_groups_vertically);
  unsigned last_super_column_width = n_groups_horizontally % SUPER_COLUMN_WIDTH;
  
  if (group_id_xy < (n_groups_horizontally - last_super_column_width)*n_groups_vertically){
    group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % SUPER_COLUMN_WIDTH;
    group_id_vertical = (group_id_xy / SUPER_COLUMN_WIDTH) % n_groups_vertically;
  }
  else{
    group_id_horizontal = wg_super_column * SUPER_COLUMN_WIDTH + group_id_xy % last_super_column_width;
    group_id_vertical = (group_id_xy  - (n_groups_horizontally - last_super_column_width)*n_groups_vertically) / last_super_column_width;
  }
"""
    else:
      raise RuntimeError("Invalid group_allocation parameter : %d. It should be one 1/2/3."%(group_allocation,))
  
    return group_allocation_string

  def get_super_column_width_defn():
    if group_allocation != 3:
      return ""
    else:
      super_column_width = None
      if split_on_k:
        super_column_width = int(math.sqrt((n_target_active_workgroups + 0.) / n_work_items_per_c_elm))
      else:
        super_column_width = int(math.sqrt((n_target_active_workgroups + 0.)))
      return """/* This variable defines the width of super-columns (we have GROUP_ALLOCATION 3). It is ~ sqrt (N_TARGET_ACTIVE_WORKGROUPS / N_WORK_ITEMS_PER_C_ELM) */
#define SUPER_COLUMN_WIDTH %d"""%(super_column_width,)
    

  def get_inttype_for_atomics():
    # (1) check that the request makes sense (split_in_k != 1)
    
    if n_work_items_per_c_elm == 1:
      raise RuntimeError("The request for inttype for atomics is strange, given that n_work_items_per_c_elm is 1.")
    
    if t_float == "float":
      return "uint"
    
    elif t_float == "double":
      return "ulong"

    else:
      raise RuntimeError("in get_inttype_for_atomics in make_kernel, and don't recognise the t_float, %"%(t_float,))
  
  def get_split_on_k_vardecl_write_string():
    
    split_on_k_vardecl_write_string = None
    
    if split_on_k == 0:
      split_on_k_vardecl_write_string = """
  """
    
    else:
      split_on_k_vardecl_write_string = """
  /* the following variables are used in implementing a basic atomic increment */
  global TFLOAT * ptr_to_c_elm;  // with `restrict' is no faster
  TFLOAT previous_value;
  %s newVal;
  %s prevVal;
  
  """%(get_inttype_for_atomics(), get_inttype_for_atomics())
    
    return split_on_k_vardecl_write_string
  


  def get_pragma_unroll_string(indent):
    pragma_unroll_string = ""
    if unroll_pragma != 0:
      for i in range(indent):
        pragma_unroll_string += "  "
      pragma_unroll_string += "#pragma unroll\n"
  
    return pragma_unroll_string

  def get_a_load_for_perp():
    varname = "mu_perp_i"
    
    if not load_to_lds_interwoven:
      bound_string = "MICRO_A_TILE_PERP_UNROLL"
      increment_string = "++mu_perp_i"
      

    else:
      bound_string = "MACRO_TILE_HEIGHT"
      increment_string = "mu_perp_i += MACRO_TILE_HEIGHT/MICRO_A_TILE_PERP_UNROLL"

    return get_loop_var_bound_incr(varname, bound_string, increment_string)

  def get_loop_var_bound_incr(varname, bound_string, increment_string):
    return  """for (unsigned %s = 0; %s < %s; %s)"""%(varname, varname, bound_string, increment_string)

    
  def get_a_load_for_pll():    
    varname = "mu_pll_i"
    if not load_to_lds_interwoven:
      bound_string = "MICRO_A_TILE_PLL_UNROLL"
      increment_string = "++%s"%(varname,)
  
    else:
      bound_string = "UNROLL"
      increment_string = "%s += UNROLL/MICRO_A_TILE_PLL_UNROLL"%(varname,)
    
    return get_loop_var_bound_incr(varname, bound_string, increment_string)

  def get_b_load_for_perp():
    varname = "mu_perp_i"
    if not load_to_lds_interwoven:
      bound_string = "MICRO_B_TILE_PERP_UNROLL"
      increment_string = "++%s"%(varname,)
    
    else:
      bound_string = "MACRO_TILE_WIDTH"
      increment_string = "%s += MACRO_TILE_WIDTH/MICRO_B_TILE_PERP_UNROLL"%(varname,)
        
    return get_loop_var_bound_incr(varname, bound_string, increment_string)
    
  
    
  def get_b_load_for_pll():
    varname = "mu_pll_i"
    if not load_to_lds_interwoven:
      bound_string = "MICRO_B_TILE_PLL_UNROLL"
      increment_string = "++%s"%(varname,)
    
    else:
      bound_string = "UNROLL"
      increment_string = "%s += UNROLL/MICRO_B_TILE_PLL_UNROLL"%(varname, )
    
    return get_loop_var_bound_incr(varname, bound_string, increment_string)


  def get_mn_factor_string():
    m_factor = 1 if use_edge_trick else macro_tile_height
    n_factor = 1 if use_edge_trick else macro_tile_width
  
    mn_factor_string = """/* We define values which must be factors of m and n. Again, these have no influence on the running of the kernel */
/* If  use_edge_trick is true, these are just 1 (every m and n are permissible) otherwise they are macro-tile dimensions */
/* They are used by host code during checks for compatibility (for kernels with use_edge_trick false) of tile size with m and n */
#define M_FACTOR %s 
#define N_FACTOR %s"""%(m_factor, n_factor)
  
    return mn_factor_string

  def get_atomic_compare_exchange_function():
    if t_float == "float":
      return "atomic_cmpxchg"
    elif t_float == "double":
      return "atom_cmpxchg"
    else:
      raise RuntimeError("Currently, only types float and double are permitted. Throwing from get_atomic_compare_exchange_function") 
    
  def get_alpha_scaled():
    if not c_micro_tiles_interwoven:
      return "alpha*rC[row][col]"
    else:
      return "alpha*rC[row/N_MICRO_TILES_VERTICALLY][col/N_MICRO_TILES_HORIZONTALLY]"


  def get_final_write_element(atomic_increment, with_beta_scaling, with_alpha_increment):
  
    beta_string = None
    if with_beta_scaling:
      beta_string = "c[index] *= beta; "
      
    else:
      beta_string =  ""

    alpha_string = None
    if not with_alpha_increment:
      alpha_string = ""
      
    else:
      if not atomic_increment:
        alpha_string = """
      c[index] += %s;"""%(get_alpha_scaled())        

      else:
        alpha_string = """
      ptr_to_c_elm = c + index;
      do {
        previous_value = *ptr_to_c_elm;
        prevVal = as_%s(previous_value);
        newVal = as_%s(%s + previous_value);
      } while (%s(( __global %s *)(ptr_to_c_elm), prevVal, newVal) != prevVal);"""%(get_inttype_for_atomics(), get_inttype_for_atomics(), get_alpha_scaled(), get_atomic_compare_exchange_function(), get_inttype_for_atomics())
      
    
    final_write_element = """
      index = row_stride_c*(write_start_row + row) + col_stride_c*(write_start_col + col);
      %s
      %s
      """%(beta_string, alpha_string)
    
    
    return final_write_element
  
 
    
  def get_for_loops_for_c_write(inside_the_loops):
    
    height_for_loop = None
    width_for_loop = None
    
    if not c_micro_tiles_interwoven:
      height_for_loop = get_loop_var_bound_incr("row", "MICRO_TILE_HEIGHT", "++row")
      width_for_loop = get_loop_var_bound_incr("col", "MICRO_TILE_WIDTH", "++col")
    
    else:
      height_for_loop = get_loop_var_bound_incr("row", "MACRO_TILE_HEIGHT", "row += N_MICRO_TILES_VERTICALLY")
      width_for_loop = get_loop_var_bound_incr("col", "MACRO_TILE_WIDTH", "col += N_MICRO_TILES_HORIZONTALLY")
      
    return """
  /* loops for writing to c */
%s  %s {
%s    %s {
      %s
    }
  }"""%(get_pragma_unroll_string(1), height_for_loop, get_pragma_unroll_string(2), width_for_loop, inside_the_loops)
  
  def get_id_beta_scaling_wrapped(a_string):
    
    raise RuntimeError("get_id_beta_scaling_wrapped is obsolete")

    if not split_on_k:
      raise RuntimeError("It is strange, almost certainly incorrect, that beta scale wrapping is being used with split_on_k False")
      
    return  """
      /* only one thread will scale c */
      if (true){ //group_id_z == (n_work_groups_with_1_more + 1)%N_WORK_ITEMS_PER_C_ELM){""" + """
        %s
      }
      barrier (CLK_LOCAL_MEM_FENCE);
      barrier (CLK_GLOBAL_MEM_FENCE);"""%(a_string,)

  def get_check_wrapped(a_string):
    return """
      /* catching the write cases for lower(l), right(r) and lr-corner tiles */
     if (
       ((write_start_col + col >= MACRO_TILE_WIDTH*(n_groups_horizontally - 1)) && group_id_vertical   != (n_groups_vertically   - 1 )) ||
       ((write_start_row + row >= MACRO_TILE_HEIGHT*(n_groups_vertically - 1 )) && group_id_horizontal != (n_groups_horizontally - 1 )) ||
       (
       group_id_vertical == (n_groups_vertically-1)     && 
       group_id_horizontal == (n_groups_horizontally-1) && 
       write_start_col + col >= MACRO_TILE_WIDTH*(n_groups_horizontally - 1) && 
       write_start_row + row >= MACRO_TILE_HEIGHT*(n_groups_vertically - 1)
       )){
        %s
      }"""%(a_string,)    



  def get_checked_wrapped_loops_from_bools(with_check, atomic_increment, with_beta_scaling, with_alpha_increment):
    a_base_string = get_final_write_element(atomic_increment, with_beta_scaling, with_alpha_increment)
    if not with_check:
      return  get_for_loops_for_c_write(a_base_string)
    else:
      return  get_for_loops_for_c_write(get_check_wrapped(a_base_string))
      

  def  get_final_write_loops(with_check):    
    if not split_on_k:
      return get_checked_wrapped_loops_from_bools(with_check = with_check, atomic_increment = False, with_beta_scaling = True, with_alpha_increment = True)
    
    else:
      return get_checked_wrapped_loops_from_bools(with_check = with_check, atomic_increment = True, with_beta_scaling = False, with_alpha_increment = True)


  def get_final_write_loops_no_check():
    return get_final_write_loops(with_check = False)
  
  def get_final_write_loops_with_check():
    return get_final_write_loops(with_check = True)
   
    
  def get_k_remaining_string():
    k_remaining_string ="""
  unsigned k_remaining = %s;"""%(get_effective_k() + " % UNROLL",)
  
    return k_remaining_string


  def get_final_unroll_string():
    if not split_on_k:
      final_unroll_string = """
    %s
  """%(get_relocate_load_math_string(final_unroll = True, special_first_unroll = False))


  
    else:
      final_unroll_string = """
  
  /* There is one workgroup which will process the remainder (less that UNROLL) */
  /* JN 16 Nov 2016 */
  if (group_id_z == n_work_groups_with_1_more && k_remaining > 0){%s
  }
  """%(get_indented(get_relocate_load_math_string(final_unroll = True, special_first_unroll = False)))


    
    return final_unroll_string
    
  def get_first_unroll_block():
    first_unroll_block = None
    if not unroll_for_offset :
      first_unroll_block  = ""
    else:
      
      pre_main_loop_math_string = get_relocate_load_math_string(final_unroll = False, special_first_unroll = True)
      first_unroll_block = """
  
  /* This is where the first unroll will be performed. Identical to what is in the main while, but with zero buffering.  */"""
 
      if not split_on_k:
        first_unroll_block += """
    %s
    --n_unrolls_remaining;    
  """%(pre_main_loop_math_string,)
  
      
      else: 
        first_unroll_block += """
  if (group_id_z == 0){
    %s
    --n_unrolls_remaining;
  }
      """%(get_indented(pre_main_loop_math_string),)

    return first_unroll_block;

    
  def get_load_ab_into_LDS_string(final_unroll, special_first_unroll = False):
    """
    simple for loops. Could consider unrolling like Cobalt, but for the moment I use the optional pragma unroll
    """
    
    if special_first_unroll and final_unroll:
      raise RuntimeError("From get_load_ab_into_LDS > It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out")
      
    if final_unroll:
      a_value_to_get = """(a_offset_pll_unroll + mu_pll_i) < k_remaining ? a[mu_pll_i*col_stride_a + mu_perp_i*row_stride_a] : 0;"""
      b_value_to_get = """(b_offset_pll_unroll + mu_pll_i) < k_remaining ? b[mu_pll_i*row_stride_b + mu_perp_i*col_stride_b] : 0;"""
      a_comment =  """
  /* load final bit of data from a into LDS, less than a full unroll */"""
      b_comment =  """
  /* load final bit of data from b into LDS, less than a full unroll */"""

    elif special_first_unroll:
      a_value_to_get = """(a_offset_pll_unroll + mu_pll_i) >= unroll_offset ? a[mu_pll_i*col_stride_a + mu_perp_i*row_stride_a] : 0;"""
      b_value_to_get = """(b_offset_pll_unroll + mu_pll_i) >= unroll_offset ? b[mu_pll_i*row_stride_b + mu_perp_i*col_stride_b] : 0;"""
      a_comment =  """
  /* load first bit of data from a into LDS, ignoring the prepended values (less than a full unroll)  */"""
      b_comment =  """
  /* load first bit of data from b into LDS, ignoring the prepended values (less than a full unroll) */"""
      

    else:
      a_value_to_get = """a[mu_pll_i*col_stride_a + mu_perp_i*row_stride_a];"""
      b_value_to_get = """b[mu_pll_i*row_stride_b + mu_perp_i*col_stride_b];"""
      a_comment =  """
  /* load data from a into LDS */"""
      b_comment =  """
  /* load data from b into LDS */"""
    



    
    

    load_ab_string =  """%s
%s  %s{
%s    %s{
      localA[MACRO_TILE_HEIGHT_AND_PAD*(a_offset_pll_unroll + mu_pll_i) + (a_offset_perp_unroll + mu_perp_i)] = 
      %s
    }
  }
"""%(a_comment, 
get_pragma_unroll_string(1),
get_a_load_for_perp(), 
get_pragma_unroll_string(2),
get_a_load_for_pll(), 
a_value_to_get)

    load_ab_string += """%s
%s  %s{
%s    %s{
      localB[MACRO_TILE_WIDTH_AND_PAD*(b_offset_pll_unroll + mu_pll_i) + (b_offset_perp_unroll + mu_perp_i)] = 
      %s
    }
  }
"""%(b_comment,
get_pragma_unroll_string(1),
get_b_load_for_perp(), 
get_pragma_unroll_string(2),
get_b_load_for_pll(), 
b_value_to_get)
    
    load_ab_string += """
%s"""%(get_worktime_increment_ab(final_unroll),)

    return load_ab_string
    

  def get_swap_string():
    return """
    temp = rA;
    rA = rA_next;
    rA_next = temp;
    
    temp = rB;
    rB = rB_next;
    rB_next = temp;"""
    
  
  
  def get_compute_string(rA_index_maths_string, rB_index_maths_string):
    return """
%s    for (unsigned row = 0; row < MICRO_TILE_HEIGHT; ++row){
%s      for (unsigned col = 0; col < MICRO_TILE_WIDTH; ++col){
        rC[row][col] += rA%s[row]*rB%s[col]; // rA[row]*rB[col];  //mad(rA[row],rB[col],rC[row][col]);
      }
    }"""%(get_pragma_unroll_string(2), get_pragma_unroll_string(3), rA_index_maths_string, rB_index_maths_string)

  
  def get_micro_offset_string():
    """
    This returns the section which makes the within work-group adjust to a, b to
    put a work item in the correct position to load its first element from global
    if the load tiles are interlaced (ala cobalt), this final offset is just 1
    row or column. If the tiles are not interlaced, this final offset is the width
    or height of the load tile. 
    """
    if load_to_lds_interwoven:
      str_a_n_pll, str_a_n_perp, str_b_n_pll, str_b_n_perp = "", "", "", ""
    
    else:
      str_a_n_pll = "MICRO_A_TILE_PLL_UNROLL *"
      str_a_n_perp = "MICRO_A_TILE_PERP_UNROLL *"
      str_b_n_pll = "MICRO_B_TILE_PLL_UNROLL *"
      str_b_n_perp = "MICRO_B_TILE_PERP_UNROLL *"
      
    return """
  /* make the micro adjustments (A) for the thread, getting ready to load */
  const unsigned a_offset_pll_unroll = %s pll_unroll_a_load_id; 
  const unsigned a_offset_perp_unroll = %s perp_unroll_a_load_id;  
  a += col_stride_a * a_offset_pll_unroll;
  a += row_stride_a * a_offset_perp_unroll;

  /* make the micro adjustments (B) for the thread, getting ready to load */
  const unsigned b_offset_pll_unroll = %s pll_unroll_b_load_id;
  const unsigned b_offset_perp_unroll = %s perp_unroll_b_load_id;
  b += row_stride_b * b_offset_pll_unroll;
  b += col_stride_b * b_offset_perp_unroll;"""%(str_a_n_pll, str_a_n_perp, str_b_n_pll, str_b_n_perp)

  def get_unrolled_loop(frag, loop_size):
    unrolled_loop = """
    """
    for i in range(loop_size):
      unrolled_loop = """
      %s
      %s"""%(unrolled_loop, frag)
    
    return unrolled_loop


  def get_load_load_string(rA_index_string, rB_index_string):
    if rA_index_string != "":
      raise RuntimeError("rA_index_string is not `'. I am raising this error out of precaution, as n_prefetch_for_register_load is 0")
    if rB_index_string != "":
      raise RuntimeError("rB_index_string is not `'. I am raising this error out of precaution, as n_prefetch_for_register_load is 0")


     
    return """
%s    for (unsigned i = 0; i < MICRO_TILE_HEIGHT; ++i){
      rA%s[i] = lA[%s];
    }
    lA += MACRO_TILE_HEIGHT_AND_PAD;
    
%s    for (unsigned i = 0; i < MICRO_TILE_WIDTH; ++i){
      rB%s[i] = lB[%s];
    }
    lB += MACRO_TILE_WIDTH_AND_PAD;"""%(get_pragma_unroll_string(2), rA_index_string, strided_i_vertical, get_pragma_unroll_string(2), rB_index_string, strided_i_horizontal)

  def get_localA_localB_decl_string():
    
    return """
  __local TFLOAT localA[N_ELEMENTS_IN_PADDED_A_UNROLL];
  __local TFLOAT localB[N_ELEMENTS_IN_PADDED_B_UNROLL];
  """
    
    
  def get_math_section(use_k_remaining):
    """
    We previously had a variable unroll_the_math_section = False.
    Experiments with unroll_the_math_section suggest that it's a bad idea.
    """
    
    math_section = "None"
    pre_main_loop_load_string = ""
    post_main_loop_maths_string = ""

  
    number_of_unrolls = None

    if use_k_remaining == False:
      number_of_unrolls = "UNROLL"
    else:
      number_of_unrolls = "k_remaining"

 
    load_load_string = get_load_load_string("", "")
    compute_string = get_compute_string("", "")
    
    
    
    math_section = """%s
  for (unsigned u = 0; u < %s; ++u){
    %s 
    %s
  }
  %s """%(pre_main_loop_load_string, number_of_unrolls, load_load_string, compute_string, post_main_loop_maths_string)

    return math_section



  def get_preshift_defns():
    
    preshift_defns = None
    
    if use_edge_trick:
      preshift_defns = """
  const unsigned preshift_bottommost_tile_height = 1 + (m - 1) % MACRO_TILE_HEIGHT; // 1 ... MACRO_TILE_HEIGHT
  const unsigned preshift_rightmost_tile_width = 1 + (n - 1) % MACRO_TILE_WIDTH; // 1 ... MACRO_TILE_WIDTH
  """
    else:
      preshift_defns = """
  """
    
    return preshift_defns
  

  def get_special_case_edge_trick_string():
    
    special_case_edge_trick_string = None
  
    if use_edge_trick:
      special_case_edge_trick_string = """
  /* Special case of the tile being on far right : pull the tile to the left just enough so that it doesn't overflow C */
  if (group_id_horizontal == n_groups_horizontally - 1){
    macro_tile_start_col_in_c -= (MACRO_TILE_WIDTH - preshift_rightmost_tile_width);
  }
    
  /* Special case of the tile being on the bottom : pull the tile up just enough so that it doesn't overflow C */    
  if (group_id_vertical == n_groups_vertically - 1){
    macro_tile_start_row_in_c -= (MACRO_TILE_HEIGHT - preshift_bottommost_tile_height);
  }"""
    
    else:
      special_case_edge_trick_string = """
      """
    
    return special_case_edge_trick_string

  def get_group_allocation_defn_string():
    if group_allocation not in [1,2,3]:
      raise RuntimeError("Invalid group_allocation value %s, it should be in [1,2,3]."%(group_allocation,))
      
    group_allocation_string = """
#define GROUP_ALLOCATION %d"""%(group_allocation,)

    if group_allocation == 3:
      group_allocation_string += """/* this variable is declared because we have GROUP_ALLOCATION type 3. */
/* It should define how many workgroups we expect to have active simulantaneuosly. */
#define N_TARGET_ACTIVE_WORKGROUPS %d"""%(n_target_active_workgroups,)
    return group_allocation_string

  def get_ngroups_grid_string():
    ngroups_grid_string = """
  /* the number of work groups vertically and horizontally. */
  /* note that this ignores n_work_items_per_c_elm, so that only one workgroup per c cell is considered */ """
    
    if use_edge_trick == 1:
      ngroups_grid_string += """
  const unsigned n_groups_vertically = m / MACRO_TILE_HEIGHT + (preshift_bottommost_tile_height != MACRO_TILE_HEIGHT);
  const unsigned n_groups_horizontally = n / MACRO_TILE_WIDTH + (preshift_rightmost_tile_width != MACRO_TILE_WIDTH);    
  """
    else:
      ngroups_grid_string += """
  const unsigned n_groups_vertically = m / MACRO_TILE_HEIGHT;
  const unsigned n_groups_horizontally = n / MACRO_TILE_WIDTH;    
  """
    return ngroups_grid_string

  def get_c_work_item_vertical_next():
    if c_micro_tiles_interwoven:
      return "1" 
    else:
      return "MICRO_TILE_HEIGHT"
  
  def get_c_work_item_horizontal_next():
    if c_micro_tiles_interwoven:
      return "1" 
    else:
      return "MICRO_TILE_WIDTH"

  
  def get_relocate_lAlB_string(final_unroll):
    lds_memory_index_string = ""
    return """
  lA = localA%s + micro_id_vertical*%s;
  lB = localB%s + micro_id_horizontal*%s;"""%(lds_memory_index_string, get_c_work_item_vertical_next(), lds_memory_index_string, get_c_work_item_horizontal_next())
  

  def get_final_write_all():
    
    final_write_all = None
    if use_edge_trick:
      final_write_all = """
  /* the case where this is not an edge tile : will write to all cells */
  if ((group_id_horizontal != n_groups_horizontally - 1 || preshift_rightmost_tile_width == MACRO_TILE_WIDTH) 
  && (group_id_vertical != n_groups_vertically - 1 || preshift_bottommost_tile_height == MACRO_TILE_HEIGHT)){
    %s
  }
    
  else{
    %s
  }"""%(get_indented(get_final_write_loops_no_check()),  get_indented(get_final_write_loops_with_check()))
  
    else:
      final_write_all = """
    %s"""%(get_final_write_loops_no_check())
  
  
    return final_write_all

  def get_split_on_k_ab_offset_adjustment_string():
    split_on_k_ab_offset_adjustment_string = None
    
    if not split_on_k:
      split_on_k_ab_offset_adjustment_string = """
  """
    else:
      split_on_k_ab_offset_adjustment_string = """
  
  /* a,b are pointing to the top left of the region required by the macro tile, but this work group  */
  /* might not process the whole of a and b. We now turn 90 and shift pointers a,b to the start for this wg */
  a += UNROLL*group_id_z*col_stride_a;
  b += UNROLL*group_id_z*row_stride_b;
  """
      
    return split_on_k_ab_offset_adjustment_string

  def get_k_unroll_offset_initial_string():
    k_unroll_offset_initial_string = None
    
    if not unroll_for_offset:
      k_unroll_offset_initial_string = ""
    else:
      k_unroll_offset_initial_string = """
  /* this additional offset of a and b appears because UNROLL_FOR_OFFSET is 1 */
  unsigned unroll_offset = (3*group_id_vertical + 11*group_id_vertical)%UNROLL;
  unsigned k_plus_offset = k + unroll_offset;
  a -= unroll_offset*col_stride_a;
  b -= unroll_offset*row_stride_b;
  """
    return k_unroll_offset_initial_string

  def get_effective_k():
    if not unroll_for_offset:
      return "k"
    else:
      return "k_plus_offset"
    
  def get_split_on_k_defns_string():
    split_on_k_defns_string = None
    if not split_on_k:
      split_on_k_defns_string = ""
    else:
      split_on_k_defns_string = """
/* the cumulative unroll. */
/* For the (standard) case of N_WORK_ITEMS_PER_C_ELM = 1, G_UNROLL would just be UNROLL*/
#define G_UNROLL %d // N_WORK_ITEMS_PER_C_ELM*UNROLL"""%(n_work_items_per_c_elm*unroll)

    return split_on_k_defns_string
    
  def get_relocate_load_math_string(final_unroll, special_first_unroll):
 
    if final_unroll and special_first_unroll:
      raise RuntimeError("From get_relocate_load_math_string : It is not possible for this to be both a `special_first_unroll' and a `final_unroll'. This is a logic error, broken alg, come and sort it out")
  
    
    base_load_ab_string = get_load_ab_into_LDS_string(final_unroll, special_first_unroll)    
    load_ab_string = base_load_ab_string


    
    return """
  %s  
  /* make sure all loads from LDS memory have completed */
  barrier(CLK_LOCAL_MEM_FENCE);
  %s
  %s
  /* make sure all maths is complete, so that we can pull in the next unroll slice (if there is one)  */
  barrier(CLK_LOCAL_MEM_FENCE);"""%(load_ab_string, get_relocate_lAlB_string(final_unroll), get_math_section(final_unroll))


  def get_group_id_defns():
    group_id_defns = None
    if not split_on_k:
      group_id_defns = """
  const unsigned group_id_xy = get_group_id(0);"""
    else:
      group_id_defns = """
  const unsigned group_id = get_group_id(0);
  const unsigned group_id_xy = group_id / N_WORK_ITEMS_PER_C_ELM;
  const unsigned group_id_z = group_id % N_WORK_ITEMS_PER_C_ELM;"""
  
    return group_id_defns

  def get_stride_defn(LETTER, letter, ldx, transposed):
    transposed_xor_is_col_major = (transposed + is_col_major) % 2
    
    row_string_string =  """
  /* To move from %s[row][col] to %s[row+1][col], how much should the pointer increment? As we have %s_TRANSPOSED = %d and IS_COL_MAJOR = %d, this is */
  const unsigned row_stride_%s = %s;"""%(LETTER, LETTER, LETTER, transposed, is_col_major, letter, 1 if transposed_xor_is_col_major else ldx) 
   
    col_stride_string = """
  /* To move from %s[row][col] to %s[row][col+1], how much should the pointer increment? As we have %s_TRANSPOSED = %d and IS_COL_MAJOR = %d, this is */
  const unsigned col_stride_%s = %s;"""%(LETTER, LETTER, LETTER, transposed, is_col_major, letter, ldx if transposed_xor_is_col_major else 1)

    return "%s%s"%(row_string_string, col_stride_string)
    
  def get_stride_defns():
    
    return """
  /*a performance note : moving these (row_stride_x, col_stride_x) definitions to precompiler does not improve memory use or speed. */
  %s
  %s
  %s"""%(get_stride_defn("A", "a", "lda", a_transposed),  get_stride_defn("B", "b", "ldb", b_transposed), get_stride_defn("C", "c", "ldc", c_transposed))
    
  def get_worktime_increment_ab(final_unroll):
    n_jumps = None
    if not split_on_k:
      n_jumps_string = "UNROLL"
    else:
      n_jumps_string = "G_UNROLL"
    
    if final_unroll is True:
      return """"""
    
    else:
      return """
  a += col_stride_a*%s;
  b += row_stride_b*%s;  
    """%(n_jumps_string, n_jumps_string)


  def get_rA_rB_decl_string():
    return """
  TFLOAT rA[MICRO_TILE_HEIGHT];
  TFLOAT rB[MICRO_TILE_WIDTH];"""
    
  
  def get_n_unrolls_remaining_string():
    k_effective_mod_G_UNROLL = get_effective_k() + " % G_UNROLL"
    k_effective_div_G_UNROLL = get_effective_k() + " / G_UNROLL"
    k_effective_div_UNROLL = get_effective_k() + " / UNROLL"

    n_unrolls_remaining_string = None
    if not split_on_k :
      n_unrolls_remaining_string = """
  int n_unrolls_remaining = %s;"""%(k_effective_div_UNROLL,)
  
    else:
      n_unrolls_remaining_string = """
  const int n_work_groups_with_1_more = (%s) / UNROLL; //JN_NEW
  // branching between work groups : some wgs have 1 more unroll to process.
  int n_unrolls_remaining = (%s) +  (group_id_z < n_work_groups_with_1_more);  """%(k_effective_mod_G_UNROLL, k_effective_div_G_UNROLL)
  
    return n_unrolls_remaining_string
  
        
  def get_what_string():
    return "A%dB%dC%df%d"%(a_transposed, b_transposed, c_transposed, float_size,)
  
  def get_kernelname():
    
    if kernelname:
      return kernelname
    
    else:
      return "%s___Y%d_X%d_y%d_x%d_U%d_P%d_GA%d_APLU%d_BPLU%d_PU%d_LIW%d_MIW%d_ET%d_ICE%d_UFO%d"%(get_what_string(), macro_tile_height, macro_tile_width, micro_tile_height, micro_tile_width, unroll, pad, group_allocation, work_item_load_a_pll_to_unroll, work_item_load_b_pll_to_unroll, unroll_pragma, load_to_lds_interwoven, c_micro_tiles_interwoven, use_edge_trick, n_work_items_per_c_elm, unroll_for_offset)

  if t_float == "float":
    float_size = 32
  elif t_float == "double":
    float_size = 64
  else:
    raise RuntimeError("Unrecognised t_float : ", t_float)
  
  macro_tile_area = macro_tile_width * macro_tile_height
  micro_tile_area = micro_tile_width * micro_tile_height
  
  n_workitems_per_workgroup = macro_tile_area / micro_tile_area
  macro_tile_height_and_pad = macro_tile_height + pad
  macro_tile_width_and_pad = macro_tile_width + pad
  n_elements_in_a_unroll = macro_tile_height * unroll
  n_elements_in_b_unroll = macro_tile_width * unroll
  n_elements_in_padded_a_unroll = macro_tile_height_and_pad * unroll
  n_elements_in_padded_b_unroll = macro_tile_width_and_pad * unroll
  
  if (n_elements_in_a_unroll % n_workitems_per_workgroup != 0):
    raise RuntimeError("this is not supported : n_workitems_per_workgroup (%d) is not a factor of n_elements_in_%s_unroll (%d). Consider rounding unroll up. "%(n_workitems_per_workgroup, "a", n_elements_in_a_unroll))
  
  if (n_elements_in_b_unroll % n_workitems_per_workgroup != 0):
    raise RuntimeError("this is not supported : n_workitems_per_workgroup (%d) is not a factor of n_elements_in_%s_unroll (%d). Consider rounding unroll up. "%(n_workitems_per_workgroup, "b", n_elements_in_b_unroll))
    
  n_elements_of_a_to_load_per_workitem = n_elements_in_a_unroll / n_workitems_per_workgroup
  n_elements_of_b_to_load_per_workitem = n_elements_in_b_unroll / n_workitems_per_workgroup
  n_micro_tiles_vertically = macro_tile_height / micro_tile_height
  n_micro_tiles_horizontally = macro_tile_width / micro_tile_width


  ###############  Will now set the following 6 variables #################################################################################
  micro_a_tile_pll_unroll, micro_a_tile_perp_unroll, n_micro_a_tiles_pll_unroll = None, None, None
  micro_b_tile_pll_unroll, micro_b_tile_perp_unroll, n_micro_b_tiles_pll_unroll = None, None, None

  if (work_item_load_a_pll_to_unroll == 0): #tiles are tall perpendicular to the unroll direction
    micro_a_tile_perp_unroll, micro_a_tile_pll_unroll = get_tile_dimensions(macro_tile_height, unroll, n_elements_of_a_to_load_per_workitem)
  
  elif (work_item_load_a_pll_to_unroll == 1): #tiles are tall parallel to the unroll direction
    micro_a_tile_pll_unroll, micro_a_tile_perp_unroll = get_tile_dimensions(unroll, macro_tile_height, n_elements_of_a_to_load_per_workitem)
  
  else:
    errm = "this is strange, not sure what to do with `work_item_load_a_pll_to_unroll = " + str(work_item_load_a_pll_to_unroll) + "'. It should be 0 (for false) or 1 (for true)."
    raise RuntimeError(errm)
    
  n_micro_a_tiles_pll_unroll = unroll / micro_a_tile_pll_unroll


  if (work_item_load_b_pll_to_unroll == 0):
    micro_b_tile_perp_unroll, micro_b_tile_pll_unroll = get_tile_dimensions(macro_tile_width, unroll, n_elements_of_b_to_load_per_workitem)
  
  elif (work_item_load_b_pll_to_unroll == 1):
    micro_b_tile_pll_unroll, micro_b_tile_perp_unroll = get_tile_dimensions(unroll, macro_tile_width, n_elements_of_b_to_load_per_workitem)
  
  else:
    errm = "this is strange, not sure what to do with `work_item_load_a_pll_to_unroll = " + str(work_item_load_a_pll_to_unroll) + "'. It should be 0 (for false) or 1 (for true)."

    raise RuntimeError(errm)
    
  n_micro_b_tiles_pll_unroll = unroll / micro_b_tile_pll_unroll
  ###########################################################################################################################################
    
  
  
  sub_strings = [
"""/* ***********************************************
 * These parameters define WHAT this kernel does *
 * *********************************************** */
#define IS_COL_MAJOR %d
#define A_TRANSPOSED %d
#define B_TRANSPOSED %d
#define C_TRANSPOSED %d
#define TFLOAT  %s
/* certain kernels can only do one or the other of the terms in c <- alpha a*b + beta c. */
/* TODO : if DOES_ALPHA_C_INC is 0, then alpha should not appear as a kernel parameter */
#define DOES_ALPHA_C_INC %d
#define DOES_BETA_A_B_INC 1
"""%(is_col_major, a_transposed, b_transposed, c_transposed, t_float, does_alpha_c_inc), 


"""
/* TODO : remove final barrier, not nec. Check performance is not mysteriously hit! */
/* TODO : beta = 1 optimisation */
/* TODO : investigate mad. When should one use this instead of standard overloads, += and * ? 


/* ****************************************
 * These parameters define HOW it does it *
 * *****************************************/
/* Defines a tile shape. Each thread will process a tile of this shape from C (if N_WORK_ITEMS_PER_C_ELM > 1 the processing is shared with threads in other WGs)  */
#define MICRO_TILE_WIDTH %d
#define MICRO_TILE_HEIGHT %d
/* Area of C which a workgroup will process. Recall that a workgroup is made of several threads which share LDS memory */
#define MACRO_TILE_WIDTH %d
#define MACRO_TILE_HEIGHT %d
/* How much a workgroup load (global -> LDS) in the k-direction at each iteration of the outer-most loop */
#define UNROLL %d
/* padding in LDS to avoid bank conflicts*/
#define PAD %d
/* whether or not this kernel uses the edge trick (see documentation : (TODO, currently internal AMD document)) */
/* this precompiler defn has no direct influence on the running the kernel, implementation already done in make_kernel.py */
#define EDGETRICK %d
/* the number of work items working on the same c element. if this is 1, there will be just one thread doing all k multiply-adds, */
/* otherwise if it is greater than 1, each thread will be computing ~ k / N_WORK_ITEMS_PER_C_ELM of the multiply adds, to be atomically added at the end */ 
#define N_WORK_ITEMS_PER_C_ELM %d
/* whether or not to shimmy the starting k, in an attempt to avoid cache line overuse for cases where lda/ldb are powers of 2 */
/* if 0, no shimmying. if 1, instead of starting at k = 0 workgroups start at some negative offset dependent on work group id */
/* in the same way as the final unroll populates LDS with zeros in k mod UNROLL != 0, the initial negative indices here populate with 0 */
#define UNROLL_FOR_OFFSET %d
"""%(micro_tile_width, micro_tile_height, macro_tile_width, macro_tile_height, unroll, pad, use_edge_trick, n_work_items_per_c_elm, unroll_for_offset), 




""" 
/* define the way in which work groups are assigned to tiles */
/* 1 : column-by-column
 * 2 : row-by-row 
 * 3 : by rows within super-column  */
 %s"""%(get_group_allocation_defn_string(),),

"""
/* Whether the load tiles are long in the direction of unroll (1) or perpendicular to the direction of unroll (0) */
/* Note : if the load tiles are long in the direction of unroll, the destination tile in LDS is NOT contiguous  */
/* We include these parameters here as pre-processor variables, but the loading micro-tile shapes are set in make_kernel.py */
#define WORK_ITEM_LOAD_A_PLL_TO_UNROLL %d 
#define WORK_ITEM_LOAD_B_PLL_TO_UNROLL %d
/* Whether the load tiles are interwoven (ala Cobalt, (1)) or if the load tiles are truly contiguous tiles of A/B (0) */
/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */
#define LOAD_TO_LDS_INTERWOVEN %d
/* Whether the micro tile being processed by a compute item is interwoven with other micro tiles (ala Cobalt, (1)) or if the micro tiles are contiguous in C */
/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */
#define C_MICRO_TILES_INTERWOVEN %d
/* Whether to use the unroll pragma to encourage the compiler to unroll certain loops */
/* Included here for user, in practice it has no direct effect on this kernel, as the relevent implementation has been done in make_kernel.py */
#define PRAGMA_UNROLL_FORLOOPS %d 
/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into registers, as compared to doing the math. */
/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */ 
#define N_PREFETCH_FOR_REGISTER_LOAD 0
/* (deprecated parameter, as of 17 Nov 2016, see git log) How many steps ahead are we reading into LDS, as compared to the unroll loop */
/* This should be the domain of the compiler, and currently (26/08/2016, Catalyst) it seems that leaving this as 0 is best.  */ 
#define N_PREFETCH_FOR_LDS_LOAD 0
"""%(work_item_load_a_pll_to_unroll, work_item_load_b_pll_to_unroll, load_to_lds_interwoven, c_micro_tiles_interwoven, unroll_pragma),

"""



/* *****************************************************************************
 * The following are all implied by the preceding: these are NOT free parameters!
 * **************************************************************************** */
/*  <+_+>
 *           <+_+>
 *      <+_+>
 * TODO : rerereconsider assignment of workitems to load and math regions, there may be some sweet overlap spot where values automatically in registers for math (?) */""", 



""" 
#define MACRO_TILE_AREA %d // MACRO_TILE_WIDTH*MACRO_TILE_HEIGHT  
#define MICRO_TILE_AREA %d // MICRO_TILE_WIDTH * MICRO_TILE_HEIGHT
#define N_WORK_ITEMS_PER_WORKGROUP  %d // MACRO_TILE_AREA / MICRO_TILE_AREA
#define MACRO_TILE_HEIGHT_AND_PAD %d // MACRO_TILE_HEIGHT + PAD
#define MACRO_TILE_WIDTH_AND_PAD %d // MACRO_TILE_WIDTH + PAD
#define N_ELEMENTS_IN_A_UNROLL %d // MACRO_TILE_HEIGHT * UNROLL
#define N_ELEMENTS_IN_B_UNROLL %d // MACRO_TILE_WIDTH * UNROLL
#define N_ELEMENTS_IN_PADDED_A_UNROLL %d // MACRO_TILE_HEIGHT_AND_PAD * UNROLL
#define N_ELEMENTS_IN_PADDED_B_UNROLL %d // MACRO_TILE_WIDTH_AND_PAD * UNROLL
#define N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM %d // N_ELEMENTS_IN_A_UNROLL / N_WORK_ITEMS_PER_WORKGROUP
#define N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM %d // N_ELEMENTS_IN_B_UNROLL / N_WORK_ITEMS_PER_WORKGROUP
#define N_MICRO_TILES_VERTICALLY %d // MACRO_TILE_HEIGHT / MICRO_TILE_HEIGHT
#define N_MICRO_TILES_HORIZONTALLY %d // MACRO_TILE_WIDTH / MICRO_TILE_WIDTH"""%(macro_tile_area, micro_tile_area, n_workitems_per_workgroup, macro_tile_height_and_pad, macro_tile_width_and_pad, n_elements_in_a_unroll, n_elements_in_b_unroll, n_elements_in_padded_a_unroll, n_elements_in_padded_b_unroll, n_elements_of_a_to_load_per_workitem, n_elements_of_b_to_load_per_workitem, n_micro_tiles_vertically, n_micro_tiles_horizontally),

"""
/* MICRO_A_TILE_PLL_UNROLL * MICRO_A_TILE_PERP_UNROLL = N_ELEMENTS_OF_A_TO_LOAD_PER_WORKITEM */
/* The dimensions of a tile in A loaded by a work item.  */
#define MICRO_A_TILE_PLL_UNROLL %d // size of the loaded tile, pll to unroll
#define MICRO_A_TILE_PERP_UNROLL %d
#define N_MICRO_A_TILES_PLL_UNROLL %d // UNROLL / MICRO_A_TILE_PLL_UNROLL"""%(micro_a_tile_pll_unroll, micro_a_tile_perp_unroll, n_micro_a_tiles_pll_unroll), 


"""
/*  MICRO_B_TILE_PLL_UNROLL * MICRO_B_TILE_PERP_UNROLL = N_ELEMENTS_OF_B_TO_LOAD_PER_WORKITEM */
/* The dimensions of a tile in B read by a work item */
#define MICRO_B_TILE_PLL_UNROLL %d
#define MICRO_B_TILE_PERP_UNROLL %d
#define N_MICRO_B_TILES_PLL_UNROLL %d // UNROLL / MICRO_B_TILE_PLL_UNROLL
%s
%s
%s"""%(micro_b_tile_pll_unroll, micro_b_tile_perp_unroll, n_micro_b_tiles_pll_unroll, get_mn_factor_string(), get_split_on_k_defns_string(), get_super_column_width_defn()), 



"""


__attribute__((reqd_work_group_size(%d,1, 1)))"""%(n_workitems_per_workgroup,),

"""
__kernel void gpu_%s("""%(get_kernelname()),




"""  
  __global TFLOAT       *          c,
  __global const TFLOAT * restrict a,
  __global const TFLOAT * restrict b,
  const TFLOAT alpha,
  const TFLOAT beta,
  const unsigned lda,
  const unsigned ldb,
  const unsigned ldc,
  const unsigned m,
  const unsigned n,
  const unsigned k,
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

  %s 
  %s        
  const unsigned local_id = get_local_id(0);
  %s
  %s 
"""%(get_stride_defns(),  get_group_id_defns(), get_preshift_defns(), get_ngroups_grid_string()), 



  

  

"""
%s
"""%(get_group_allocation_string(), ),


"""
  unsigned macro_tile_start_row_in_c = group_id_vertical*MACRO_TILE_HEIGHT;
  unsigned macro_tile_start_col_in_c = group_id_horizontal*MACRO_TILE_WIDTH;

  %s
"""%(get_special_case_edge_trick_string()),

"""
  
  /* move to the top left corner of a (top left corner of b) of region required by the macro tile */
  a += macro_tile_start_row_in_c*row_stride_a;   
  b += macro_tile_start_col_in_c*col_stride_b;""",
  
  get_split_on_k_ab_offset_adjustment_string(),
  
  get_k_unroll_offset_initial_string(),
  
"""  
  /* Define which rows or columns of A, B this thread will load from global into LDS */
  const unsigned pll_unroll_a_load_id = local_id % N_MICRO_A_TILES_PLL_UNROLL;
  const unsigned perp_unroll_a_load_id = local_id / N_MICRO_A_TILES_PLL_UNROLL;
  const unsigned pll_unroll_b_load_id = local_id % N_MICRO_B_TILES_PLL_UNROLL;
  const unsigned perp_unroll_b_load_id = local_id / N_MICRO_B_TILES_PLL_UNROLL;
""",

  get_micro_offset_string(),

"""

  /* Define which part of the C macro-tile this thread will process */
  const unsigned micro_id_vertical = local_id % N_MICRO_TILES_VERTICALLY;
  const unsigned micro_id_horizontal = local_id / N_MICRO_TILES_VERTICALLY;""",
  
  
  """
  
  /* LDS memory */%s"""%(get_localA_localB_decl_string(),),

  """
  /* register memory */
  TFLOAT rC[MICRO_TILE_HEIGHT][MICRO_TILE_WIDTH] = {{0.}};""",

  get_rA_rB_decl_string(), 
  

"""
  
  /* jumping pointers to locate the LDS to load into register memory */
  __local const TFLOAT * lA;
  __local const TFLOAT * lB;
""", 


  get_n_unrolls_remaining_string(),

  get_first_unroll_block(),
    
  
"""
  
  while (n_unrolls_remaining > 0){
    %s
    --n_unrolls_remaining;    
  }
"""%(get_indented(get_relocate_load_math_string(final_unroll = False, special_first_unroll = False ))), 
  
  
  get_k_remaining_string(),
  
  get_final_unroll_string(),
  
      

    
"""  
  const unsigned write_start_row = macro_tile_start_row_in_c + micro_id_vertical*%s;
  const unsigned write_start_col = macro_tile_start_col_in_c + micro_id_horizontal*%s;  
  unsigned index;
  %s
  %s 
}

"""%(get_c_work_item_vertical_next(),  get_c_work_item_horizontal_next(), get_split_on_k_vardecl_write_string(), get_final_write_all() ),

'\0 '
]


  full_string = "".join(sub_strings)
  if os.path.exists(dir_name) and not os.path.isdir(dir_name):
    errm =  "The specified `path', ",  dir_name, " exists but is not a directory. It needs to be directory."
    raise RuntimeError(errm)
  
  if not os.path.isdir(dir_name):
    print "(make_kernel.py) making directory, ", dir_name
    os.makedirs(dir_name)
   
  
     
  filename = os.path.join(dir_name, "%s.cl"%(get_kernelname()))



  filly = open(filename, "w")
  filly.write(full_string)
  filly.close()

  


def get_default_call():
  """
  return a command line call string to generate a kernel with all default parameterss
  """
  function_call_spec = inspect.getargspec(make_kernel)
  function_parms = function_call_spec[0]
  default_values = function_call_spec[3]
  default_call = ''.join(['%s%s'%(x,y) for x,y in zip(['--%s '%(x,) for x in function_parms], ['%s  '%(x,) for x in default_values])])
  default_call = default_call.replace("where_to_write_kernel_to___filename_is_auto_from_parms", ".|.-.|.")
  return default_call


