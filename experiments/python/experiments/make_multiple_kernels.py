import sys
sys.path.append("../../../tinygemm/python")

import make_kernel
reload(make_kernel)

import write_directories
reload(write_directories)

import os
import numpy as np


def make_multiple_kernels(dir_name = os.path.join(write_directories.kernels_base_directory, "some_sub_directory_to_write_kernels"), data_geometry = None, kernel_span = None, double_type = None):
  """
  Write kernels (currently .cl files) for set product of parameters. 
  """

  if not data_geometry or not kernel_span or not double_type:
    raise RuntimeError("Invalid parameters : either data_geometry or kernel_span or double_type is missing")


  t_float = "float"
  if double_type == np.float64:
    t_float = "double"


  #if data_geometry['isColMajor'] == False:
    #raise RuntimeError("isColMajor should be True at this point (make_multiple_kernels)")

  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

  print "\n\n", kernel_span['unroll_for_offsets'], "--------------"

  for macro_tile_height, macro_tile_width, micro_tile_height, micro_tile_width in kernel_span['Y_X_y_x']:
    for unroll in kernel_span['unrolls']:
      for pad in kernel_span['pads']:#8
        for group_allocation in kernel_span['group_allocations']:#16
          for work_item_load_a_pll_to_unroll in kernel_span['work_item_load_a_pll_to_unrolls']: #32
            for work_item_load_b_pll_to_unroll in kernel_span['work_item_load_b_pll_to_unrolls']: #64
              for unroll_pragma in kernel_span['unroll_pragmas']:
                for load_to_lds_interwoven in kernel_span['load_to_lds_interwovens']:
                  for c_micro_tiles_interwoven in kernel_span['c_micro_tiles_interwovens']:
                    for use_edge_trick in kernel_span['use_edge_tricks']:
                      for n_work_items_per_c_elm in kernel_span['n_work_items_per_c_elms']:
                        for unroll_for_offset in kernel_span['unroll_for_offsets']:
                          make_kernel.make_kernel( dir_name = dir_name, a_transposed = data_geometry['tA'], b_transposed = data_geometry['tB'], c_transposed = data_geometry['tC'],is_col_major = data_geometry['isColMajor'], t_float = t_float, micro_tile_width = micro_tile_width, micro_tile_height = micro_tile_height, macro_tile_width = macro_tile_width, macro_tile_height = macro_tile_height, unroll = unroll, pad = pad, group_allocation = group_allocation, work_item_load_a_pll_to_unroll = work_item_load_a_pll_to_unroll, work_item_load_b_pll_to_unroll = work_item_load_b_pll_to_unroll, unroll_pragma = unroll_pragma, load_to_lds_interwoven = load_to_lds_interwoven, c_micro_tiles_interwoven = c_micro_tiles_interwoven, use_edge_trick = int(use_edge_trick), n_work_items_per_c_elm = int(n_work_items_per_c_elm), unroll_for_offset = int(unroll_for_offset))
    
    
  
  
  
  
    
  
