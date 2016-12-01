import os
import numpy as np
import numpy.random as npr
import sys

import utility_functions
reload(utility_functions)   

from IPython.core.debugger import Tracer

sys.path.append("../deepbench")

import baidu_bench
reload(baidu_bench)

import write_directories
reload(write_directories)


def do_find_test(matrix_mnk = {'m': 4001, 'n':4002, 'k':4055}, data_geometry = {'tA':True, 'tB':False, 'tC': False, 'isColMajor':True}):
     
  utility_functions.go_experiment(kernel_savedir = None, kernel_span = None, double_type = np.float32, outputfilename = "", factorwaste = {'a':1, 'b':1, 'c':1}, constantwaste = {'a':0, 'b':0, 'c':0}, n_runs = 5, do_test = False, findfirst = True, allotted_time = 200., matrix_mnk = matrix_mnk, data_geometry = data_geometry, enforce_deterministic = False)



  
def do_geometry_tests(kernel_savedir = None):
  """
  Run through [col/row major] x [tA, ~tA] x [tB, ~tB] x[C tall, C wide] and confirm that all tests pass (compared to correct C computed in python)
  if kernel_savedir is "default", it kernels will be saved in sub-directories of write_directories.output_base_directory
  """
  
 
 
  for local_is_col_major in [False, True]:
    for tA in [True, False]:
      for tB in [True, False]:
        for tC in [True, False]:
          for m,n in [[7*32-1, 2*32+1]]: #,[2*32+1, 7*32-1] ]: #[[1285, 322], [311, 1286]]:            
            for k in [512, 515]:
              data_geometry = {'tA':tA, 'tB':tB, 'tC': tC, 'isColMajor':local_is_col_major}
            
              kernel_span = {'Y_X_y_x' :[[16*2, 16*3, 2, 3]], 'unrolls':[16], 'pads' : [1], 'group_allocations' : [3], 'work_item_load_a_pll_to_unrolls' : [0],'work_item_load_b_pll_to_unrolls' : [1],'unroll_pragmas' : [1], 'load_to_lds_interwovens' : [0], 'c_micro_tiles_interwovens': [1], 'use_edge_tricks':[1], 'n_work_items_per_c_elms':[3], 'unroll_for_offsets':[0]}
            
              matrix_mnk = {'m':m, 'n':n, 'k':k}
  
  
              if isinstance(kernel_savedir, str) and "default" in kernel_savedir:
                siter = 0
                frag = "default_geometry_test_att%d"%(siter)
                kernel_savedir = os.path.join(write_directories.kernels_base_directory, frag)
                while os.path.isdir(kernel_savedir):
                  kernel_savedir = kernel_savedir.replace("att%d"%(siter,), "att%d"%(siter + 1,))
                  siter += 1
            
              
              utility_functions.go_experiment(kernel_savedir = kernel_savedir, data_geometry = data_geometry, matrix_mnk = matrix_mnk, kernel_span = kernel_span, n_runs = 3, outputfilename = None, do_test = True)
    
    
#def do_kernel_tests(outputfilename = "default", kernel_savedir = '/home/james/tinygemmout/kernels/ufotesting', do_test = True, forcefilewrite = True):
#def do_kernel_tests(outputfilename = "default", kernel_savedir = '/home/james/tinygemmout/kernels/test101', do_test = False, forcefilewrite = False):  
def do_kernel_tests(outputfilename = "default", kernel_savedir = None, do_test = False, forcefilewrite = False):
  """
  Current champion : Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_??? TODO : add missing hyper params.
  

  outputfilename 
  'default'    : use a default path
  None         : TODO : check what happens
  otherwise    : save ouput to outputfilename (should be a full path)
  
  kernel_savedir
  'default'    : use a default path
  None         : don't save kernels
  otherwise    : save kernels in kernel_savedir
  
  
  
  """
  
  double_type = np.float32
  

  data_geometry = {'tA':False, 'tB': False, 'tC': False, 'isColMajor':True}
  
  
  matrix_mnk = {'m': 4567, 'n': 4567, 'k': 4567}
  
  micro_tiles = [[8,8]] 
  mm_tiles = [[16*a, 16*b, a, b] for a, b in micro_tiles]
  #with padding 2 I sometimes get better than with padding 1 (!)
  kernel_span = {'Y_X_y_x' :mm_tiles, 'unrolls':[8], 'pads' : [1], 'group_allocations' : [1], 'work_item_load_a_pll_to_unrolls' : [0],'work_item_load_b_pll_to_unrolls' : [1],'unroll_pragmas' : [0], 'load_to_lds_interwovens' : [0], 'c_micro_tiles_interwovens': [1], 'use_edge_tricks':[1], 'n_work_items_per_c_elms':[1], "unroll_for_offsets":[0]}#3,6,8,10]}#,3,4,5,6,7,8]}
  constantwaste = {'a':3, 'b':5, 'c':7}
  
  
  



  base_string = "tA%d_tB%d_tC%d_CM%d___m%d_n%d_k%d"%(data_geometry['tA'], data_geometry['tB'], data_geometry['tC'], data_geometry['isColMajor'], matrix_mnk['m'],matrix_mnk['n'],matrix_mnk['k'])

  if kernel_savedir == 'default':
    attempt = 0
    
    def get_kernel_savedir(attempt):
      return os.path.join(write_directories.kernels_base_directory, "%s_%d"%(base_string, attempt))
    
    kernel_savedir = get_kernel_savedir(attempt)
    while os.path.isdir(kernel_savedir):
      attempt += 1
      kernel_savedir = get_kernel_savedir(attempt)
  
    os.makedirs(kernel_savedir)
      

  if outputfilename == 'default':
    attempt = 0
    
    def get_outputfilename(attempt):
      return os.path.join(write_directories.output_base_directory, "do_kernel_tests_default", "%s_%d.txt"%(base_string, attempt))
    
    outputfilename = get_outputfilename(attempt)
    while os.path.exists(outputfilename):
      attempt += 1
      outputfilename = get_outputfilename(attempt)
  
    if not os.path.isdir(os.path.dirname(outputfilename)):
      os.makedirs(os.path.dirname(outputfilename))
  
  print "kernel_savedir : ", kernel_savedir
  print "outputfilename : ", outputfilename
  
  utility_functions.go_experiment(kernel_savedir = kernel_savedir, data_geometry = data_geometry, matrix_mnk = matrix_mnk, kernel_span = kernel_span, n_runs = 4, outputfilename = outputfilename, do_test = do_test, double_type = double_type, constantwaste = constantwaste, forcefilewrite = forcefilewrite)

def do_baidu_kernel_tests():
  """
  Run through the baidu problem set parameters, and do a kernel search
  """

  baidu_problem_set = baidu_bench.get_baidu_problem_set(aslist = True) 

  benchmark_n = 0
  basedir_kernels = os.path.join(write_directories.kernels_base_directory, "baidu_benching_%d"%(benchmark_n))
  basedir_output = os.path.join(write_directories.output_base_directory, "baidu_benching_%d"%(benchmark_n))
  
  while os.path.isdir(basedir_kernels) or os.path.isdir(basedir_output):
    benchmark_n += 1
    basedir_kernels = basedir_kernels.replace("baidu_benching_%d"%(benchmark_n - 1), "baidu_benching_%d"%(benchmark_n))
    basedir_output = basedir_output.replace("baidu_benching_%d"%(benchmark_n - 1), "baidu_benching_%d"%(benchmark_n))
    
  os.mkdir(basedir_kernels)
  os.mkdir(basedir_output)
    
  potential_micro_tiles = [[2,2], [8,8]]
  potential_tiles = [[8*a,8*b, a, b] for a,b in potential_micro_tiles]

  for bp in baidu_problem_set:
    
    tA = bp['tA']
    tB = bp['tB']
    m = bp['M']
    n = bp['N']
    k = bp['K']
    
    data_geometry = {'tA':tA, 'tB': tB, 'tC': False, 'isColMajor':True}
    matrix_mnk = {'m':m, 'n':n, 'k':k}
    
    
    big_problem = True if m*n > 10**6 else False
    small_problem = not big_problem
    
    def tile_is_reasonable(T):
      n_threads_1_app = (m*n + 0.)/(T[2]*T[3]  + 0.)
      not_too_tall = T[0] <= m  
      not_too_wide = T[1] <= n
      not_too_small = not (big_problem and T[2]*T[2] < 20)
      not_too_big = not (small_problem and T[2]*T[2] > 20)
      return not_too_tall and not_too_wide and not_too_small and not_too_big
    
    mm_tiles = [T for T in potential_tiles if tile_is_reasonable(T)]
    
    n_work_items_per_c_elms = None 
    if small_problem:
      n_work_items_per_c_elms = [1,3,6]
    else:
      n_work_items_per_c_elms = [1]
    
    kernel_span = {'Y_X_y_x' :mm_tiles, 'unrolls':[8,16], 'pads' : [1], 'group_allocations' : [1], 'work_item_load_a_pll_to_unrolls' : [0],'work_item_load_b_pll_to_unrolls' : [1],'unroll_pragmas' : [1], 'load_to_lds_interwovens' : [0], 'c_micro_tiles_interwovens': [1], 'use_edge_tricks':[1], 'n_work_items_per_c_elms':n_work_items_per_c_elms, "unroll_for_offsets":[0]}
      
    base_string = "tA%d_tB%d_tC%d_CM%d___m%d_n%d_k%d"%(data_geometry['tA'], data_geometry['tB'], data_geometry['tC'], data_geometry['isColMajor'], matrix_mnk['m'],matrix_mnk['n'],matrix_mnk['k'])

    kernel_savedir = os.path.join(basedir_kernels, base_string)

    outputfilename = os.path.join(basedir_output, "%s.txt"%(base_string))
    utility_functions.go_experiment(kernel_savedir = kernel_savedir, data_geometry = data_geometry, matrix_mnk = matrix_mnk, kernel_span = kernel_span, n_runs = 6, outputfilename = outputfilename, do_test = False)
    



def do_mn_tests(micro_tile_hw = 8, start_dim = 256 - 8, end_dim = 256 + 8, stride = 4):
  """
  for generating data to illustrate the pointer offset trick  
  """

  macro_tile_hw = 16*micro_tile_hw 
    
  attempt = 0
  savedir = os.path.join(write_directories.output_base_directory, 'multidimexp_micro%d_dir%d'%(micro_tile_hw, attempt,))
  while os.path.isdir(savedir):
    attempt += 1
    savedir = os.path.join(write_directories.output_base_directory, 'multidimexp_micro%d_dir%d'%(micro_tile_hw, attempt,))
  
  os.mkdir(savedir)
  

  #make the kernel and save it.
  data_geometry = {'tA':False, 'tB':False, 'tC': False, 'isColMajor':True}

  if micro_tile_hw == 2:
    kernel_span = {'Y_X_y_x' :[[macro_tile_hw, macro_tile_hw, micro_tile_hw, micro_tile_hw]], 'unrolls': [16], 'pads' : [1], 'group_allocations' : [1], 'work_item_load_a_pll_to_unrolls' : [0],'work_item_load_b_pll_to_unrolls' : [1],'unroll_pragmas' : [1], 'load_to_lds_interwovens' : [0], 'c_micro_tiles_interwovens' : [1], 'use_edge_tricks':[1], 'n_work_items_per_c_elms':[6]}
  
  elif micro_tile_hw == 4:
    kernel_span = {'Y_X_y_x' :[[macro_tile_hw, macro_tile_hw, micro_tile_hw, micro_tile_hw]], 'unrolls': [16], 'pads' : [1], 'group_allocations' : [1], 'work_item_load_a_pll_to_unrolls' : [0],'work_item_load_b_pll_to_unrolls' : [1],'unroll_pragmas' : [1], 'load_to_lds_interwovens' : [0], 'c_micro_tiles_interwovens' : [1], 'use_edge_tricks':[1], 'n_work_items_per_c_elms':[4]}
   
  else:
    kernel_span = {'Y_X_y_x' :[[macro_tile_hw, macro_tile_hw, micro_tile_hw, micro_tile_hw]], 'unrolls': [8], 'pads' : [1], 'group_allocations' : [1], 'work_item_load_a_pll_to_unrolls' : [0],'work_item_load_b_pll_to_unrolls' : [1],'unroll_pragmas' : [1], 'load_to_lds_interwovens' : [0], 'c_micro_tiles_interwovens' : [1], 'use_edge_tricks':[1], 'n_work_items_per_c_elms':[1]}

  for dimension in range(start_dim, end_dim, stride):
    kernel_span['use_edge_tricks'] = [0]
    matrix_mnk = { 'm': dimension, 'n': dimension, 'k': 2000}
    outputfilename = os.path.join(savedir, "dimension%s.txt"%(dimension))
    utility_functions.go_experiment(kernel_savedir = None, data_geometry = data_geometry, matrix_mnk = matrix_mnk, kernel_span = kernel_span, n_runs = 8, outputfilename = outputfilename, do_test = False)
    
    if dimension % macro_tile_hw == 0:
      kernel_span['use_edge_tricks'] = [1]
      outputfilename = os.path.join(savedir, "dimension%s_noredirect.txt"%(dimension))
      utility_functions.go_experiment(kernel_savedir = None, data_geometry = data_geometry, matrix_mnk = matrix_mnk, kernel_span = kernel_span, n_runs = 10, outputfilename = outputfilename, do_test = False)
