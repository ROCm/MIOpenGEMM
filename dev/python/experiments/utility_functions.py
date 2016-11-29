from IPython.core.debugger import Tracer 

import os
import numpy as np
import numpy.random as npr
import time
import sys
sys.path.append("../../../build/dev/python")
import pytinygemm


import write_directories
reload(write_directories)

import make_multiple_kernels
reload(make_multiple_kernels)

def get_possible_unrolls(H, W, h, w):

  if ((H*W) % (h*w) != 0):
    raise RuntimeError("micro tile does not fit macro tile : (%d %d) (%d %d)"%(H, W, h, w))
  
  H_good_a = H/(h*w + 0.)*np.arange(1,11)
  H_good_b = 1./H_good_a
  H_good = np.concatenate([H_good_a, H_good_b])
  H_good = H_good[H_good%1 == 0]   
  W_good_a = W/(h*w + 0.)*np.arange(1,11)
  W_good_b = 1./W_good_a
  W_good = np.concatenate([W_good_a, W_good_b])
  W_good = W_good[W_good%1 == 0]
  con_good = np.concatenate([H_good, W_good])  
  bla = np.histogram(con_good, bins = 0.5 + np.arange(50))
  unrolls = 1 + np.array(np.where(bla[0] > 1))[0]

  return unrolls


def get_reduced_geometry(data_geometry, data_dimensions):
  """
  This is a nice but deprecated function
  """
  reduced_data_geometry = data_geometry
  
  if data_geometry['isColMajor'] == False:
    X = {
    'tA':data_geometry['tB'], 
    'tB':data_geometry['tA'], 
    'tC':data_geometry['tC'], 
    'isColMajor':True
    }
    reduced_data_geometry = get_reduced_geometry(X, {'n': data_dimensions['m'], 'm':data_dimensions['n']})
    
  
  elif (data_geometry['tA'] == True and data_geometry['tB'] == True):
    reduced_data_geometry =  {
    'tA':False, 
    'tB':False,
    'tC': not data_geometry['tC'], 
    'isColMajor':data_geometry['isColMajor']
    }


  
  elif (data_dimensions['m']  > data_dimensions['n'] and ((data_geometry['tA'] == True  and data_geometry['tB']== False) or (data_geometry['tA'] == False  and data_geometry['tB'] == True))):
    reduced_data_geometry = {
    'tA':data_geometry['tA'], 
    'tB':data_geometry['tB'],
    'tC':not data_geometry['tC'], 
    'isColMajor':data_geometry['isColMajor']
    }

  return reduced_data_geometry
    
 
 
 
def get_minimal_stride(h, w, is_transpose, is_colmajor):
  """
  an h by w matrix : return the non-1 stride if no padding around matrix.
  """
  if (is_transpose + is_colmajor) % 2 == 0:
    return w
  else:
    return h
  
def get_ldxs(matrix_mnk, factorwaste, constantwaste, data_geometry):
  ldxs = {}
  m, n, k = matrix_mnk['m'], matrix_mnk['n'], matrix_mnk['k'] 
  ldxs['ldc'] = factorwaste['c']*get_minimal_stride(m, n, data_geometry['tC'], data_geometry['isColMajor']) + constantwaste['c']
  ldxs['lda'] = factorwaste['a']*get_minimal_stride(m, k, data_geometry['tA'], data_geometry['isColMajor']) + constantwaste['a']
  ldxs['ldb'] = factorwaste['b']*get_minimal_stride(k, n, data_geometry['tB'], data_geometry['isColMajor']) + constantwaste['b']
  return ldxs

def get_data_dimensions(matrix_mnk, factorwaste, constantwaste, data_geometry):
  data_dimensions = matrix_mnk.copy()
  ldxs = get_ldxs(matrix_mnk, factorwaste, constantwaste, data_geometry)
  for k in ldxs.keys():
    data_dimensions[k] = ldxs[k]
  
  return data_dimensions

def set_stroud_memory(x_true, ldx, is_transpose, is_colmajor, offset, npftype):
  """
  stroud == stride 
  """
  height, width = x_true.shape
  x_mem = None
  if (is_transpose + is_colmajor) % 2 == 0:
    x_mem = 10.*np.array(npr.randint(-10, 10, size = (height*ldx + offset, )), dtype = npftype)
    x_mem_postoff = x_mem[offset::].reshape(height, ldx)
    x_mem_postoff[0:height, 0:width] = x_true
  else:
    x_mem = 10.*np.array(npr.randint(-10, 10, size = (width*ldx + offset, )), dtype = npftype)
    x_mem_postoff = x_mem[offset::].reshape(width, ldx)
    x_mem_postoff[0:width, 0:height] = x_true.T
  return x_mem




def go_experiment(kernel_savedir = None, kernel_span = {'Y_X_y_x' :[[64, 64, 4, 4]], 'unrolls': [16], 'pads' : [1], 'group_allocations' : [1], 'work_item_load_a_pll_to_unrolls' : [0],'work_item_load_b_pll_to_unrolls' : [0],'unroll_pragmas' : [0,1], 'load_to_lds_interwovens' : [0], 'use_edge_tricks' :[1], 'n_work_items_per_c_elms': [1], 'unroll_for_offsets' : [0]}, 
matrix_mnk = {'m': 640, 'n':2560, 'k':11213}, data_geometry = {'tA':False, 'tB':False, 'tC': False, 'isColMajor':True}, double_type = np.float32, outputfilename = "",factorwaste = {'a':1, 'b':1, 'c':1}, constantwaste = {'a':5, 'b':7, 'c':13}, offsets = {'a': 0, 'b': 0, 'c': 0}, n_runs = 7, do_test = True, findfirst = False, allotted_time = -1., enforce_deterministic = False, forcefilewrite = False):
  """
  kernel_savedir
    if findfirst == False : 
      if None, then it will will be `(somepath)/experiment_temporary_directory', and will be removed when finished
      otherwise, the directory where all the .cl files will be saved
  
  kernel_span
    the set of parameters to explore 
  
  matrix_mnk 
    m,n,k
  
  data_geometry
    tA, tB, tV, isColMajor
  
  double_type
    np.float32 or np.float64
  
  outputfilename
    if "", no output will be written
    otherwise, file where all the output should be written
    
   factorwaste, constantwaste:
    lda, ldb, ldc will be F*min_possible + C  where min_possible is what they would be if the matrices were compact, F is factor C is constant
      
  n_runs:
    number of runs to perform with each kernel
    
  forcefilewrite:
    if False, then refuse to create a kernel if the path already exists
  """
  
  if findfirst == True and kernel_span:
    raise RuntimeError("This error is being thrown from go_experiment. findfirst is False, and kernel_span is non empty (it is True). This is a contradictory request. With findfirst, a heuristic search is performed using an in-built algorithm. If kernel_span is passed in, the kernels to generate, compile, and benchmark are specified up-front by the user. Please change one or the other")
    
  if allotted_time >= 0 and kernel_span:
    raise RuntimeError("allotted_time >= 0 with non-empty kernel_span. Note that allotted_time is only relevent for findfirst == True")
    
  
  data_dimensions = get_data_dimensions(matrix_mnk, factorwaste, constantwaste, data_geometry)

  ######## the benchmarking cartesian product case #########################################################
  if findfirst == False:
    cleanup = None
    temporary_directory_name = os.path.join(write_directories.kernels_base_directory, "experiment_temporary_directory")
    
    if not kernel_savedir:
      cleanup = True #True (if you want the directory to be removed afterwards)
      kernel_savedir = temporary_directory_name
  
      if not os.path.isdir(kernel_savedir):
        os.mkdir(kernel_savedir)
      
      for fname in os.listdir(kernel_savedir)[-1::-1]:
        if fname[-3::] == ".cl":
          os.remove(os.path.join(kernel_savedir, fname))
        else:
          raise RuntimeError("This is strange. The file %s should not have been encountered in directory %s while cleaning up "%(fname, kernel_savedir))

    else:
      cleanup = False
      
      if not os.path.isdir(kernel_savedir):
        os.mkdir(kernel_savedir)
      
      if os.listdir(kernel_savedir) and not forcefilewrite:
        contents_string = "\n".join(os.listdir(kernel_savedir))
        raise RuntimeError("The proferred directory for saving kernel (.cl) is non-empty. The directory kernel_savedir (`%s') must be empty or non-existant. It currenty contains : %s "%(kernel_savedir, contents_string))
  ############################################################################################################


  print "\n\n(py)setting memories. ",
  a_up = 0.1*np.array(npr.randint(-10, 11, size = (data_dimensions['m'], data_dimensions['k'])), dtype = double_type)
  b_up = 0.1*np.array(npr.randint(-10, 11, size = (data_dimensions['k'], data_dimensions['n'])), dtype = double_type)
  c_pre_up = 0.1*np.array(npr.randint(-10, 11, size = (data_dimensions['m'], data_dimensions['n'])), dtype = double_type)
  
  a_mem = set_stroud_memory(a_up, data_dimensions['lda'], data_geometry['tA'], data_geometry['isColMajor'], offsets['a'], double_type)
  b_mem = set_stroud_memory(b_up, data_dimensions['ldb'], data_geometry['tB'], data_geometry['isColMajor'], offsets['b'], double_type)
  c_pre_mem = set_stroud_memory(c_pre_up, data_dimensions['ldc'], data_geometry['tC'], data_geometry['isColMajor'], offsets['c'], double_type)
  
  REDUCE_THE_GEOMETRY_IS_DEPRECATED = True
  reduced_data_geometry = None
  if (REDUCE_THE_GEOMETRY_IS_DEPRECATED == False):
    reduced_data_geometry = get_reduced_geometry(data_geometry, data_dimensions)
  
  else:
    reduced_data_geometry = data_geometry
  
  
  ######## the benchmarking cartesian product case #########################################################
  if findfirst == False:
    print "(py)making kernel sources. ", 
    make_multiple_kernels.make_multiple_kernels(kernel_savedir, reduced_data_geometry, kernel_span, double_type)
  ###########################################################################################################

  just_make = False
  if just_make == True:
    return 

  alpha = double_type(1.0)
  beta = double_type(1.0) #double_type(0.1)
  cpu_algs_list = [] #["3fors"]

  if do_test:
    print "(py)computing true c. ",
    time00 = time.time()
    c_pos_up = beta*c_pre_up + alpha*np.array(np.dot(np.array(a_up, np.float64), np.array(b_up, np.float64)), dtype = double_type)
    time01 = time.time()
    print "(py)computed in", 1000*(time01 - time00), "[ms]"

  else:
    c_pos_up = np.array([3.14], dtype = double_type)


  ######## the benchmarking cartesian product case #########################################################
  if findfirst == False:
    gpu_kernel_filenames_list_list = [[os.path.join(kernel_savedir, x)] for x in os.listdir(kernel_savedir)]
  else:
    gpu_kernel_filenames_list_list = []
  ###########################################################################################################  
    

  if double_type == np.float64:
    print """
          *********************************************************************************************************
          **** Running (in go_experiment) with double_type float64, what are you some kind of crazy scientist? **** 
          *********************************************************************************************************
          """
          
  
  pytinygemm.pytinygemm(data_geometry['isColMajor'], data_geometry['tA'], data_geometry['tB'], data_geometry['tC'], data_dimensions['m'], data_dimensions['n'], data_dimensions['k'], alpha, a_mem, data_dimensions['lda'], offsets['a'], b_mem, data_dimensions['ldb'], offsets['b'], beta, data_dimensions['ldc'], offsets['c'], cpu_algs_list, gpu_kernel_filenames_list_list, capture_output = False, c_pre_mem = c_pre_mem.ravel(), c_pos_up = c_pos_up.ravel(), do_test = do_test, n_runs = n_runs, outputfilename = outputfilename, findfirst = findfirst, allotted_time = allotted_time, enforce_deterministic = enforce_deterministic)
  

  ######## the benchmarking cartesian product case #########################################################
  if findfirst == False:
    if cleanup and kernel_savedir == temporary_directory_name:
      for fname in os.listdir(kernel_savedir)[-1::-1]:
        if fname[-3::] == ".cl":
          os.remove(os.path.join(kernel_savedir, fname))
        else:
          raise RuntimeError("This is strange. The file %s should not have been encountered in directory %s while cleaning up "%(fname, kernel_savedir))
      
      if len(os.listdir(kernel_savedir)) == 0:
        os.rmdir(kernel_savedir)
      else:
        print os.listdir(kernel_savedir)
        raise RuntimeError("This is strange. kernel_savedir `%s' should be empty now"%(kernel_savedir,))
  ###########################################################################################################  
  
