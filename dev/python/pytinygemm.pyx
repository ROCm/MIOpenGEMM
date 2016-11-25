import numpy as np
import multiprocessing as mpr
from libcpp.string cimport string
from libcpp.vector cimport vector 
from libcpp cimport bool
cimport cython
cimport cython.floating


cdef extern from "devtinygemm.hpp" namespace "tinygemm::dev":	
  void hello() except +

  void benchgemm[TFloat](bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, TFloat alpha, const TFloat * a, unsigned lda, unsigned a_offset, const TFloat * b, unsigned ldb, unsigned b_offset, TFloat beta, TFloat * c, unsigned ldc, unsigned c_offset, vector[string] cpu_algs, vector[vector[string]] gpu_kernel_filenames, bool capture_output, string & output, const TFloat * c_true_for_test, unsigned do_test, unsigned n_runs, string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic) except +;



def basehello():
  hello()
  return "goodbye ! :():"


def dangerwrap(f):
  """
  I assume f is a function which returns 
  an object and takes no parameters
  """
  event  = mpr.Event()
  q = mpr.Queue()
  
  def signalling_f():
    try:
      q.put(f())
    
    except Exception as e:
      print "Caught exception in dangerwrap:"
      print e
      q.put(e)
      event.set()
      return None
      
    event.set()
  
  
  
  f_process = mpr.Process(target = signalling_f)
  f_process.start()
  try:
    event.wait()
  
  except KeyboardInterrupt:
    f_process.terminate()
    f_process.join()
    raise KeyboardInterrupt("Caught KeyboardInterrupt in dangerwrap")
  
  return q.get()


def basetinygemm(datatype, isColMajor, tA, tB, tC, m, n, k, alpha, cython.floating [:] a, lda, a_offset, cython.floating [:] b, ldb, b_offset, beta, ldc, c_offset, cpu_algs_list, gpu_kernel_filenames_list_list, capture_output, cython.floating [:] c_pre_mem, cython.floating [:] c_pos_up, do_test, n_runs, outputfilename, findfirst, allotted_time, enforce_deterministic):

  cdef string astring  
  cdef vector[string] filenames_vec
  cdef vector[vector[string]] filenames_vec_vec

  cdef string captured_output
  
  for gpu_kernel_filenames_list in gpu_kernel_filenames_list_list:
    filenames_vec = []
    for fn in gpu_kernel_filenames_list:
      filenames_vec.push_back(fn)
    filenames_vec_vec.push_back(filenames_vec)
  
  cdef vector[string] cpu_algs_vec
  for alg in cpu_algs_list:
    cpu_algs_vec.push_back(alg)  
  
      
  cdef void (*cw_gemm) (bool, bool, bool, bool, unsigned, unsigned, unsigned, cython.floating, const cython.floating *, unsigned, unsigned,  const cython.floating *, unsigned, unsigned,  cython.floating, cython.floating *, unsigned, unsigned, vector[string], vector[vector[string]], bool capture_output, string & output, const cython.floating *, unsigned, unsigned, string, bool, float, bool) except+

  X = [1]
  if cython.floating is double:
    cw_gemm=&benchgemm[double]

  else:
    cw_gemm=&benchgemm[float]
    
  #TODO : remove the lda, ldb, ldc excess !!!!!!! (DONE)
  cw_gemm(isColMajor, tA, tB, tC, m,n,k, alpha, &a[0],lda, a_offset, &b[0], ldb, b_offset, beta, &c_pre_mem[0], ldc, c_offset, cpu_algs_vec, filenames_vec_vec, capture_output, captured_output, &c_pos_up[0], do_test, n_runs, outputfilename, findfirst, allotted_time, enforce_deterministic)
  
  X = np.array(c_pos_up)#.reshape(m, ldc)

  return {'c' : X.copy(), 'output': captured_output}



def pyhello():
  """
  Example function
  """	
  return dangerwrap(lambda : basehello)
  
  

def pytinygemm(isColMajor, tA, tB, tC, m, n, k, alpha, a, lda, a_offset, b, ldb, b_offset, beta, ldc, c_offset, cpu_algs_list, gpu_kernel_filenames_list_list, capture_output, c_pos_up, c_pre_mem, do_test, n_runs, outputfilename, findfirst, allotted_time, enforce_deterministic):
  """
  isColMajor:
    currently only 0 is supported
  tA
  
  tB
  
  m
  
  n
  
  k
  
  alpha
  
  a
  
  lda
  
  b
  
  ldb
  
  beta
  
  ldc
  
  cpu_algs_list
  
  gpu_kernel_filenames_list_list
  
  capture_output
    if True, then the output stream is not printed to terminal, but returned as 'output' string in returned dict
    
  c_true_for_test
    pass in the true c, used for testing that the kernel has performed the correct multiplication
    note : c_true must be in row major and have ldc = n, i.e. no excess rows. 
    
  do_test
    use the c_true_for_test passed to do reliability test
    
  n_runs
    number of runs with each algorithm/kernel
    
  outputfilename
    if a valid non-empty string, the output stream will be written (in addition) to this file.
    
  """
#  return basetinygemm("not important TODO remove", isColMajor, tA, tB, m, n, k, alpha, a.ravel(), lda, b.ravel(), ldb, beta, ldc, cpu_algs_list, gpu_kernel_filenames_list_list, capture_output)
  
  if not outputfilename:
    outputfilename = ""
  
  return dangerwrap(lambda : basetinygemm("not important TODO remove", isColMajor, tA, tB, tC, m, n, k, alpha, a.ravel(), lda, a_offset, b.ravel(), ldb, b_offset, beta, ldc, c_offset, cpu_algs_list, gpu_kernel_filenames_list_list, capture_output, c_pre_mem = c_pre_mem.ravel(), c_pos_up = c_pos_up.ravel(), do_test = do_test, n_runs = n_runs, outputfilename = outputfilename, findfirst = findfirst, allotted_time = allotted_time, enforce_deterministic = enforce_deterministic))
  
