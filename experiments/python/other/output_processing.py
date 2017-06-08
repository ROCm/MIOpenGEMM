import commands
import sys
import os
import re
#Could add most recent build number here?
sys.path.append("../build/cleaninstall/lib")
import pytinygemm

from IPython.core.debugger import Tracer
import matplotlib.pyplot as pl
pl.ion()
pytinygemm.pyhello()()

import time
import numpy as np
import numpy.random as npr
import sys

import baidu_bench
reload(baidu_bench)

import write_directories
reload(write_directories)



def get_output(filename):
  filly = open(filename,'r')
  output = filly.read()
  filly.close()
  return output
  
def get_processed_output(output):
  """
  get a dict, keys are filenames and values are lists of times.
  """
  lines = output.split("\n")
  times = []
  pou = {}
  key = None
  for l in lines:
    if "Running with" in l:
      if times:
        pou[key] = times
    
      key = l.split("/")[-1].split(".cl")[0]
      times = []
      
    if "elapsed time" in l:
      times.append(float(l.split()[3]))
  
  if key:
    pou[key] = times
    
  return pou



def get_shattered_list(pou, frags, exclusion_frags):
  """
  returns a list of dicts
  A dict looks like this {'GA1' : {'fn' : settings_with_GA1, 'min': bla, 'vals': bla }, 'GA2': {}, 'GA3': {} }
  """
  shattered_list = []  
  
  for fn in pou.keys():
    if sum([x in fn for x in exclusion_frags]) == 0:
      if frags[0] in fn:
        shattered_list.append({})
        for frag in frags:
          fn_f = fn.replace(frags[0], frag)
          shattered_list[-1][frag] = {
          'fn':fn_f,  
          'min':min(pou[fn_f]), 
          'vals':pou[fn_f], 
          }

  return shattered_list



def get_variable_dict_from_fn(fn):
  allparms_string = fn.split("___")[1].split(".")[0]
  allparm_frags = allparms_string.split("_")  
  variable_dict = {} 
  for frag in allparm_frags:
    variable_dict[re.sub('[0-9]+', '', frag)] = int(re.sub('[a-zA-Z]+', '', frag))
    
  return variable_dict
  


def get_frag_dict(pou, exclusion_frags):

  frag_dict = {}
  frag_dict['Y_X_y_x'] = {'title' : "Macro and micro \n tile sizes"}
  frag_dict['U'] = {'title' : "unroll"}
  frag_dict['P'] = {'title' : "Padding in LDS. A : \n (H + pad) * unroll, B : etc."} 
  frag_dict['GA'] = {'title': "How work groups \n are assigned to tiles"}
  frag_dict['APLU_BPLU'] = {'title' : "How data is read \n from global memory to LDS"}
  #frag_dict['BPLU'] = {'title' : "How data is read from global b to LDS"} 
  frag_dict['PU'] = {'title': "Pragma unrolling in loops"}
  frag_dict['LIW'] = {'title': "Load ala cobalt \n (inter-woven loading)"}
  frag_dict['ICE'] = {'title': "Level of k-direction splitting"}
  frag_dict['UFO'] = {'title': "An intial unroll of size determined by row & col of C"}
  
  variable_dicts = []
  for k in pou.keys():
    if sum([f in k for f in exclusion_frags]) == 0:
      variable_dicts.append(get_variable_dict_from_fn(k))
  
  for k in frag_dict.keys():
    target_variables = k.split("_")
    combos = [[vd[tv] for tv in target_variables] for vd in variable_dicts]
    unique_combos = []
    for combo in combos:
      if combo not in unique_combos:
        unique_combos.append(combo)

    unique_frags = ["_".join(["%s%s"%(tv, x) for tv, x in zip(target_variables, uc)]) for uc in unique_combos]
    frag_dict[k]['frags'] = unique_frags
  
  return frag_dict


def get_calls(output):
  lines = output.split("\n")
  input_call = lines[1]
  redirect_call = lines[2]
  return {'input':input_call, 'redirected':redirect_call}




def get_baidu_incremented(basedir = os.path.join(write_directories.output_base_directory, "baidu_dimensions")):
  """
  TODO : sort out paths
  """
  def get_keys(frag):
    m = int(frag.split("m")[1].split("_")[0])
    n = int(frag.split("n")[1].split("_")[0])
    k = int(frag.split("k")[1].split("_")[0])
    tA = int(frag.split("tA")[1].split("_")[0])
    tB = int(frag.split("tB")[1].split("_")[0])
    return m,n,k,tA,tB
    
  baidu_problem_set = baidu_bench.get_baidu_problem_set(aslist = False)
  

  for dirn in os.listdir(basedir):

    m,n,k,tA,tB = get_keys(dirn)  
    pou = get_processed_output(get_output(os.path.join(basedir, dirn, "results.txt")))
    
    best_kernel = None
    best_time = 10**44
    for pouk in pou.keys():
      median_time = np.median(pou[pouk])
      if median_time < best_time:
        best_time = median_time
        best_kernel = pouk
    
    key = "m%s_n%s_k%s_tA%s_tB%s"%(m,n,k,tA,tB)
    baidu_problem_set[key]["jn_kernel"] = best_kernel
    baidu_problem_set[key]['t_jn_ms'] = best_time
    baidu_problem_set[key]['tflops_jn'] = (2*m*n*k + 0.) / (best_time * 10e8)
  
  baidu_keys = baidu_problem_set.keys()
  baidu_keys.sort(lambda x, y : 1 - 2*([int(t[1::]) for t in x.split("_")[0:3]] < [int(t[1::]) for t in y.split("_")[0:3]]))
  
  def gs(x):
    if isinstance(x, float):
      return '%3.3f'%(x,)
    else:
      return x
  
  def gsp(x, padding = 7):
    return baidu_bench.padded(gs(x), padding)
  
  
  filly = open(os.path.join(write_directories.baidu_base_directory, "cublas_baidu_results", "combined_results", "combined_results.txt", "w"))
     
  title_string = "M \tN \tK \ttA \ttB \tt_pascal[ms] \tt_maxwell[ms] \tt_jn[ms] \ttflops_pascal \ttflops_maxwell \ttflops_jn \tkernel_used"
  filly.write(title_string + "\n")
  for bk in baidu_keys:
    ps = baidu_problem_set[bk]
    m,n,k,tA,tB = get_keys(bk)
    string = "%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s \t%s"%(m, n, k, tA, tB, gsp(ps['t_pascal_ms']), gsp(ps['t_maxwell_ms']), gsp(ps['t_jn_ms']), gsp(ps['tflops_pascal']), gsp(ps['tflops_maxwell']), gsp(ps['tflops_jn']), gsp(ps['jn_kernel']))
    filly.write(string + "\n")
