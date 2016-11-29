import numpy as np
import sys
import os
import matplotlib.pyplot as pl
from IPython.core.debugger import Tracer

import deeptensile
reload(deeptensile)

import sys
sys.path.append("../../../tinygemm/python")
import make_kernel



def get_baidu_problem_frags():
  """
  direct from baidu's DeepBench gemm for cuda code. 
  """
  
  baidu_string = """
  std::make_tuple(1760, 16, 1760, false, false),
  std::make_tuple(1760, 32, 1760, false, false),
  std::make_tuple(1760, 64, 1760, false, false),
  std::make_tuple(1760, 128, 1760, false, false),
  std::make_tuple(1760, 7000, 1760, false, false),
  std::make_tuple(2048, 16, 2048, false, false),
  std::make_tuple(2048, 32, 2048, false, false),
  std::make_tuple(2048, 64, 2048, false, false),
  std::make_tuple(2048, 128, 2048, false, false),
  std::make_tuple(2048, 7000, 2048, false, false),
  std::make_tuple(2560, 16, 2560, false, false),
  std::make_tuple(2560, 32, 2560, false, false),
  std::make_tuple(2560, 64, 2560, false, false),
  std::make_tuple(2560, 128, 2560, false, false),
  std::make_tuple(2560, 7000, 2560, false, false),
  std::make_tuple(4096, 16, 4096, false, false),
  std::make_tuple(4096, 32, 4096, false, false),
  std::make_tuple(4096, 64, 4096, false, false),
  std::make_tuple(4096, 128, 4096, false, false),
  std::make_tuple(4096, 7000, 4096, false, false),
  std::make_tuple(1760, 16, 1760, true, false),
  std::make_tuple(1760, 32, 1760, true, false),
  std::make_tuple(1760, 64, 1760, true, false),
  std::make_tuple(1760, 128, 1760, true, false),
  std::make_tuple(1760, 7000, 1760, true, false),
  std::make_tuple(2048, 16, 2048, true, false),
  std::make_tuple(2048, 32, 2048, true, false),
  std::make_tuple(2048, 64, 2048, true, false),
  std::make_tuple(2048, 128, 2048, true, false),
  std::make_tuple(2048, 7000, 2048, true, false),
  std::make_tuple(2560, 16, 2560, true, false),
  std::make_tuple(2560, 32, 2560, true, false),
  std::make_tuple(2560, 64, 2560, true, false),
  std::make_tuple(2560, 128, 2560, true, false),
  std::make_tuple(2560, 7000, 2560, true, false),
  std::make_tuple(4096, 16, 4096, true, false),
  std::make_tuple(4096, 32, 4096, true, false),
  std::make_tuple(4096, 64, 4096, true, false),
  std::make_tuple(4096, 128, 4096, true, false),
  std::make_tuple(4096, 7000, 4096, true, false),
  std::make_tuple(1760, 7133, 1760, false, true),
  std::make_tuple(2048, 7133, 2048, false, true),
  std::make_tuple(2560, 7133, 2560, false, true),
  std::make_tuple(4096, 7133, 4096, false, true),
  std::make_tuple(5124, 9124, 1760, false, false),
  std::make_tuple(35, 8457, 1760, false, false),
  std::make_tuple(5124, 9124, 2048, false, false),
  std::make_tuple(35, 8457, 2048, false, false),
  std::make_tuple(5124, 9124, 2560, false, false),
  std::make_tuple(35, 8457, 2560, false, false),
  std::make_tuple(5124, 9124, 4096, false, false),
  std::make_tuple(35, 8457, 4096, false, false),
  std::make_tuple(5124, 9124, 1760, true, false),
  std::make_tuple(35, 8457, 1760, true, false),
  std::make_tuple(5124, 9124, 2048, true, false),
  std::make_tuple(35, 8457, 2048, true, false),
  std::make_tuple(5124, 9124, 2560, true, false),
  std::make_tuple(35, 8457, 2560, true, false),
  std::make_tuple(5124, 9124, 4096, true, false),
  std::make_tuple(35, 8457, 4096, true, false),
  std::make_tuple(7680, 16, 2560, false, false),
  std::make_tuple(7680, 32, 2560, false, false),
  std::make_tuple(7680, 64, 2560, false, false),
  std::make_tuple(7680, 128, 2560, false, false),
  std::make_tuple(7680, 16, 2560, true, false),
  std::make_tuple(7680, 32, 2560, true, false),
  std::make_tuple(7680, 64, 2560, true, false),
  std::make_tuple(7680, 128, 2560, true, false),
  std::make_tuple(3072, 16, 1024, false, false),
  std::make_tuple(3072, 32, 1024, false, false),
  std::make_tuple(3072, 64, 1024, false, false),
  std::make_tuple(3072, 128, 1024, false, false),
  std::make_tuple(3072, 16, 1024, true, false),
  std::make_tuple(3072, 32, 1024, true, false),
  std::make_tuple(3072, 64, 1024, true, false),
  std::make_tuple(3072, 128, 1024, true, false),
  std::make_tuple(3072, 7435, 1024, false, true),
  std::make_tuple(7680, 5481, 2560, false, true)"""
  
  
  problems = []
  for l in  baidu_string.split("\n"):
    if l:
      frags = l.split("tuple(")[1].split(")")[0].split(",")
      m = int(frags[0])
      n = int(frags[1])
      k = int(frags[2])
      
      tA = 0 + 1*(frags[3].strip() == "true")
      tB = 0 + 1*(frags[4].strip() == "true")
      if tA == 1 and tB == 1:
        m,k = k,m
    
      problems.append("m%s_n%s_k%s_tA%s_tB%s"%(m,n,k,tA,tB))

  return problems







def get_baidu_results():
  """
  """  
  #Previous results, and those from baidu.
  previousresults_fn = "/home/james/tinygemmout/baidu/combined_results/cublas_results_from_baidu_incr.txt"
  filly = open(previousresults_fn)
  oldlines = filly.readlines()
  pas_tfs = {}
  max_tfs = {}
  for l in oldlines[1::]:
    #Get old results and problem dimensions. 
    m, n, k, tA, tB, t_pas, t_max, t_old, tf_pas, tf_max, tf_old, kern_old = l.split()
    tA, tB = int(tA),int(tB)
    ##To fix Baidu's DeepBench funny-business, we need to switch m and k for TN.
    if (tA == 1 and tB == 0):
      m,k = k,m

    target_frag = "m%s_n%s_k%s_tA%s_tB%s"%(m,n,k,tA,tB)

    if (tf_max != '-'):
      max_tfs[target_frag] = float(tf_max)

    if (tf_pas != '-'):
      pas_tfs[target_frag] = float(tf_pas)

  return {'pas':pas_tfs, 'max':max_tfs}


def get_tensile_results(nruns = 4):
  return deeptensile.get_tensile_floppage(nruns)
  


def get_tinygemm_results():
  """
  Benchmarked on 18 Nov 2016, starting from 1 of three hyper parameter 
  settings, with allotted_time at 30 seconds.
  """
  deepbenchresults_dir = "/home/james/tinygemmout/deepbench"
  files = os.listdir(deepbenchresults_dir)
  filefound = False
  #offset, enforce deterministic
  tinygemm_results = {'00': {}, '10': {}, '01':{} } 
  #for key in tinygemm_results.keys():
    #tinygemm_results[key] = {'gflop/s':{}, 'hyperparams':{}, 'timesfound':{}}
  
  for newfile in files:    
    geomfrag =  newfile.split("ed")[1][2::].split(".txt")[0]
    offset = int(newfile.split("off")[1].split("_")[0])
    ed = int(newfile.split("ed")[1].split("_")[0])
    tinykey = '%d%d'%(offset, ed)
    filly = open(os.path.join(deepbenchresults_dir, newfile))
    allnewlines = filly.readlines()
    for l in allnewlines:
      if "INPUT_CALL" in l:
        if geomfrag not in tinygemm_results[tinykey].keys():
          tinygemm_results[tinykey][geomfrag] = {}
           
        tinygemm_results[tinykey][geomfrag]["INPUT_CALL"] = l
        break
        
    kern_new, time_found, gf_new = allnewlines[-2].split()
    tf_new = float(gf_new)/1000.
    tinygemm_results[tinykey][geomfrag]["gflop/s"] = tf_new
    tinygemm_results[tinykey][geomfrag]["hyperparams"] = kern_new
    tinygemm_results[tinykey][geomfrag]["timesfound"] = time_found

  return tinygemm_results


def get_kernel_cache_string():

  add_best_string = ""
  
  tgrs = get_tinygemm_results()
  for okey in ["00", "10"]:
    for k in tgrs[okey].keys():
      geom = {}
      for kv in tgrs[okey][k]["INPUT_CALL"].split(": ")[1].strip().split(" "):
        key, val = kv.split(":")
        geom[key] = int(val)
      
      for k2 in ['tA', 'tB', 'tC','colMaj']:
        geom[k2] = "true"*(geom[k2] == 1) + "false"*(geom[k2] == 0)
        
      add_best_string  += '  std::make_tuple<gemmgeometry::Geometry, std::string> ( {%s, %s, %s, %s, %d, %d, %d, %d, %d, %d} ,   "%s" ), \n'%(geom['colMaj'], geom['tA'], geom['tB'], geom['tC'], geom['lda'], geom['ldb'], geom['ldc'], geom['m'], geom['n'], geom['k'], tgrs[okey][k]["hyperparams"])

  kernel_cache_string = r"""
std::vector<std::tuple<gemmgeometry::Geometry, std::string> > 
HyperParams::kernel_cache = {
%s
};
"""%(add_best_string)
  
  print kernel_cache_string

def plot_full_summary():


  baidu_results = get_baidu_results()

  frags = [k for k in get_baidu_problem_frags() if k in baidu_results["max"].keys() and k in baidu_results["pas"].keys()]


  nvidia_tfs = {}
  for key in ["pas", "max"]:
    nvidia_tfs[key] = np.array([baidu_results[key][frag] for frag in frags])



  tgresults = get_tinygemm_results()
  
  tg_tfs = {}
  for key in ["10", "01", "00"]:
    tg_tfs[key] = np.array([tgresults[key][frag]["gflop/s"] for frag in frags])
  
 
  tenresults = get_tensile_results()
  tensile_tfs = np.array([tenresults[frag] for frag in frags])

    
    
  print "geomean ( tflops00 / tflopspas ) ", 2**((1./nvidia_tfs["max"].size)*np.log2(tg_tfs["00"]/nvidia_tfs["pas"]).sum())
  print "geomean ( tflops00 / tflopsmax ) ", 2**((1./nvidia_tfs["max"].size)*np.log2(tg_tfs["00"]/nvidia_tfs["max"]).sum())
  
  
  kwargs = {'linestyle':':', 'markersize':5, 'marker':'x'}
  pl.clf()
  

  pl.figure(1, figsize = (10,10))
  pl.clf()
  pl.subplot(7,1,1)
  pl.plot(tg_tfs["00"]/tg_tfs["01"], label = "gained with splitting in k (non-det / deterministic)", **kwargs)
  pl.ylim(ymin = 0, ymax = 2.0)
  pl.plot([0,pl.xlim()[-1]], [1,1])
  pl.legend(loc = 'lower right')
  pl.ylabel ("tf / tf")
  
  pl.subplot(7,1,2)
  pl.plot(tg_tfs["10"]/tg_tfs["00"], label = "gained by ld{a,b} padding {5,7} (if only we could...)", **kwargs)
  pl.ylim(ymin = 0, ymax = 2.0)
  pl.plot([0,pl.xlim()[-1]], [1,1])
  pl.legend(loc = 'lower right')
  pl.ylabel ("tf / tf")
  
  pl.subplot(7,1,5)
  pl.plot(tg_tfs["00"]/nvidia_tfs["max"], label = "current speed up over maxwell (mean(gf/gf) = 1.57) :)", **kwargs)
  pl.ylim(ymin = 0, ymax = 5.0)
  pl.plot([0,pl.xlim()[-1]], [1,1])
  pl.legend(loc = 'upper right')
  pl.ylabel ("tf / tf")
  
  pl.subplot(7,1,6)
  pl.plot(tg_tfs["00"]/nvidia_tfs["pas"], label = "... and over pascal (mean(gf/gf) = 1.02)", **kwargs)
  pl.ylim(ymin = 0, ymax = 5.0)
  pl.plot([0,pl.xlim()[-1]], [1,1])
  pl.legend(loc = 'upper left', frameon = True, framealpha = 0.5)
  pl.ylabel ("tf / tf")
  
  pl.subplot(7,1,7)
  pl.plot(tg_tfs["00"], label = 'tinygemm', **kwargs)  
  pl.legend(loc = 'upper left', frameon = True, framealpha = 0.5)
  pl.ylabel ("current tflop/s")
  pl.xlabel("dimensions id, ordered by problem size m.n.k")

def plot_tensile_tinygemm(dosave = True, withdeterministic = False):
  
  
  pl.figure(2, figsize = (21, 8))
  
  pl.clf()
  
  baidu_problem_frags = get_baidu_problem_frags()
  tgresults = get_tinygemm_results()
  tg_tfs = {}
  for key in ["10", "01", "00"]:
    
    print len(tgresults[key].keys())
    print len(baidu_problem_frags)
    tg_tfs[key] = np.array([tgresults[key][frag]["gflop/s"] for frag in baidu_problem_frags])
  
 
  tenresults = get_tensile_results()
  tensile_tfs = np.array([tenresults[frag] for frag in baidu_problem_frags])
    
  
  
  index = np.arange(tensile_tfs.size)
  if withdeterministic == False:
    bar_width = 0.35
  
  opacity = 0.8
  error_config = {'ecolor': '0.3'}
  
  if withdeterministic == True:
    bar_width = 0.24
    rects0 = pl.bar(index, tg_tfs["01"], bar_width, alpha=opacity, color='g', label='tinygemm, no-k-split')
  
  rects2 = pl.bar(index + 1*bar_width, tensile_tfs/1000., bar_width, alpha=opacity, color='r', label='Tensile')
  rects1 = pl.bar(index + 2*bar_width, tg_tfs["00"], bar_width, alpha=opacity, color='b', label='tinygemm')

  pl.ylabel('tflop/s')
  pl.legend()
  pl.xlim([0,index.max() + 1])
  pl.plot([0,index.max() + 1], [8.2, 8.2], color = 'orange', linewidth = 5)
  pl.ylim([0,14.5])
  pl.xlabel("deepbench problem")
  pl.subplots_adjust(left = 0.1, bottom = 0.42, right = 0.95)
  pl.text(1, 8.4, "FIJI TARGET", fontsize = 14)

  
  pl.xticks(range(len(baidu_problem_frags)), baidu_problem_frags, rotation = 80)

  
  if dosave:
    if withdeterministic == False:
      fn = "./data/tensile_tinygemm_fiji.pdf"
    else:
      fn = "./data/tensile_tinygemm_fiji_with_deterministic.pdf"
    pl.savefig(fn)
    import commands
    commands.getstatusoutput("pdfcrop %s %s"%(fn, fn))
  


def make_time_table():


  baidu_results = get_baidu_results()

  frags = [k for k in get_baidu_problem_frags() if k in baidu_results["max"].keys() and k in baidu_results["pas"].keys()]


  nvidia_tfs = {}
  for key in ["pas", "max"]:
    nvidia_tfs[key] = np.array([baidu_results[key][frag] for frag in frags])



  tgresults = get_tinygemm_results()
  
  tg_tfs = {}
  for key in ["10", "01", "00"]:
    tg_tfs[key] = np.array([tgresults[key][frag]["gflop/s"] for frag in frags])
  
 
  tenresults = get_tensile_results()
  tensile_tfs = np.array([tenresults[frag] for frag in frags])

  #Tracer()()
  
  #Tracer()()

  print "problem \t\t\t\tpas \t max \t tinygemm \t tgdet \t tensile" 
  for frag in frags:
    
    flops = 1e-9*float( int(frag.split("m")[1].split("_")[0]) * int(frag.split("n")[1].split("_")[0]) * int(frag.split("k")[1].split("_")[0]))
    print frag, "\t  %.5f %.5f \t%.5f \t%.5f \t%.5f"%(flops/baidu_results["pas"][frag], flops/baidu_results["max"][frag], flops/tgresults["00"][frag]["gflop/s"], flops/tgresults["01"][frag]["gflop/s"],  flops/(tenresults[frag]/1000.)) #, nvidia_tfs["max"][frag], tensile_tfs[frag]



    #print frag, "\t  ", baidu_results["pas"][frag], baidu_results["max"][frag], tgresults["00"][frag]["gflop/s"], tgresults["01"][frag]["gflop/s"],  (tenresults[frag]/1000.) #, nvidia_tfs["max"][frag], tensile_tfs[frag]

   

def make_best_kernels():
  tinygemm_results = get_tinygemm_results()
  
  for key in tinygemm_results["00"].keys():
    k = int(key.split("k")[1].split("_")[0])
    m = int(key.split("m")[1].split("_")[0])
    n = int(key.split("n")[1].split("_")[0])    
    tA = int(key.split("tA")[1].split("_")[0])    
    tB = int(key.split("tB")[1].split("_")[0])    
  
    for detstring in ["00", "01"]:
      hp = tinygemm_results[detstring][key]['hyperparams']
      Y = int(hp.split("Y")[1].split("_")[0])
      X = int(hp.split("_X")[1].split("_")[0])
      y = int(hp.split("_y")[1].split("_")[0])
      x = int(hp.split("_x")[1].split("_")[0])
      U = int(hp.split("_U")[1].split("_")[0])
      P = int(hp.split("_P")[1].split("_")[0])
      GA = int(hp.split("_GA")[1].split("_")[0])
      APLU = int(hp.split("_APLU")[1].split("_")[0])
      BPLU = int(hp.split("_BPLU")[1].split("_")[0])
      PU = int(hp.split("_PU")[1].split("_")[0]) 
      LIW = int(hp.split("_LIW")[1].split("_")[0]) 
      MIW = int(hp.split("_MIW")[1].split("_")[0]) 
      ET = int(hp.split("_ET")[1].split("_")[0]) 
      ICE = int(hp.split("_ICE")[1].split("_")[0]) 
      UFO = int(hp.split("_UFO")[1].split("_")[0]) 
      
      print key
      print tinygemm_results[detstring][key]['hyperparams']
      print k,m,n,tA,tB, hp, Y, X, y, x, U, P, GA, APLU, BPLU, PU, LIW, MIW, ET, UFO
      make_kernel.make_kernel(dir_name = "./kernels_found_deepbench/geom_%s/"%(key,), kernelname = None, t_float = "float", 
  #tA1 tB0 tC0 CM1 
  a_transposed = tA, b_transposed = tB, c_transposed = 0, is_col_major = 1, 
  #x
  micro_tile_width = x, 
  #y
  micro_tile_height = y, 
  #X
  macro_tile_width = X, 
  #Y
  macro_tile_height = Y, 
  #U
  unroll = U, 
  #P
  pad = P, 
  #GA
  group_allocation = GA, 
  #APLU
  work_item_load_a_pll_to_unroll = APLU, 
  #BPLU
  work_item_load_b_pll_to_unroll = BPLU, 
  #PU
  unroll_pragma = PU, 
  #LIW
  load_to_lds_interwoven = LIW, 
  #MIW
  c_micro_tiles_interwoven = MIW, 
  #ET
  use_edge_trick = ET,  
  #ICE
  n_work_items_per_c_elm = ICE,
  #UFO
  unroll_for_offset = UFO,
   # haven't optimised over this
  n_target_active_workgroups = 64, 
  
  with_tailing_end_char = False)
  
