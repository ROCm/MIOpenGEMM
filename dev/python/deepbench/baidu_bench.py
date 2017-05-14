import os
import sys

sys.path.append("../experiments")
import write_directories
reload(write_directories)

baidu_txt_fn = os.path.join(write_directories.baidu_base_directory, "baidu_cudnn_results", "cublas_results_from_baidu.txt")
titanx_ods_fn = os.path.join("./data", "DeepBench_NV_TitanX.ods")
titanx_pascal_ods_fn = os.path.join("./data", "DeepBench_NV_TitanX_Pascal.ods")





def make_baidu_txt():
  """
  extract the times from the baidu spreadsheets, and write to a txt.
  """
  def line_looks_good(l):
    """ Hand-made function to check that a line in odt file (list in python) is what we want """ 
    if len(l) < 10:
      return False
  
    #using empty column 0 as a scratchpad, ignore here.
    for i in [1,7]: 
      if l[i]:
        return False
     
    for i in [2,3,4]:
      if not isinstance(l[i], int):
        return False
    
    for i in [5,6]:
      if l[i] not in [u'N', u'T']:
        return False
    
    for i in [8,9]:
      if not isinstance(l[i], (int, float)):
        return False
    
    
    return True  
  
  
  from pyexcel_ods import get_data
  print "Loading TitanX Maxwell..."
  baidu_data_maxwell = get_data(titanx_ods_fn)
  
  print "Loading TitanX Pascal..."
  baidu_data_pascal = get_data(titanx_pascal_ods_fn)
  
  both_results_dicts = {}
  for baidu_data, model in zip([baidu_data_maxwell, baidu_data_pascal], ["maxwell", "pascal"]):
    results_dicts = []
    keys = baidu_data['Results'][0]
    keys = {}
    good_columns = [2,3,4,5,6,8,9]
    for i in good_columns:
      keys[i] = baidu_data['Results'][0][i]
    
    print "\n", model, "---"
    for li, l in enumerate(baidu_data['Results'][1:90]):
      
      if line_looks_good(l):
        results_dicts.append({})
        for i in good_columns:
          results_dicts[-1][keys[i]] = l[i]
  
    
    results_dicts_2 = {}
    for d in results_dicts:
      results_dicts_2["M%s_N%s_K%s_tA%s_tB%s"%(d["M"], d["N"], d["K"], d["A Transpose"], d["B Transpose"])] = {'t_ms' :d['Time (msec)'], 'tflops': d["TERAFLOPS"]}
    
    both_results_dicts[model] = results_dicts_2
  
  
    
  filly = open(baidu_txt_fn, "w")
  


  problem_keys = ['M', 'N', 'K', 'tA', 'tB']
  for problem_key in problem_keys:
    filly.write("%s\t"%(problem_key,))
    
  filly.write("t_pascal_ms\ttflops_pascal\tt_maxwell_ms\ttflops_maxwell\n")
  
    
  pascal_keys = both_results_dicts["pascal"].keys()
  pascal_keys.sort(lambda x, y : 1 - 2*([int(t[1::]) for t in x.split("_")[0:3]] < [int(t[1::]) for t in y.split("_")[0:3]]))
  
  for k in pascal_keys:
    r_pascal = both_results_dicts['pascal'][k]
    
    for problem_key in problem_keys:
      filly.write(k.split(problem_key)[1].split("_")[0])
      filly.write("\t")
  
    
    if k in both_results_dicts["maxwell"].keys():
      r_maxwell = both_results_dicts['maxwell'][k]
      filly.write("%.9f\t%.9f\t%.9f\t%.9f\n"%(r_pascal['t_ms'], r_pascal["tflops"], r_maxwell['t_ms'], r_maxwell["tflops"]))
    
    else:
      filly.write("%.9f\t%.9f\t-       \t-       \n"%(r_pascal['t_ms'], r_pascal["tflops"]))
     
  for k in both_results_dicts["maxwell"].keys():
    if k not in both_results_dicts["pascal"].keys():
      for problem_key in problem_keys:
        filly.write(k.split(problem_key)[1].split("_")[0])
        filly.write("\t")
      r_maxwell = both_results_dicts['maxwell'][k]
      filly.write("-       \t-       \t%.9f\t%.9f\n"%(r_maxwell['t_ms'], r_maxwell["tflops"]))
  
    
  filly.close()



def get_baidu_problem_set(aslist = True):
  """
  aslist : if True, [{'m':m, ... }, ...] if False : {'m1000_n2000_...':, ...}
  """
  if not os.path.exists(baidu_txt_fn):
    raise RuntimeError("The file " + baidu_txt_fn + " still needs to be generated, consider running the function make_baidu_txt")
  
  filly = open(baidu_txt_fn)
  lines = filly.readlines()
  filly.close()
  keys = lines[0].split()
  baidu_problem_set = []
  for l in lines[1::]:
    entries = zip(keys, l.split())
    baidu_problem = {}
    for k,v in entries:
      if v == '-':
        baidu_problem[k] = v
      elif k in ['M', 'N', 'K']:
        baidu_problem[k] = int(v)
      elif k in ['tA', 'tB']:
        baidu_problem[k] = False if v == 'N' else True
      else:
        baidu_problem[k] = float(v)
  
    baidu_problem_set.append(baidu_problem)

  if aslist == True:
    return baidu_problem_set
  
  else:
    baidu_problem_set_dict = {}
    for e in baidu_problem_set:
      baidu_problem_set_dict['m%d_n%d_k%d_tA%d_tB%d'%(e['M'], e['N'], e['K'], e['tA'], e['tB'])] = e

    return baidu_problem_set_dict



def post_process_baidu():
  """
  We take the txt file which made the rounds at AMD, and extract which kernel won when. 
  """
  
  import numpy as np
  
  filly = open("/home/james/Downloads/cublas_results_from_baidu_incr.txt", "r")
  alllines = filly.readlines()
  titleline = alllines[0] 
  lines = alllines[1::]
  filly.close()
  split_lines = [l.split() for l in lines]
  kernel_dict = {}
  for x in split_lines:
    if x[-1] not in kernel_dict.keys():
      kernel_dict[x[-1]] = []
    kernel_dict[x[-1]].append(x[0:5] + [x[-2]])
  
  micro_to_c_area = {} #'2':[], '4':[], '8':[]}
  for k in kernel_dict.keys():
    size = int(k.split("_y")[1].split("_")[0])
    ksplit = int(k.split("ICE")[1])
    
    for l in kernel_dict[k]:
      m = int(l[0])
      n = int(l[1])
      #k = int(l[2])
      if size not in micro_to_c_area.keys():
        micro_to_c_area[size] = {}
      if ksplit not in micro_to_c_area[size].keys():
        micro_to_c_area[size][ksplit] = []
        
      micro_to_c_area[size][ksplit].append(np.log2(m*n))
  
  import matplotlib.pyplot as pl
  
  pl.clf()
  x_multi = []
  labels = []
  
  
  fn = 0
  for size, ksplit in [[2,6], [2,3], [2,1], [4,6], [4,3], [4,1],  [8,1]]:
    fn += 1
    x_multi.append(micro_to_c_area[size][ksplit])
    labels.append("mt:%s ks:%s"%(size, ksplit))

     
#pl.hist(x_multi, histtype='bar', label = labels, bins = 12)
    pl.subplot(7,1, fn)
    pl.hist(micro_to_c_area[size][ksplit], alpha = 0.8, label = "mt:%s ks:%s"%(size, ksplit), cumulative = True, range = [14, 30], bins = 52)
    
    pl.legend() 
    if fn is not 7:
      pl.xticks([])

  pl.xlabel("log_2 m*n")

  pl.savefig("/home/james/amdongit/reports/baidu_micro_ksplit_best_hists.pdf")
  
      #loc = 'upper left')
    #kernel_dict[x[-1]].append(x[0:5] + [x[-2]] + ['%.2f'%(np.log2(int(x[0])*int(x[1]) + 0.))])
  
  #for k in kernel_dict.keys():
    #print k 
    #for l in kernel_dict[k]:
      #print l
  
  #print titleline    
  #return kernel_dict
       
  


def make_old_txt():
  X = get_data("./data/gemm_old_results.ods")
  X = X['Sheet3'] 
  
  write_fn = os.path.join(write_directories.baidu_base_directory, "old_results", "previous.txt")
  filly = open(write_fn, 'w')
  for x in X[0]:
    filly.write("%s\t"%(x,))
  filly.write("\n")
  
  for x in X[1:-1]:
    if len(x) == 7:
      for c in x:
        filly.write("%s\t"%(c,))
      filly.write("\n")
        
  
