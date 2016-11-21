import numpy as np
import sys
import os
import matplotlib.pyplot as pl

from IPython.core.debugger import Tracer


import deeptense
reload(deeptense)

tensile_floppage_1run = deeptense.get_tensile_floppage(1)
tensile_floppage_4run = deeptense.get_tensile_floppage(4)



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
    print l
    frags = l.split("tuple(")[1].split(")")[0].split(",")
    print frags
    m = int(frags[0])
    n = int(frags[1])
    k = int(frags[2])
    tA = 0 + 1*(frags[3] == "true")
    tB = 0 + 1*(frags[4] == "true")
    if tA == 1 and tB == 1:
      m,k = k,m
  
  problems.append("m%s_n%s_k%s_tA%s_tB%s"%(m,n,k,tA,tB))
  

def tensile_tinygemm():

  pl.clf()
  mnks = []
  newlines = []
  times_found_10 = []
  new_tfs_10 = []
  new_tfs_01 = []
  new_tfs_00 = []
  tensile_tfs_1run = []
  tensile_tfs_4run = []
  old_kerns = []
  new_kerns = []
  files_added = []
  frags = []
  made_switches = []
  old_tfs = []
  front_one_completes = []


  for target_frag in problems:
    filefound = False
    for newfile in files:
      if target_frag in newfile:
        filly = open(os.path.join(deepbenchresults_dir, newfile))
        allnewlines = filly.readlines()
        if True : #(tf_max != '-' and tf_pas != '-'):
          #get the new results : best kernel, the time it was found, and its floppage.        
          kern_new, time_found, gf_new = allnewlines[-2].split()
          tf_new = float(gf_new)/1000.
          
          #no offset, and enforces determinism (no k-splitting)
          if "at30_off0_ed1_" in newfile:
            new_tfs_01.append(tf_new)
          
          #no offset, and allows non-determinism (k-splitting)  
          elif "at30_off0_ed0_" in newfile:      
            new_tfs_00.append(tf_new)
          
          #with ldx offset, allows non-determinism.  
          elif "at30_off1_ed0_" in newfile:      
            new_tfs_10.append(tf_new)
            times_found_10.append(float(time_found))
            
            #we also append the previous results here (could do at any of the 3 cases)
            frags.append(target_frag)
            mnks.append(float(m)*float(n)*float(k))
            tensile_tfs_1run.append(tensile_floppage_1run[target_frag])
            tensile_tfs_4run.append(tensile_floppage_4run[target_frag])
  
          else:
            print "what is this?"
            #print target_frag, float(tf_new)/float(tf_old)
  
  
  
  new_tfs_10 = np.array(new_tfs_10)
  new_tfs_01 = np.array(new_tfs_01)
  new_tfs_00 = np.array(new_tfs_00)
  tensile_tfs_1run = np.array(tensile_tfs_1run)
  tensile_tfs_4run = np.array(tensile_tfs_4run)
  frags = np.array(frags)
  mnks = np.array(mnks)
  
  #perfrats = new_tfs_00/max_tfs
  #perfrats = mnks
  indices = np.arange(new_tfs_00.size)
  
  index = np.arange(new_tfs_00.size)
  bar_width = 0.34
  opacity = 0.8
  error_config = {'ecolor': '0.3'}
  rects2 = pl.bar(index, tensile_tfs_4run[indices]/1000., bar_width, alpha=opacity, color='r', label='Tensile')
  rects1 = pl.bar(index + bar_width, new_tfs_00[indices], bar_width, alpha=opacity, color='g', label='tinygemm')
  pl.ylabel('tflop/s')
  pl.legend()
  pl.xlim([0,new_tfs_00.size])
  
  pl.savefig("tensile_tinygemm.pdf")
  



pl.clf()
tensile_tinygemm()














  

def full_plotter():
  #Newly benchmarked data (18 Nov 2016)
  deepbenchresults_dir = "/home/james/tinygemmout/deepbench"
  files = os.listdir(deepbenchresults_dir)
  
  #Previous results, and those from baidu.
  previousresults_fn = "/home/james/tinygemmout/baidu/combined_results/cublas_results_from_baidu_incr.txt"
  filly = open(previousresults_fn)
  oldlines = filly.readlines()
  
  #Getting ready to store results.
  mnks = []
  newlines = []
  times_found_10 = []
  new_tfs_10 = []
  new_tfs_01 = []
  new_tfs_00 = []
  pas_tfs = []
  max_tfs = []
  tensile_tfs_1run = []
  tensile_tfs_4run = []
  old_kerns = []
  new_kerns = []
  files_added = []
  frags = []
  made_switches = []
  old_tfs = []
  front_one_completes = []
  
  for l in oldlines[1::]:
    #Get old results and problem dimensions. 
    m, n, k, tA, tB, t_pas, t_max, t_old, tf_pas, tf_max, tf_old, kern_old = l.split()
    
    tA = int(tA)
    tB = int(tB)
   
    ##To fix Baidu's DeepBench funny-business, we need to switch m and k for TN.
    if (tA == 1 and tB == 0):
      m,k = k,m
  
    
  
    target_frag = "m%s_n%s_k%s_tA%s_tB%s"%(m,n,k,tA,tB)
    filefound = False
    for newfile in files:
      if target_frag in newfile:
        filly = open(os.path.join(deepbenchresults_dir, newfile))
        allnewlines = filly.readlines()
        if (tf_max != '-' and tf_pas != '-'):
          #get the new results : best kernel, the time it was found, and its floppage.        
          kern_new, time_found, gf_new = allnewlines[-2].split()
          tf_new = float(gf_new)/1000.
          
          #no offset, and enforces determinism (no k-splitting)
          if "at30_off0_ed1_" in newfile:
            new_tfs_01.append(tf_new)
          
          #no offset, and allows non-determinism (k-splitting)  
          elif "at30_off0_ed0_" in newfile:      
            new_tfs_00.append(tf_new)
          
          #with ldx offset, allows non-determinism.  
          elif "at30_off1_ed0_" in newfile:      
            new_tfs_10.append(tf_new)
            times_found_10.append(float(time_found))
            
            #we also append the previous results here (could do at any of the 3 cases)
            old_tfs.append(float(tf_old))
            frags.append(target_frag)
            pas_tfs.append(tf_pas)
            max_tfs.append(tf_max)
            mnks.append(float(m)*float(n)*float(k))
            old_kerns.append(kern_old)
            new_kerns.append(allnewlines[-1].split()[0])
            tensile_tfs_1run.append(tensile_floppage_1run[target_frag])
            tensile_tfs_4run.append(tensile_floppage_4run[target_frag])
            files_added.append(newfile)
            if sum(["SWITCHING" in l for l in allnewlines]) == 1:
              made_switches.append(1)
            else:
              made_switches.append(0)
  
          else:
            print "what is this?"
            #print target_frag, float(tf_new)/float(tf_old)
  
  
  
  new_tfs_10 = np.array(new_tfs_10)
  new_tfs_01 = np.array(new_tfs_01)
  new_tfs_00 = np.array(new_tfs_00)
  tensile_tfs_1run = np.array(tensile_tfs_1run)
  tensile_tfs_4run = np.array(tensile_tfs_4run)
  old_tfs = np.array(old_tfs)
  frags = np.array(frags)
  pas_tfs = np.array([float(x) for x in pas_tfs])
  max_tfs = np.array([float(x) if x!= '-' else 0 for x in max_tfs])
  mnks = np.array(mnks)
  
  print "geomean ( tflops00 / tflopspas ) ", 2**((1./pas_tfs.size)*np.log2(new_tfs_00/pas_tfs).sum())
  print "geomean ( tflops00 / tflopsmax ) ", 2**((1./pas_tfs.size)*np.log2(new_tfs_00/max_tfs).sum())
  
  
  perfrats = new_tfs_00/max_tfs
  perfrats = mnks
  indices = np.array(perfrats).argsort()
  
  
  kwargs = {'linestyle':':', 'markersize':5, 'marker':'x'}
  pl.clf()
  
  
  if False:
    pl.subplot(7,1,1)
    pl.plot(new_tfs_00[indices]/new_tfs_01[indices], label = "gained with splitting in k (non-det / deterministic)", **kwargs)
    pl.ylim(ymin = 0, ymax = 2.0)
    pl.plot([0,pl.xlim()[-1]], [1,1])
    pl.legend(loc = 'lower right')
    pl.ylabel ("tf / tf")
    
    pl.subplot(7,1,2)
    pl.plot(new_tfs_10[indices]/new_tfs_00[indices], label = "gained by ld{a,b} padding {5,7} (if only we could...)", **kwargs)
    pl.ylim(ymin = 0, ymax = 2.0)
    pl.plot([0,pl.xlim()[-1]], [1,1])
    pl.legend(loc = 'lower right')
    pl.ylabel ("tf / tf")
    
    
    pl.subplot(7,1,3)
    pl.plot(new_tfs_10[indices]/old_tfs[indices], label = "gained by using new stochastic (30 second hard limit)", **kwargs)
    pl.ylim(ymin = 0, ymax = 2.0)
    pl.plot([0,pl.xlim()[-1]], [1,1])
    pl.legend(loc = 'lower right')
    pl.ylabel ("tf / tf")
    
    pl.subplot(7,1,4)
    pl.plot(new_tfs_00[indices]/old_tfs[indices], label = "overall change since last numbers (+TN, -padding, +stochastic search, +UFO)", **kwargs)
    
    pl.ylim(ymin = 0, ymax = 2.0)
    pl.plot([0,pl.xlim()[-1]], [1,1])
    pl.legend(loc = 'lower right')
    pl.ylabel ("tf / tf")
    
    pl.subplot(7,1,5)
    pl.plot(new_tfs_00[indices]/max_tfs[indices], label = "current speed up over maxwell (mean(gf/gf) = 1.57) :)", **kwargs)
    pl.ylim(ymin = 0, ymax = 5.0)
    pl.plot([0,pl.xlim()[-1]], [1,1])
    pl.legend(loc = 'upper right')
    pl.ylabel ("tf / tf")
    
    pl.subplot(7,1,6)
    pl.plot(new_tfs_00[indices]/pas_tfs[indices], label = "... and over pascal (mean(gf/gf) = 1.02)", **kwargs)
    pl.ylim(ymin = 0, ymax = 5.0)
    pl.plot([0,pl.xlim()[-1]], [1,1])
    pl.legend(loc = 'upper left', frameon = True, framealpha = 0.5)
    pl.ylabel ("tf / tf")
    
    pl.subplot(7,1,7)
    pl.plot(new_tfs_00[indices], label = 'tinygemm', **kwargs)
  
  #pl.ylim(ymin = 0, ymax = 9)
  #pl.plot([0,pl.xlim()[-1]], [8.2,8.2], color = 'r', linewidth = 2)
  
    pl.legend(loc = 'upper left', frameon = True, framealpha = 0.5)
    pl.ylabel ("current tflop/s")
    pl.xlabel("dimensions id, ordered by problem size m.n.k")
  
  if False:  
    pl.figure(45, figsize = (15, 8))
    pl.subplot(2,1,1)
    pl.plot((mnks[indices]/(tensile_tfs_1run[indices]/1000.)), label = 'tensile (1 run)', **kwargs)
    pl.plot((mnks[indices]/(tensile_tfs_4run[indices]/1000.)), label = 'tensile (4 run)', **kwargs)
    pl.plot((mnks[indices]/new_tfs_00[indices]), label = 'tinygemm', **kwargs)
    pl.plot((mnks[indices]/new_tfs_01[indices]), label = 'tinygemm no k-split (deterministic)', **kwargs)
    pl.ylabel ("time [?s]")
    pl.yscale('log', basey = 2)
    pl.legend(loc = 'upper left', frameon = True, framealpha = 0.5)
    
    pl.subplot(2,1,2)
    pl.plot((tensile_tfs_1run[indices]/1000.), label = 'tensile (1 run)', **kwargs)
    pl.plot((tensile_tfs_4run[indices]/1000.), label = 'tensile (4 run)', **kwargs)
    pl.plot(new_tfs_00[indices], label = 'tinygemm', **kwargs)
    pl.plot(new_tfs_01[indices], label = 'tinygemm no k-split (deterministic)', **kwargs)
    pl.ylabel ("tflop/s")
    
    pl.legend(loc = 'upper left', frameon = True, framealpha = 0.5)
    pl.xlabel("dimensions id, ordered by problem size m.n.k")
    
    pl.savefig("tentative_tensile_1run_4run.pdf")
    
    pl.figure(55, figsize = (15, 8))
  #pl.plot((tensile_tfs_4run[indices]/1000.), label = 'tensile (4 run)', **kwargs)
  #pl.plot(new_tfs_00[indices], label = 'tinygemm', **kwargs)
  #pl.ylabel ("tflop/s")
  
  #fig, ax = plt.subplots()
  
  index = np.arange(new_tfs_00.size)
  bar_width = 0.35
  opacity = 0.8
  error_config = {'ecolor': '0.3'}
  rects1 = pl.bar(index, new_tfs_00[indices], bar_width, alpha=opacity, color='b', label='tinygemm')
  rects2 = pl.bar(index + bar_width, tensile_tfs_4run[indices]/1000., bar_width, alpha=opacity, color='r', label='Tensile')
  pl.ylabel('tflop/s')
  pl.legend()
  
  
  #pl.savefig("deepbenchres18nov2016.pdf")
  
