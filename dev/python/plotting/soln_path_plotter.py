import numpy as np
import matplotlib.pyplot as pl

def is_hyper_param_string(frag):
  if "_ICE" in frag and "_MIW" in frag:
    return True
  else:
    return False
    
filly = open("/home/james/libraries/tinygemm/examples/findlog.txt", "r")
lines = filly.readlines()

hyper_params = []
median_gflops = []
elapsed_seconds = []

for l in lines:
  if is_hyper_param_string(l):
    hyper_params.append(l.strip())
  if "m-Gflops/s" in l:
    median_gflops.append(float(l.split()[-1]))
  if "elapsed seconds" in l:
    elapsed_seconds.append(float(l.split()[-1]))
  
  if "SWITCHING" in l:
    t_switch = elapsed_seconds[-1]

for x,y,z in zip(hyper_params, elapsed_seconds, median_gflops):
  print x,y,z



pl.close('all')
pl.figure(num = 1, figsize = (4,4)) 
pl.ion()
pl.plot(elapsed_seconds, median_gflops, linestyle = "none", marker = "x", markersize = 10 )

pl.plot([t_switch, t_switch], [0, max(median_gflops)], color = 'r')


pl.xlabel("elapsed time [s]")
pl.ylabel("gflop/s")

pl.xlim(xmin = 0)
pl.ylim(ymin = 0)

pl.subplots_adjust(bottom = 0.3, left = 0.2)
