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

import output_processing
reload(output_processing)


    
def get_xlabel_from_filename(fn, frag):
  """
  used for x-label in bar plots 
  """
  label = fn.split('___')[-1].split('.')[0]
  clean_frag = re.sub('[0-9]+', '*', frag)
  label = label.replace(frag, clean_frag)
  label = label.replace("_", " ")
  return label
  

def scatterplot1(shattered_list, title = "Unroll", frags = ['U8', 'U16'], base_colors = ['r', 'g']):
  
  if len(frags) < 2:
    raise RuntimeError("More that 2 frags are required")
    
  all_times = {}
  for k in frags:
    all_times[k] = []
  
    
  for shatteringi, shattering in enumerate(shattered_list):
    for fragi, frag in enumerate(frags):
      all_times[frag].extend(shattering[frag]['vals'])

  fastest_per_frag = {}
  for frag in frags:
    fastest_per_frag[frag] = min(all_times[frag])

  pairs = [(fastest_per_frag[k], k) for k in frags]
  pairs.sort()
  fastest_frag = pairs[0][1]
  other_frags = [f for f in frags if f != fastest_frag]
  frag1 = fastest_frag
  
  all_ALL_times = np.concatenate([all_times[f] for f in frags])
  min_ALL_time = all_ALL_times.min()
  max_ALL_time = all_ALL_times.max()

  
  for subploti, frag2 in enumerate(other_frags):
    times_xy = np.array([all_times[frag1], all_times[frag2]])
    min_time = times_xy.min()
    max_time = times_xy.max()
    upper_plot_lim = min_time + 1.02*(max_time - min_time)
    lower_plot_lim = 0.0
    tick_posis = [min_ALL_time, min_ALL_time + 0.95*(max_ALL_time - min_ALL_time)]
    nrows = np.ceil(np.sqrt(len(frags) - 1))
    ncols = np.ceil((len(frags) - 1) / nrows)
    pl.subplot(nrows, ncols, subploti + 1)
    
    pl.plot([0, times_xy.max()], [0, times_xy.max()], color = 'k', linewidth = 1, linestyle = '-')
    pl.plot([0, times_xy.max()], [0, 1.05*times_xy.max()], color = (0.5, 0.5, 0.5), linewidth = 1, linestyle = ':')
    pl.plot([0, times_xy.max()], [0, 1./1.05*times_xy.max()], color = (0.5, 0.5, 0.5), linewidth = 1, linestyle = ':')
    pl.plot(times_xy[0], times_xy[1], linestyle = "None", markersize = 1, marker = ".")#, alpha = 0.5)
    
    if len(frags) - subploti < ncols + 2 :
      pl.xlabel('%s [ms]'%(frag1,) )
      pl.xticks(tick_posis, ['%.3f'%(x,) for x in tick_posis])
    else:
      pl.xticks([])
    
    if subploti%ncols == 0:
      pl.ylabel('%s '%(frag2,) )
    else:
      pl.ylabel('%s '%(frag2,) )
      pl.yticks([])
    

    pl.xlim(lower_plot_lim, upper_plot_lim)
    pl.ylim(lower_plot_lim, upper_plot_lim)
      
  pl.suptitle(title)
  pl.subplots_adjust(bottom = 0.3, left = 0.3, top = 0.85, right = 0.85, hspace = 0.3, wspace = 0.3)
        
def barplot1(shattered_list, title = "Unroll", frags = ['U8', 'U16', 'U32'], base_colors = ['r', 'g','b']):
  """
  where
  
  * pou is the processed output from get_processed_output
  
  * title:
      Example : "Padding in LDS"

  * frags:
      Example : ["_P1", "_P2"]
      
  * base_colors:
      Example :  ['r', 'g']

  * exclusion frags : all kernels with one of these frags in their name will be ignored. 
  
  """
      
  pl.clf()


  
  global_min = 10**10
  
  colors = {}
  for fi, f in enumerate(frags):
    colors[f] = base_colors[fi]
  
  
  ax = pl.gca()
  ax.grid(True)
  x_plot_vals, x_labels = [], []
  min_series = {}
  for frag in frags:
    min_series[frag] = [shattering[frag]['min'] for shattering in shattered_list]
    
  width = 0.7/(len(frags))
  xlabels = [get_xlabel_from_filename(shattering[frags[0]]['fn'], frags[0]) for shattering in shattered_list]
  xlabel_positions = np.arange(len(shattered_list)) + width*len(frags)/2.
  pl.xticks(xlabel_positions, xlabels, rotation = 85)


  all_times = {} 
  for frag in frags:
    all_times[frag] = {'x_vals': [], 'y_vals': []}
  for shatteringi, shattering in enumerate(shattered_list):
    for fragi, frag in enumerate(frags):
      all_times[frag]['x_vals'].extend(len(shattering[frag]['vals'])*[shatteringi + width*(fragi + 0.5)])
      all_times[frag]['y_vals'].extend(shattering[frag]['vals'])


  rectangles = {}
  x_range = np.arange(len(shattered_list))
  for fragi, frag in enumerate(frags):
    pl.plot(all_times[frag]['x_vals'], all_times[frag]['y_vals'], color = colors[frag], marker = 'x', markersize = 5, linestyle = 'None') #, alpha = 0.5)
    rectangles[frag] = ax.bar(x_range + width*fragi, min_series[frag], width, color=colors[frag], edgecolor = 'None', label = frag.replace("_", ""))#, yerr=menStd)
     
  
  pl.xlim(xmin = -1.5)
  pl.ylabel('time [ms]')
  pl.subplots_adjust(bottom = 0.6)
  pl.legend(loc = 'upper left', framealpha = 0.4)

  pl.title(title)




def get_color_list(number):
  colors = ["#8dd3c7","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd","#ccebc5","#ffffb3"]
  return colors


def get_texstring_start(calls):
  texstring = r"""
\documentclass[french]{beamer}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb}
\usepackage{babel}
\usepackage{tikz}
\usetheme{PaloAlto}
\title{%s \newline %s}
\subtitle{Running on Catalyst}
\begin{document}
\begin{frame}
	\titlepage
\end{frame}
"""%(calls['input'].replace("_", " "), calls['redirected'].replace("_", " "))

  return texstring
      
        
def multiplotter(pou, exclusion_frags = [], make_scatterplots = True, make_barplots = False, savedir = None):
  """
  any kernel/fn with a frag from exclusion_frags will not be included in the plot.
  """
  
  frag_dict = output_processing.get_frag_dict(pou, exclusion_frags)
  figure_names = []
  
  ##Scatterplots:
  if make_scatterplots:
    for ik, k in enumerate(frag_dict.keys()):
      if len(frag_dict[k]['frags']) > 1: # == 2
        print " good "
        shattered_list = output_processing.get_shattered_list(pou, frag_dict[k]['frags'], exclusion_frags)
        pl.figure(ik, figsize = (8, 8))
        scatterplot1(shattered_list, frag_dict[k]['title'], frag_dict[k]['frags'], base_colors = get_color_list(len(frag_dict[k]['frags'])))
        if savedir:
          fname = os.path.join(savedir, "scatter_%s.pdf"%(k,))
          pl.savefig(fname)
          commands.getstatusoutput('pdfcrop %s %s'%(fname, fname))
          pl.close('all')
          figure_names.append(fname)

  
  
  # Barplots : 
  if make_barplots:
    for ik, k in enumerate(frag_dict.keys()):
      if len(frag_dict[k]['frags']) > 1:
        shattered_list = get_shattered_list(pou, frag_dict[k]['frags'], exclusion_frags)
        
        print "-->", k
        pl.figure(k, figsize = (11, 9))
        barplot1(shattered_list, frag_dict[k]['title'], frag_dict[k]['frags'], base_colors = get_color_list(len(frag_dict[k]['frags'])))
        if savedir:
          fname = os.path.join(savedir, "barplots_%s.pdf"%(k,))
          pl.savefig(fname)
          commands.getstatusoutput('pdfcrop %s %s'%(fname, fname))
          pl.close('all')
          figure_names.append(fname)

  return figure_names


def get_tex_figfrag(figure_name):
  return r"""
{
  \setbeamertemplate{navigation symbols}{}
  \begin{frame}[plain, noframenumbering]
  \begin{tikzpicture}[remember picture,overlay]
  \node[at=(current page.center)] {
  \includegraphics[height = 0.9\paperheight]{%s}};
  \end{tikzpicture}
  \end{frame}
}
    """%(figure_name,)


def plot_scatterplots(outputfilename = "some_output_file_probably_in_output_base_directory", savedir = "some_directory_probably_in_plot_base_directory"):   
  """
  make and save scatter plots and barplots. 
  
  if savedir is:
  None      : Don't save, and don't close the figures generated
  "default" : Will save to default location
  otherwise : save to specified path. 
  
  """
  output = output_processing.get_output(outputfilename)
  pou = output_processing.get_processed_output(output)
  calls = output_processing.get_calls(output)
  
  if savedir == "default":
    savedir = os.path.join(write_directories.plot_base_directory, "default")
    if not os.path.isdir(savedir):
      os.mkdir(savedir)
    
    for fn in [os.path.join(savedir, x) for x in os.listdir(savedir)[-1::-1]]:
      os.remove(fn)
    
  
  if savedir: 
    if not os.path.isdir(savedir):
      os.mkdir(savedir)
  
  
  fnames1 = multiplotter(pou, [], make_scatterplots = True, make_barplots = False, savedir = savedir)
  
  if savedir: 
    
    texfn = os.path.join(savedir, "allfigs.tex")
    texstring = get_texstring_start(calls)
    for figure_name in fnames1: 
      texstring += get_tex_figfrag(figure_name)
      
    texstring += r"""
    \end{document}"""
    
    filly = open(texfn, "w")
    filly.write(texstring)
    filly.close()
    
    for i in range(2):  
      print commands.getstatusoutput("pdflatex -output-directory %s %s"%(savedir, texfn,))


def make_mn_plot(results_dir = "some_output_file_probably_in_output_base_directory_maybe_pointer_shift", savefn = "some_filename_probably_in_plot_base_directory"):
  
  pl.clf()
  if results_dir == "default":
    results_dir = os.path.join(write_directories.output_base_directory, "multidimexp")
    if not os.path.isdir(results_dir):
      raise RuntimeError("The default place to find mn data, \n%s, \ndoes not seem to exists"%(results_dir,))
    
  if savefn == "default":
    atte = 0
    savefn = os.path.join(write_directories.plot_base_directory, "mn_plots", "default", "plot%d.pdf"%(atte,))

    while os.path.exists(savedir):
      atte += 1
      savefn = os.path.join(write_directories.plot_base_directory, "mn_plots", "default", "plot%d.pdf"%(atte,))    
    
    
    #for fn in os.listdir(savedir):
      #os.remove(os.path.join(savedir, fn))
  
  
  runtimes = {'redir' : {}, 'noredir' : {}}
  pl.clf()
  
  for fn in os.listdir(results_dir):
    
    if "noredirect" in fn:
      redir = 'noredir'
    else:
      redir = 'redir'
    
    fullpath = os.path.join(results_dir, fn)
    if ".txt" not in fn:
      raise RuntimeError("Not sure : just found the file `%s', am I in the right directory for make_mn_plot?"%(fullpath,))
    
    filly = open(fullpath, "r")
    lines = filly.readlines()

    macro_tile_Y, macro_tile_X = None, None
    for l in lines:
      if "m:" in l and "n:" in l and "REDIRECTED_CALL" in l:
        
        
        m = int(l.split("m:")[1].split(" ")[0])
        n = int(l.split("n:")[1].split(" ")[0])
        if m != n:
          raise RuntimeError("Hmmm, i thought m = n in this experiment")
        
        if m in runtimes.keys():          
          raise RuntimeError("That's odd, there appear to have been more than one run with dimension = %d in thus runtimes dict"%(m,))
         
        runtimes[redir][m] = []
        dimension = m
      
      elif "k:" in l:
        k = int(l.split("k:")[1].split(" ")[0])
        
      elif "elapsed time" in l:
        runtimes[redir][m].append(float(l.split("elapsed time :")[1].strip().split()[0].strip()))
        
      elif "Running with" in l:
        
        macro_tile_Y = int(l.split("___Y")[1].split("_")[0])
        macro_tile_X = int(l.split("_X")[1].split("_")[0])
        micro_tile_y = int(l.split("_y")[1].split("_")[0])
        micro_tile_x = int(l.split("_x")[1].split("_")[0])

        title = "Times and flop/s of a GEMM kernel with a %dx%d macro-tile and a %dx%d micro-tile "%(macro_tile_X, macro_tile_Y, micro_tile_x, micro_tile_y)
        
        #l.split("___")[1].split(".cl")[0] #Running with tempory_directory/A0B0C0f32___Y128_X128_y8_x8_U16_P1_GA3_APLU1_BPLU1_PU0_LIW0.cl
  
  if macro_tile_X == None:
    raise RuntimeError("This is estranged, macro_tile_X is None")
  
  if macro_tile_X != macro_tile_Y:
    raise RuntimeError("The macro tile used appears not to be square. This is unexpected, are you sure this is right? Veto this throw hhhhere if so")
  
  
        
  
  runtimes_vectors = {'redir':[], 'noredir':[]}
  dimensions_vectors = {'redir':[], 'noredir':[]}
  for redir in ['redir', 'noredir']:
    for d in runtimes[redir].keys():
      runtimes_vectors[redir].extend(runtimes[redir][d])
      dimensions_vectors[redir].extend(len(runtimes[redir][d])*[d])

  tilecutters = [x for x in dimensions_vectors['redir'] if x % macro_tile_X == 0]
  
  for redir in ['redir', 'noredir']:
    runtimes_vectors[redir] = np.array(runtimes_vectors[redir])
    dimensions_vectors[redir] = np.array(dimensions_vectors[redir])

  pl.suptitle(title)
  pl.subplot(2,1,1)
  

  mintime, maxtime = [oper([oper(runtimes_vectors[klopper]) for klopper in ['redir', 'noredir']]) for oper in [min, max]]
  
  #, runtimes_vectors['redir'].max()
  
  for x in tilecutters:
    pl.plot([x,x], [0.0*mintime, 2.*maxtime], color = (0., 0., 0.5), alpha = 0.1, linestyle = "-", linewidth = 0.1)
  
  base_kwargs = {'linestyle':'None', 'alpha':0.5}

  kwargs_redir = base_kwargs.copy()
  kwargs_redir['color'] = '#00BFFF'#get_color_list(4)[2]
  kwargs_redir['markersize'] = 3
  kwargs_redir['marker'] = '.'
  
  kwargs_noredir = base_kwargs.copy()
  kwargs_noredir['color'] = 'r' #get_color_list(4)[3]
  kwargs_noredir['markersize'] = 9
  kwargs_noredir['marker'] = 'x'
  
  for redir, kwargs in zip(['noredir', 'redir'], [kwargs_noredir, kwargs_redir]):
    pl.plot(dimensions_vectors[redir], runtimes_vectors[redir], **kwargs)


  pl.xlim(dimensions_vectors['redir'].min() - 0.5, dimensions_vectors['redir'].max() + 0.5)
  pl.ylabel('time [ms]')
  pl.ylim(ymin = 0)
  
  pl.ylim(0, 1.05*maxtime)



  
  pl.subplot(2,1,2)
  gflops = {}
  for redir in ['redir', 'noredir']: 
    gflops[redir] = ((2. * dimensions_vectors[redir]**2 * k)/runtimes_vectors[redir])/10**6
  
  mingflops, maxgflops = gflops['redir'].min(), gflops['redir'].max()
  for x in tilecutters:
    pl.plot([x,x], [0.0*mingflops, 1.05*maxgflops], color = (0., 0., 0.5), alpha = 0.1, linestyle = "-", linewidth = 0.1)
  
  for redir, kwargs in zip(['noredir', 'redir'], [kwargs_noredir, kwargs_redir]):
    pl.plot(dimensions_vectors[redir], gflops[redir], **kwargs)

  
  #Tracer()()
  pl.xlim(dimensions_vectors['redir'].min() - 0.5, dimensions_vectors['redir'].max() + 0.5)
    
  pl.xlabel("M ( = N ), [K = %d]"%(k,))
  pl.ylabel('GFlop/s')
  pl.ylim(0, 1.05*maxgflops)
  
  filly = open(fullpath, "r")
  lines = filly.readlines()
  


  pl.subplots_adjust(left = 0.15, hspace = 0.3, wspace = 0.3)
   
   
  savedirname = os.path.dirname(savefn) 
  if not os.path.isdir(savedirname):
    os.makedirs(savedirname)
  
  pl.savefig(savefn)
  commands.getstatusoutput("pdfcrop %s %s"%(savefn, savefn))

def make_mn_plots():
  loopy = 0
  #texfn = os.path.join(savedir, "allfigs.tex")
  
  #texstring = get_texstring_start("")
  #for figure_name in fnames1: 
        
  dirs = ["multidimexp_micro2_dir0", "multidimexp_micro4_dir0", "multidimexp_micro8_dir0"]        
  fns = ["micro2.pdf", "micro4.pdf", "micro8.pdf"]
  for results_dir, savefn in zip(dirs, fns):
    pl.figure(loopy)
    pl.clf()
    figname = os.path.join(write_directories.plot_base_directory, "multidimexp", savefn)
    make_mn_plot(os.path.join(write_directories.output_base_directory, results_dir), figname)
    #pl.close('all')
    #texstring += get_tex_figfrag(figure_name)
    loopy += 1


  

      
  #texstring += r"""
#\end{document}"""
    
    #filly = open(texfn, "w")
    #filly.write(texstring)
    #filly.close()
    
    #for i in range(2):  
      #print commands.getstatusoutput("pdflatex -output-directory %s %s"%(savedir, texfn,))
