#Description of MIOpenGEMM examples

(OUT OF DATE) 

#basicexample.cpp

The starting point of a minimal example running the full find-then-run pipeline. The best place start to understand how to use MIOpenGEMM. 


#benchthecache.cpp

Benchmark all cached Solutions (currently only for cached DeepBench geometries)   



#deepbench.cpp

Find Solutions for DeepBench GEMM problems.


#devtest.cpp

Experimental tests, and easy way to printing kernels to file without any benchmarking


#devtest.cpp

A set of benchmark experiments


#gencache.cpp  

Illustrating how to generate Solutions and cache them for later use


#initialisationdemo.cpp  

Not really part of MIOPenGEMM, a simple experiment to see if initialising matrices with ints or floats makes a difference


#redirectionexample.cpp

Illustrating how problems are redirected to a problem  with is column major, and NN or NT (m < n) or TN (m < n). currently (1/12/2016) it is used only for cpu kernels.


