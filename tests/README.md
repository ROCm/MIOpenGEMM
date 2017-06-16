#OUT OF DATE !

# smallgeometrytests.cpp

Runs the full find-then-run pipeline for all 32 possible (a,b,c transposes, column major, m > n)  cases, but only for small matrices. 
    
#basicexample.cpp

The starting point of a minimal example running the full find-then-run pipeline. The best place start to understand how to use MIOpenGEMM. 

#basicfind.hpp

The basic code of all examples, in a header file as it templated over float type. It contains all the OpenCL boiler-plate, cpu and gpu matrix allocations, accuracy testing, etc used in all examples and tests. 

#deepbench.cpp

Benchmarking of DeepBench GEMM problems.

#backconvwrw.cpp

Benchmarking of DeepBench back convolution with respect to weights.

#redirectionexample.cpp

Illustrating how problems are redirected to a problem  with is column major, and NN or NT (m < n) or TN (m < n). currently (1/12/2016) it is used only for cpu kernels.

#devtest.cpp

Experimental tests, currently (1/12/2016) not for end-user use.
