# tinygemm

tinygemm is a tool for generating OpenCL matrix multiplication (GEMM) kernels. The project is inspired by very similar to Tensile (link), a more general purpose tool for tensor problems. 

A GEMM problem, op(c) = alpha * op(a) * op(b) + beta * op(c), is defined by the dimensions and memory layout of matrices a,b and c : isColMaj, tA, tB, tC, m, n, k, alpha, beta, lda, ldb, and ldc. Details of what these parameters are can be found below. 

There is an exponentially huge number of ways (kernels) for doing matrix multiplication on a gpu, and the best way (kernel) depends on the dimensions (m,n,k) and memory layout of the matrices. For a given GEMM problem, tinygemm performs a non-exhaustive search through the kernels, and returns the best (fastest) that it finds in an allotted user-defined time. The goal of tinygemm is to rapidly find a reasonable kernel, and then if given enough time, to find an exceptional kernel.


## Configuring
This may change soon (01 Dec 2016).

If you do want to install experimental code (which requires Cython), comment out add_subdirectory(dev) in ./CMakeLists.txt, so that it looks like this, 
#add_subdirectory(dev)

In CMakeLists.txt, you need to set a path to an empty and existing directory where tinygemm can write and delete, 
add_definitions(-DDIR_FOR_WRITING="/path/to/where/tinygemm/can/write/and/delete/")


## Building
This may change soon (01 Dec 2016).

In testsexamples/basicexample.cpp, set std::string logfile to the full path to where log output should be written by the basic example, or to an empty string if you'd prefer no logging, So that it looks like this:
std::string logfile("/some/path/to/a/findlog.txt");

mkdir build
cd build
cmake ..
(if this fails, you may to need to set OpenCL_INCLUDE_DIR and OpenCL_LIBRARY in CMakeCache.txt and cmake again)
make

## Installing
Currently (01 Dec 2016) tinygemm cannot be installed on the system path.


## Usage



## GEMM problem parameters


## Hyperparameters 


## Heuristic graph search
