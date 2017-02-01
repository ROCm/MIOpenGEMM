# tinygemm

tinygemm is a tool for generating OpenCL matrix multiplication (GEMM) kernels. The project is inspired by, and very similar to, [Tensile](https://github.com/RadeonOpenCompute/Tensile), a general purpose tool for tensor problems.

## Background

A GEMM problem,  `c.tC = alpha * a.tA * b.tB + beta * c.tC`, is defined by the dimensions and memory layout of matrices a,b and c. These 10 *geometry* parameters are isColMaj, tA, tB, tC, m, n, k, lda, ldb, and ldc, to be described later.

The standard matrix multiplication algorithm requires `2mnk` floating point operations to solve a GEMM problem is. A device with `t` threads of clock-speed `f` *should* solve a GEMM problem in time `2mnk/(ft)`.  On an ideal device where all data is accessible in a single cycle, obtaining this lower bound would be trivial, but in practice moving data to a register, where it is ready for computing, takes *thousands* of cycles. A good GEMM kernel is one which hides such *latency*.

There are three levels of accessible GPU memory, referred to with OpenCL lingo as global, local, and private. Transferring data between these layers can be done in several different ways. We refer to the parameters controlling how and where threads read and write at each of the three layers as *hyper-parameters*. How well latency is hidden depends on these parameters.  

tinygemm currently defines 15 *hyper-parameters*, with most combinations defining a valid kernel. Thus there are an exponentially large number of GEMM kernels to consider. Moreover, the best hyper-parameters (the best kernel) depends on the geometry parameters, and so it is not a question of finding a \`best' kernel. 

For a given GEMM problem, tinygemm performs a non-exhaustive search through the hyper-parameters (kernels), and returns the best that it finds in an allotted user-defined time. The goal of tinygemm is to rapidly find a reasonable kernel, and then if given enough time, to find an exceptional kernel.


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
