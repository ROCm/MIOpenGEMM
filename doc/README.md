# MIOpenGEMM

A tool for generating OpenCL matrix multiplication (GEMM) kernels. More information is available at (link coming soon). 

## Prerequisites
* OpenCL - OpenCL libraries and header files
* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) 

## Configure with cmake

First create a build directory:
```
mkdir build; cd build;
```

Next configure cmake:
```
cmake ..
```


The above assumes that OpenCL is installed in one of the standard locations. If not, then manually set these two cmake variables, either in ` CMakeCache.txt ` or

```
cmake -DOPENCL_LIBRARIES=<opencl-library-path> -DOPENCL_INCLUDE_DIRS<opencl-headers-path> ..
```


By default the install location is set to '/opt/rocm', this can be set by using `CMAKE_INSTALL_PREFIX`:

```
cmake -DCMAKE_INSTALL_PREFIX=<miopen-installed-path> ..
```


## Build the library

The library can be built, from the `build` directory 

```
make miopengemm
```

And can be installed by using the 'install' target
```
make install
```

## Build the examples

All examples can be built with
```
make examples
```

Or individually for examples ` basicexample ` , ` deepbench ` , ` devtest ` , ` experiment1 ` , ` gencache ` , ` initialisationdemo ` , ` redirectionexample ` and ` smallgeometry ` with make "example-name"

## Usage 

To use MIOpenGEMM, the ` miopengemm.hpp ` header file should be included in your C++ source file 

```c++
#include <miopengemm/miopengemm.hpp>
```

The key function is
```c++
Solution find(
  float allotted_time,             // Amount of time [s] allotted to search for a solution
  cl_command_queue command_queue,  // OpenCL command queue
  cl_mem a, cl_mem b, cl_mem c,    // Read-only OpenCL memory buffers
  bool enforce_determinism,        // Guarantee bit-wise fidelity with for-for-for GEMM.
  const Geometry & tgg,            // Matrix geometry, see below
  bool verbose,                    // Print summary information to terminal while searching. 
  bool with_warnings);             // Print performance warnings.
```

One of the parameters of ` find ` is a ` Geometry ` object, which describes the GEMM problem.  The ` Geometry ` class has a constructor with one std::string as argument, for example

```c++
Geometry tgg("tC0_tA0_tB1_colMaj0_m55_n65_k122_lda123_ldb124_ldc69_ws0_f32");
```

creates a Geometry for the Row-Major SGEMM problem ` C = alpha A B.T + beta C ` where A is 55 x 122, B.T is 122 x 65, and C is 55 x 65. lda, ldb, and ldc are the standard matrix paddings. ` tC ` allows for C to be transposed, which is not in the standard GEMM API. ` ws ` is currently an experimental work-space parameter, and should be left as zero. MIOpenGEMM currently supports SGEMM (f32) and DGEMM (f64).


` find ` returns a ` Solution ` object. A ` Solution ` object has member ` std::vector<KernelString> v_tgks ` where each KernelString in the vector has members

```c++
  std::string kernstr;      // OpenCL source
  std::string fname;        // The name of the kernel function in kernstr
  size_t global_work_size;  // The global work size to be passed to clEnqueueNDRangeKernel 
  size_t local_work_size;   // The local work size to be passed to clEnqueueNDRangeKernel
```

Note that the ` Solution ` object returned by ` find ` is only valid for the ` Geometry ` passed as to ` find ` . Currently, there are either 1 or 2 `KernelString`s in ` v_tgks ` , to be executed serially. 


If a ` Geometry ` is used frequently, it is possible to cache the ` Solution `  for future use, so that ` find ` does not need to be re-run. See the example ` gencache ` to see how this is done. Cached Solutions can be retrieved with the function ` get_default ` .  


## Building the documentation

Instructions for building the documentation are identical to those for [MIOpen](https://github.com/AMDComputeLibraries/MLOpen#building-the-documentation) 


