# MIOpenGEMM

An OpenCL general matrix multiplication (GEMM) API and kernel generator. More information is available on the [wiki](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/wiki). 

## Prerequisites
* OpenCL - OpenCL libraries and header files
* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) 

## Configure with cmake

First create a build directory:
```
mkdir build; cd build;
```

Next configure cmake; if OpenCL is installed in one of the standard locations, 
```
cmake ..
```

otherwise manually set OpenCL cmake variables, either in ` CMakeCache.txt ` or

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

## Use the library

MIOpenGEMM provides an OpenCL GEMM API in ` gemm.hpp `, which should be included in your C++ source file 

```c++
#include <miopengemm/gemm.hpp>
```  

The key function is 
```c++
template <typename T>
MIOpenGEMM::GemmStatus xgemm(...)
```
which provides the same functionality as clBLAS' `clblasSgemm` and `clblasDgemm`. Currently only `T=float` and `T=double` are supported. More information on `xgemm` can be found on the wiki [here](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/wiki).

To obtain just OpenCL kernel strings without executing GEMM, one can use ` miogemm.hpp ` , as done by [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen).  

## Run the test

Currently there is 1 basic test, which can be run with

```
make smallgeometrytests
./tests/smallgeometrytests 
```
or
```
make check
```

## Build the examples

All examples can be built with 
```
make examples
```
or individually by name, for example
```
make find
```

The examples are described on the the wiki [here](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/wiki).


## Build the documentation
HTML and PDF documentation can be built using:

`make doc`

This will build a local searchable web site inside the ./MIOpenGEMM/doc/html folder and a PDF document inside the ./MIOpenGEMM/doc/pdf folder.

Documentation is generated using [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html) and should be installed separately.

HTML and PDFs are generated using [Sphinx](http://www.sphinx-doc.org/en/stable/index.html) and [Breathe](https://breathe.readthedocs.io/en/latest/), with the [ReadTheDocs theme](https://github.com/rtfd/sphinx_rtd_theme).

Requirements for both Sphinx, Breathe, and the ReadTheDocs theme can be filled for these in the MIOpenGEMM/doc folder:

`pip install -r ./requirements.txt`

Depending on your setup `sudo` may be required for the pip install.
