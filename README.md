# tinygemm


## Installation

This is not yet polished.

If you do want to install experimental and testing code (which requires Cython), comment out add_subdirectory(dev) in ./CMakeLists.txt
 #add_subdirectory(dev)

Now
mkdir build
cd build
cmake ..
If cmake failed, you may to need to set OpenCL_INCLUDE_DIR and OpenCL_LIBRARY in CMakeCache.txt (or via ccmake)  and cmake again
make

## Usage

Check out the code in ./examples, the binaries are in build/examples.
