/* functions for the end-user */ 


#ifndef OPENCLGEMMAPI_HPP
#define OPENCLGEMMAPI_HPP

#include <vector> 
#include  <CL/cl.h> 

#include "problemgeometry.hpp"
#include "tinygemmsolution.hpp"

namespace clgemm{


/* floattype
 * 'f' : 32-bit single precision
 * 'd' : 64-bit double precision
 * the user must guarantee that a,b and c are in agreement with floattype. 
 * 
 * TODO : is there a way to signal that a,b should be CL_MEM_READ_ONLY ?
 * */


tinygemm::TinyGemmSolution
find(
float allotted_time,
cl_command_queue & command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
const bool enforce_deterministic,
const char floattype, 
const gemmgeometry::Geometry & gg,
const double alpha,
const double beta, 
bool verbose = false,   
std::string logfile = "");



void benchgemm(
cl_command_queue & command_queue, 
std::string kernelfilename,         // could be a .cl file, or a file with a list of the parameters required to make a .cl file with the python script (TODO).
unsigned n_runs,
const char floattype, 
const gemmgeometry::Geometry & gg,
const double alpha,                 // alpha and beta will have precision changed if necessary (in impl)
const double beta,
cl_mem a,                           
cl_mem b, 
cl_mem c,
bool verbose = true,
std::string logfile = "");


/* I have a rough idea what I want this to do : it will check that strictly smaller problems that MNK always at least as fast as MNK. If they're not, they should be using MNK's kernel, update this in the cache table. Of course, should check that MNK's tile is smaller than the matrix C being updated. TODO */
void kernel_cache_audit(std::string floattype);

} //namespace
#endif
























