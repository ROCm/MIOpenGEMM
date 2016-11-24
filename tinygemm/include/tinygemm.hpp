/* functions for the end-user */ 


#ifndef OPENCLGEMMAPI_HPP
#define OPENCLGEMMAPI_HPP

#include  <CL/cl.h> 

#include "tinygemmgeometry.hpp"
#include "tinygemmsolution.hpp"
#include "tinygemmerror.hpp"

namespace tinygemm{

tinygemm::TinyGemmSolution
get_default(
const bool enforce_deterministic, 
const char floattype, 
const tinygemm::TinyGemmGeometry & gg
);


tinygemm::TinyGemmSolution
find(
/* in seconds */
float allotted_time, 
cl_command_queue command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
const bool enforce_deterministic,
/* floattype
 * 'f' : 32-bit single precision
 * 'd' : 64-bit double precision 
 * the user must guarantee that a,b and c are in agreement with floattype, 
 * TODO is there a way to check float type from a,b,c? If so, floattype is not nec. */
const char floattype,
/* see tinygemm/include/tinygemmgeometry.hpp for TinyGemmGeometry parameters */
const tinygemm::TinyGemmGeometry & gg,
const double alpha,
const double beta, 
bool verbose = false,   
std::string logfile = "");





void benchgemm(
cl_command_queue  command_queue, 
std::string kernelfilename,         // name of a .cl file
unsigned n_runs,
const char floattype, 
const tinygemm::TinyGemmGeometry & gg,
const double alpha,                 
const double beta,
cl_mem a,                           
cl_mem b, 
cl_mem c,
bool verbose = true,
std::string logfile = "");


/* (experimental) I have a rough idea what I want this to do : it will check that strictly smaller problems that MNK always at least as fast as MNK. If they're not, they should be using MNK's kernel, update this in the cache table. Of course, should check that MNK's tile is smaller than the matrix C being updated. TODO */
void kernel_cache_audit(std::string floattype);

} //namespace
#endif
