/* functions for the end-user */ 


#ifndef OPENCLGEMMAPI_HPP
#define OPENCLGEMMAPI_HPP

#include  <CL/cl.h> 

#include "tinygemmgeometry.hpp"
#include "tinygemmsolution.hpp"
#include "tinygemmerror.hpp"
#include "outputwriter.hpp"

namespace tinygemm{


static const double default_alpha = 0.415693029182345929;
static const double default_beta = 0.273539340934809345;

tinygemm::TinyGemmSolution
find(
/* in seconds */
float allotted_time, 
cl_command_queue command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
cl_mem workspace,
/* a substring of a hyperstring, defines hyper-parameters with fixed values */
const std::string constraint_string,
/* see tinygemm/include/tinygemmgeometry.hpp for TinyGemmGeometry parameters */
const tinygemm::TinyGemmGeometry & gg,
/* this is nec so that we know where in a,b,c and workspace to start */
const tinygemm::TinyGemmOffsets & toff,
outputwriting::OutputWriter & mowri,
/* if c_is_const == false, then c will be corrupted */
bool c_is_const = true);


tinygemm::TinyGemmSolution
get_default(
/* use this to extract device info */
cl_command_queue command_queue,
std::string constraint_string,
const tinygemm::TinyGemmGeometry & gg,
outputwriting::OutputWriter & mowri);


void benchgemm(
cl_command_queue command_queue, 
const std::string & hyperstring,
unsigned n_runs,
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
cl_mem a,
cl_mem b,
cl_mem c,
cl_mem workspace,
outputwriting::OutputWriter & mowri,
bool c_is_const = false);


} //namespace
#endif
