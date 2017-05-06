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

/* one of Default and Random */
FindStartType fst,

/* see tinygemm/include/tinygemmgeometry.hpp for TinyGemmGeometry parameters */
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
outputwriting::OutputWriter & mowri,
/* if c_is_const == false, then c will be corrupted */
bool c_is_const = true);


tinygemm::TinyGemmSolution
get_default(
std::string constraint_string,
const tinygemm::TinyGemmGeometry & gg,
bool verbose = false, 
std::string logfile = "");

void benchgemm(
cl_command_queue command_queue, 
const std::string & hyperstring,
//const std::vector<hyperparams::HyperParams> & hps,         
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
