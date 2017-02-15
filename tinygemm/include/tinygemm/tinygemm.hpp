/* functions for the end-user */ 


#ifndef OPENCLGEMMAPI_HPP
#define OPENCLGEMMAPI_HPP

#include  <CL/cl.h> 

#include "tinygemmgeometry.hpp"
#include "tinygemmsolution.hpp"
#include "tinygemmerror.hpp"

namespace tinygemm{



static const double default_alpha = 0.415693029182345929;
static const double default_beta = 0.273539340934809345;


//tinygemm::TinyGemmSolution
//get_default(
//const bool enforce_deterministic, 
//const tinygemm::TinyGemmGeometry & gg
//);



tinygemm::TinyGemmSolution
find(
/* in seconds */
float allotted_time, 
cl_command_queue command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
cl_mem workspace,
const bool enforce_deterministic,

/* see tinygemm/include/tinygemmgeometry.hpp for TinyGemmGeometry parameters */
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
bool verbose = false,
std::string logfile = "", 
/* if c_is_const == false, then c will be corrupted */
bool c_is_const = true);


tinygemm::TinyGemmSolution
get_default(
const bool enforce_deterministic,
const tinygemm::TinyGemmGeometry & gg,
bool verbose = false, 
std::string logfile = "");

void benchgemm(
cl_command_queue command_queue, 
const std::vector<hyperparams::HyperParams> & hps,         
unsigned n_runs,

const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
cl_mem a,
cl_mem b,
cl_mem c,
cl_mem workspace,
bool verbose = true,
std::string logfile = "",
bool c_is_const = false);


/* (experimental) I have a rough idea what I want this to do : it will check that strictly smaller problems that MNK always at least as fast as MNK. If they're not, they should be using MNK's kernel, update this in the cache table. Of course, should check that MNK's tile is smaller than the matrix C being updated. TODO */
void kernel_cache_audit(std::string floattype);

} //namespace
#endif
