/* functions for the end-user */ 


#ifndef OPENCLGEMMAPI_HPP
#define OPENCLGEMMAPI_HPP

#include  <CL/cl.h> 

#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/solution.hpp>
#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/outputwriter.hpp>
#include <MIOpenGEMM/findparams.hpp>

namespace MIOpenGEMM{


static const double default_alpha = 0.415693029182345929;
static const double default_beta = 0.273539340934809345;


Solution
find(
/* in seconds */
cl_command_queue command_queue,
const FindParams & find_params,
cl_mem a,   
cl_mem b,
cl_mem c,
cl_mem workspace,
/* a substring of a hyperstring, defines hyper-parameters with fixed values */
const std::string constraints_string,
/* see MIOpenGEMM/include/geometry.hpp for Geometry parameters */
const Geometry & gg,
/* this is nec so that we know where in a,b,c and workspace to start */
const Offsets & toff,
outputwriting::OutputWriter & mowri,
/* if c_is_const == false, then c will be corrupted */
bool c_is_const,

bool use_mowri_tracker);




Solution
get_default(
/* use this to extract device info */
cl_command_queue command_queue,
std::string constraints_string,
const Geometry & gg, 
std::string k_comment,
outputwriting::OutputWriter & mowri
);

std::tuple<bool, std::string> 
check_for_default(
cl_command_queue command_queue,
std::string constraints_string,
const Geometry & gg, 
std::string k_comment);

void benchgemm(
cl_command_queue command_queue, 
const std::string & hyperstring,
unsigned n_runs,
const Geometry & gg,
const Offsets & toff,
cl_mem a,
cl_mem b,
cl_mem c,
cl_mem workspace,
outputwriting::OutputWriter & mowri,
bool c_is_const = false);


/* reduced form, patch for miopen */
Solution
find(float allotted_time, cl_command_queue command_queue, cl_mem a, cl_mem b, cl_mem c, bool enforce_determinism, const Geometry & tgg);

Solution
get_default(const Geometry & gg);

} //namespace
#endif
