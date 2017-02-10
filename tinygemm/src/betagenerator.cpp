#include <string>
#include <sstream>

#include <tinygemm/betagenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/generatorutil.hpp>

//#include <tinygemm/config.hpp>

namespace tinygemm{
namespace betagen{


static const std::string genericbetackernelname = "tg_betac";
const size_t work_per_thread = 4;
const size_t n_work_items_per_group = 64;


class BetaGenerator{

private:
  const tinygemm::TinyGemmGeometry & gg;
  const tinygemm::derivedparams::DerivedParams & dp;

size_t get_local_work_size(){
  return n_work_items_per_group;
}

size_t get_global_work_size(){
  size_t n_betac_threads = gg.derived.dim_c_uncoal*(gg.derived.dim_c_coal/work_per_thread + ((gg.derived.dim_c_coal%work_per_thread) != 0));
  size_t number_of_betac_work_groups = (n_betac_threads / n_work_items_per_group) + ((n_betac_threads % n_work_items_per_group) != 0) ; 
  size_t betac_global_work_size = number_of_betac_work_groups*n_work_items_per_group;
  return betac_global_work_size;
}

public:
  BetaGenerator(const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_): gg(gg_), dp(dp_) {}


KernelString get_betac_kernelstring(){

  std::stringstream ss;
  std::string type = "betac";
  ss << genutil::get_time_string(type);
  ss << 
R"(

/* ****************************************************
 * It is used to perform the beta*C step in GEMM, 
 * where recall GEMM has C <- alpha*A*B + beta*C
 * It is not quite an axpy, as when ldc is not minimal, 
 * C is not contiguous memory  
 ****************************************************** / )";
 
  ss << "\n\n" << genutil::get_what_string() << "\n";
  ss << "#define TFLOAT  "  << dp.t_float << "\n";
 

  ss<< "#define LDC " << gg.ldc << "\n" << 
"/* less than or equal to ldc, DIM_COAL is size in the contiguous direction (m for c matrix if col contiguous and not transposed) */ \n" << 
"#define DIM_COAL " << gg.derived.dim_c_coal << "\n" <<
"/* DIM_UNCOAL is the other dimension of the matrix */ \n" << 
"#define DIM_UNCOAL " << gg.derived.dim_c_uncoal << "\n\n";

  ss << genutil::get_how_string() << R"(
/* The number of values from C which each non-edge work-item will scale by beta */
#define WORK_PER_THREAD  )" << work_per_thread  << R"(
/* The number of work items per work group
 * TODO : generalise for vega support */
#define N_WORK_ITEMS_PER_GROUP )" << n_work_items_per_group << "\n\n\n";



  
  
  
  
  
  ss <<
  
R"(

__attribute__((reqd_work_group_size(N_WORK_ITEMS_PER_GROUP,1,1)))
__kernel void )"  << genericbetackernelname << R"((const unsigned c_offset, __global TFLOAT * c, TFLOAT beta){


c += c_offset;

unsigned group_id = get_group_id(0);
unsigned local_id = get_local_id(0);
unsigned global_id = group_id*N_WORK_ITEMS_PER_GROUP + local_id; 

unsigned n_full_work_items_per_line = DIM_COAL / WORK_PER_THREAD;
unsigned n_work_items_per_line = n_full_work_items_per_line + (DIM_COAL % WORK_PER_THREAD != 0);

unsigned n_full_work_items = n_full_work_items_per_line*DIM_UNCOAL;
unsigned n_work_items = n_work_items_per_line*DIM_UNCOAL;

unsigned start_uncoal = 0;
unsigned start_coal = 0;

bool is_in_full_zone = (global_id < n_full_work_items);
if (is_in_full_zone){   
start_uncoal = global_id / n_full_work_items_per_line;
start_coal = WORK_PER_THREAD * (global_id % n_full_work_items_per_line);
}

else if (global_id < n_work_items){
start_uncoal = (global_id - n_full_work_items)% DIM_UNCOAL;
start_coal = WORK_PER_THREAD*n_full_work_items_per_line;
}

c += start_uncoal * LDC;
c += start_coal;

if (is_in_full_zone){
#pragma unroll WORK_PER_THREAD
for (unsigned i = 0; i < WORK_PER_THREAD; ++i){
c[i] *= beta;
}
}

else if (global_id < n_work_items){
for (unsigned i = 0; i < (DIM_COAL % WORK_PER_THREAD); ++i){
c[i] *= beta;
}
}
}


)";

  return {"betac", ss.str(), genericbetackernelname, get_global_work_size(), get_local_work_size()};
}

};



KernelString get_beta_kernelstring(const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 BetaGenerator bg(gg, dp);
 return bg.get_betac_kernelstring();
}


}
}







