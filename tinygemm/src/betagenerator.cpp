#include <string>
#include <sstream>

#include <tinygemm/betagenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/generatorutil.hpp>

//#include <tinygemm/config.hpp>

namespace tinygemm{
namespace betagen{


const size_t work_per_thread = 4;
const size_t n_work_items_per_group = 64;


class BetaGenerator{

private:
  const tinygemm::TinyGemmGeometry & gg;
  const tinygemm::derivedparams::DerivedParams & dp;
  std::string type;
  std::string kernelname;

  unsigned n_full_work_items_per_line;
  unsigned n_work_items_per_line;
  unsigned n_full_work_items;
  unsigned n_work_items;
  unsigned start_in_coal_last_work_item;
  unsigned work_for_last_item_in_coal;

void set_beta_derived(){
  n_full_work_items_per_line = gg.derived.dim_c_coal / work_per_thread;
  n_work_items_per_line = n_full_work_items_per_line + (gg.derived.dim_c_coal % work_per_thread != 0);
  n_full_work_items = n_full_work_items_per_line*gg.derived.dim_c_uncoal;
  n_work_items = n_work_items_per_line*gg.derived.dim_c_uncoal;
  start_in_coal_last_work_item = work_per_thread*n_full_work_items_per_line;
  work_for_last_item_in_coal = gg.derived.dim_c_coal % work_per_thread;
}

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
  BetaGenerator(const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_): 
  gg(gg_), dp(dp_), type("betac"), kernelname(genutil::get_generic_kernelname(type))
  {
    set_beta_derived();
  }


KernelString get_betac_kernelstring(){

  std::stringstream ss;

  ss << genutil::get_time_string(type);
  ss << 
R"(

/* ****************************************************
 * It is used to perform the beta*C step in GEMM, 
 * where recall GEMM has C <- alpha*A*B + beta*C
 * It is not quite an axpy, as when ldc is not minimal, 
 * C is not contiguous memory  
 ****************************************************** */ )";
 
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
#define N_WORK_ITEMS_PER_GROUP )" << n_work_items_per_group << "\n\n";

  ss << genutil::get_derived_string() << "\n";
  ss << "/*      each (full) work item will process WORK_PER_THREAD elements in the coalesced direction, */ \n";
  ss << "/*      so the number of work items per coalesced line is DIM_COAL / WORK_PER_THREAD */ \n"; 
  ss << "#define N_FULL_WORK_ITEMS_PER_LINE " << n_full_work_items_per_line << "\n";
  ss << "/*      including the possible final tail thread, */\n";
  ss << "/*      there are N_FULL_WORK_ITEMS_PER_LINE + (DIM_COAL % WORK_PER_THREAD != 0) */ \n";
  ss << "#define N_WORK_ITEMS_PER_LINE " << n_work_items_per_line << "\n";
  ss << "/*      in total there are N_FULL_WORK_ITEMS_PER_LINE * DIM_UNCOAL full work items, */ \n";
  ss << "#define N_FULL_WORK_ITEMS " << n_full_work_items << "\n";
  ss << "/*      and a grand total of N_WORK_ITEMS_PER_LINE * DIM_UNCOAL work items. */ \n";
  ss << "#define N_WORK_ITEMS " << n_work_items << "\n";  
  ss << "/*      tail work items start at WORK_PER_THREAD * N_FULL_WORK_ITEMS_PER_LINE in the coalesced direction,  */\n";
  ss << "#define START_IN_COAL_LAST_WORK_ITEM " << start_in_coal_last_work_item <<  "\n";
  ss << "/*      and process DIM_COAL % WORK_PER_THREAD elements of c */\n";
  ss << "#define WORK_FOR_LAST_ITEM_IN_COAL " << work_for_last_item_in_coal << "\n";
  
  ss << 
R"(

__attribute__((reqd_work_group_size(N_WORK_ITEMS_PER_GROUP,1,1)))
__kernel void )"  << kernelname << R"((const unsigned c_offset, __global TFLOAT * c, TFLOAT beta){


c += c_offset;

unsigned group_id = get_group_id(0);
unsigned local_id = get_local_id(0);
unsigned global_id = group_id*N_WORK_ITEMS_PER_GROUP + local_id; 


unsigned start_uncoal = 0;
unsigned start_coal = 0;

bool is_in_full_zone = (global_id < N_FULL_WORK_ITEMS);
if (is_in_full_zone){   
start_uncoal = global_id / N_FULL_WORK_ITEMS_PER_LINE;
start_coal = WORK_PER_THREAD * (global_id % N_FULL_WORK_ITEMS_PER_LINE);
}

else if (global_id < N_WORK_ITEMS){
start_uncoal = (global_id - N_FULL_WORK_ITEMS)% DIM_UNCOAL;
start_coal = START_IN_COAL_LAST_WORK_ITEM;
}

c += start_uncoal * LDC;
c += start_coal;

if (is_in_full_zone){
#pragma unroll WORK_PER_THREAD
for (unsigned i = 0; i < WORK_PER_THREAD; ++i){
c[i] *= beta;
}
}

else if (global_id < N_WORK_ITEMS){
for (unsigned i = 0; i < (WORK_FOR_LAST_ITEM_IN_COAL); ++i){
c[i] *= beta;
}
}
}



)";

  return {type, ss.str(), kernelname, get_global_work_size(), get_local_work_size()};
}


};



KernelString get_beta_kernelstring(const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 BetaGenerator bg(gg, dp);
 return bg.get_betac_kernelstring();
}


}
}







