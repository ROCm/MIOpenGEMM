#include <string>
#include <sstream>

#include <tinygemm/betagenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/generatorutil.hpp>

//#include <tinygemm/config.hpp>


//TODO : (more for alpha kernel : consider using the heyword restrict more liberally). Some initial thought suggests to me that it more be more approprialtely pult on c. where did the use og restrict inherit from anywayt??? TODO TODO TODO

namespace tinygemm{
namespace betagen{


const size_t work_per_thread = 4;
const size_t n_work_items_per_group = 64;


class ForallGenerator{

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
  char mxz;

void set_beta_derived(){
  
  if (type.compare("betac") == 0){
    mxz = 'c';
  }
  
  else if (type.compare("movea") == 0){
    mxz = 'a';
  }
  
  else if (type.compare("moveb") == 0){
    mxz = 'b';
  }
  
  else{
    throw tinygemm_error("Unrecognised type in betagenerator.cpp");
  }
  
  n_full_work_items_per_line = gg.get_coal(mxz)  / work_per_thread;
  n_work_items_per_line = n_full_work_items_per_line + (gg.get_coal(mxz) % work_per_thread != 0);
  n_full_work_items = n_full_work_items_per_line*gg.get_uncoal(mxz);
  n_work_items = n_work_items_per_line*gg.get_uncoal(mxz);
  start_in_coal_last_work_item = work_per_thread*n_full_work_items_per_line;
  work_for_last_item_in_coal = gg.get_coal(mxz) % work_per_thread;
}


void append_description_string(std::stringstream & ss){

  if (type.compare("betac") == 0){
  ss << 
R"(


/* ****************************************************
* It is used to perform the beta*C step in GEMM, 
* where recall GEMM has C <- alpha*A*B + beta*C
* It is not quite an axpy, as when ldc is not minimal, 
* C is not contiguous memory  
****************************************************** */ )";

  }
  
  
}


void append_what_definitions(std::stringstream & ss){
  ss << "#define TFLOAT  "  << dp.t_float << "\n";
  ss<< "#define LDX " << gg.get_ld(mxz) << "\n" << 
"/* less than or equal to LDX, DIM_COAL is size in the contiguous direction (m for c matrix if col contiguous and not transposed) */ \n" << 
"#define DIM_COAL " << gg.get_coal(mxz) << "\n" <<
"/* DIM_UNCOAL is the other dimension of the matrix */ \n" << 
"#define DIM_UNCOAL " << gg.get_uncoal(mxz) << "\n\n";
}



void append_how_definitions(std::stringstream & ss){
  ss << 
R"(/* The number of values from C which each non-edge work-item will scale by beta */
#define WORK_PER_THREAD  )" << work_per_thread  << R"(
/* The number of work items per work group
 * TODO : generalise for vega support */
#define N_WORK_ITEMS_PER_GROUP )" << n_work_items_per_group << "\n\n";
}


void append_derived_definitions(std::stringstream & ss){
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
}


size_t get_local_work_size(){
  return n_work_items_per_group;
}

size_t get_global_work_size(){
  size_t number_of_betac_work_groups = (n_work_items / n_work_items_per_group) + ((n_work_items % n_work_items_per_group) != 0); 
  size_t betac_global_work_size = number_of_betac_work_groups * n_work_items_per_group;
  return betac_global_work_size;
}

void append_function_definition(std::stringstream & ss){
  if (type.compare("betac") == 0){
    ss << kernelname << "(const unsigned x_offset, __global TFLOAT * x, TFLOAT beta)";
  }
  
  else if (type.compare("copya") == 0 || type.compare("copyb") == 0){
    ss << kernelname << "(const unsigned x_offset, __global const TFLOAT * restrict x, const unsigned y_offset, __global TFLOAT * y)";
  }
}


void append_setup_coordinates(std::stringstream & ss){
  
    ss << R"(
    
    
/* setting up where this thread works */
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

)";
}

void append_positioning_x_string(std::stringstream & ss){
ss << R"(

/* moving the x pointer to the first element to process */
x += x_offset;
x += start_uncoal * LDX;
x += start_coal;
)";
}



public:
  ForallGenerator(const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, std::string type_): 
  gg(gg_), dp(dp_), type(type_), kernelname(genutil::get_generic_kernelname(type))
  {
    set_beta_derived();
  }


private:


void append_inner_work(std::stringstream & ss){
  if (type.compare("betac") == 0){
    ss << "/* the beta scaling */\n";
    ss << "x[i] *= beta;";
  }
  
  else{
    ss << "y[i] = x[i];";
  }
}

void append_work_string(std::stringstream & ss){
  
  
  ss << 
R"(
if (is_in_full_zone){
#pragma unroll WORK_PER_THREAD
for (unsigned i = 0; i < WORK_PER_THREAD; ++i){  )";
  append_inner_work(ss);
  ss << "\n}\n}\n";

  ss << R"(
else if (global_id < N_WORK_ITEMS){
for (unsigned i = 0; i < WORK_FOR_LAST_ITEM_IN_COAL; ++i){  )";
append_inner_work(ss);
  ss << "\n}\n}\n";
 
}


void append_positioning_y_string(std::stringstream & ss){
  ss << R"(

/* moving the y pointer to the first element to process */
y += GLOBAL_OFFSET;
y += y_offset;
y += start_uncoal * LDY;
y += start_coal;
)";
}



public:

KernelString get_forall_kernelstring(){

  std::stringstream ss;

  ss << genutil::get_time_string(type);
  append_description_string(ss);

  ss << "\n\n" << genutil::get_what_string() << "\n";
  append_what_definitions(ss);

  ss << genutil::get_how_string() << "\n";
  append_how_definitions(ss);

  ss << genutil::get_derived_string() << "\n";
  append_derived_definitions(ss);
  
  ss << "\n\n" << "__attribute__((reqd_work_group_size(N_WORK_ITEMS_PER_GROUP,1,1)))" << "\n";
  ss << "__kernel void ";

  append_function_definition(ss);

  ss << "{";
  
  append_setup_coordinates(ss);
  append_positioning_x_string(ss);

  if (type.compare("copya") == 0 || type.compare("copyb") == 0){
    append_positioning_y_string(ss);
  }
  
  append_work_string(ss);
  
  ss << "\n}\n\n\n";

  return {type, ss.str(), kernelname, get_global_work_size(), get_local_work_size()};
}


};



KernelString get_beta_kernelstring(const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 ForallGenerator fg(gg, dp, "betac");
 return fg.get_forall_kernelstring();
}

KernelString get_copya_kernelstring(const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 ForallGenerator fg(gg, dp, "copya");
 return fg.get_forall_kernelstring();
}


}
}







