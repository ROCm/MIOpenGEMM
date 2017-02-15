#include <string>
#include <sstream>

#include <tinygemm/forallgenerator.hpp>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/generatorutil.hpp>



/* TODO : (more for alpha kernel : consider using the keyword restrict more liberally). Some initial thought suggests to me that it more be more approprialtely pult on c. where did the use og restrict inherit from anywayt??? TODO TODO TODO */

/* TODO : could interwoven be faster ? */
namespace tinygemm{
namespace forallgen{


/* TODO : these so need to made hyper parameters, (1, 2, 4), (64, 128, 256)  */

const size_t work_per_thread = 2;
const size_t n_work_items_per_group = 256;

//const size_t work_per_thread = 4;
//const size_t n_work_items_per_group = 64;

class ForallGenerator{

private:
  const tinygemm::hyperparams::HyperParams & hp;
  const tinygemm::TinyGemmGeometry & gg;
  const tinygemm::derivedparams::DerivedParams & dp;


  std::string forall_type;
  std::string kernelname;

  unsigned n_full_work_items_per_line;
  unsigned n_work_items_per_line;
  unsigned n_full_work_items;
  unsigned n_work_items;
  unsigned start_in_coal_last_work_item;
  unsigned work_for_last_item_in_coal;
  
  
  bool uses_a; 
  bool uses_b;
  bool uses_c;
  bool uses_workspace;
  bool uses_alpha;
  bool uses_beta;
  char mxz;  
  std::string description_string;
  std::string function_definition;
  std::string inner_work_string;

void set_forall_derived(){

  uses_alpha = false;  
  
  if (forall_type.compare("betac") == 0){
    uses_a = false;
    uses_b = false;
    uses_c = true;
    uses_workspace = false;
    uses_beta = true;
    mxz = 'c';
    description_string = R"(
/* ****************************************************
* It is used to perform the beta*C step in GEMM, 
* where recall GEMM has C <- alpha*A*B + beta*C
* It is not quite an axpy, as when ldc is not minimal, 
* C is not contiguous memory  
****************************************************** */ )";
    function_definition = "(__global TFLOAT * x, const unsigned x_offset, TFLOAT beta)";
    inner_work_string = "\n/* the beta scaling */\nx[i] *= beta;";
  }
    
  else if (forall_type.compare("copya") == 0 || forall_type.compare("copyb") == 0){
    uses_c = false;
    uses_workspace = true;
    uses_beta = false;
    description_string = R"()";
    function_definition = "(__global const TFLOAT * restrict x, const unsigned x_offset, __global TFLOAT * y, const unsigned y_offset)";
    inner_work_string = "\n/* the copy */\ny[i] = x[i];";
    
    if (forall_type.compare("copya") == 0){    
      uses_a = true;
      uses_b = false;    
      mxz = 'a';
    }
    else{
      uses_a = false;
      uses_b = true;    
      mxz = 'b';
    }
  }
    
  else{
    throw tinygemm_error("Unrecognised forall_type in forallgenerator.cpp : " + forall_type + ".\n");
  }
  
  
  n_full_work_items_per_line = gg.get_coal(mxz)  / work_per_thread;
  n_work_items_per_line = n_full_work_items_per_line + (gg.get_coal(mxz) % work_per_thread != 0);
  n_full_work_items = n_full_work_items_per_line*gg.get_uncoal(mxz);
  n_work_items = n_work_items_per_line*gg.get_uncoal(mxz);
  start_in_coal_last_work_item = work_per_thread*n_full_work_items_per_line;
  work_for_last_item_in_coal = gg.get_coal(mxz) % work_per_thread;
}


public:
  ForallGenerator(const tinygemm::hyperparams::HyperParams & hp_,  const tinygemm::TinyGemmGeometry & gg_, const tinygemm::derivedparams::DerivedParams & dp_, std::string forall_type_): 
  hp(hp_), gg(gg_), dp(dp_), forall_type(forall_type_), kernelname(genutil::get_generic_kernelname(forall_type))
  {
    set_forall_derived();
  }


private:

void append_description_string(std::stringstream & ss){
  ss <<  description_string;
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


void append_copy_preprocessor(std::stringstream & ss){
  if (hp.a_copy_workspace == 1 && (forall_type.compare("copyb") == 0)){
    ss << "/*      b will be copied to a section of workspace after where a is copied */\n";
    ss << "#define GLOBAL_OFFSET_B " <<  dp.bcw1_global_offset_b << "\n";
  }
  
  else{
    ss << "/*      no global offset as this is the first or only matric being copied to workspace */\n";
    ss << "#define GLOBAL_OFFSET_B  0\n";
  }
  
  ss << "/*      the target stride between lines, derived from hp and gg (see DerivedParams) */\n";
  ss << "#define LDY " << dp.get_target_ld(mxz) << "\n";
  
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
  if (forall_type.compare("copya") == 0 || forall_type.compare("copyb") == 0){
    append_copy_preprocessor(ss);
  }
    
}


size_t get_local_work_size(){
  return n_work_items_per_group;
}

size_t get_global_work_size(){
  size_t number_of_forall_work_groups = (n_work_items / n_work_items_per_group) + ((n_work_items % n_work_items_per_group) != 0); 
  size_t forall_global_work_size = number_of_forall_work_groups * n_work_items_per_group;
  return forall_global_work_size;
}

void append_function_definition(std::stringstream & ss){
  ss << kernelname << function_definition;
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





private:


void append_inner_work(std::stringstream & ss){
  ss << inner_work_string;  
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
y += GLOBAL_OFFSET_B;
y += y_offset;
y += start_uncoal * LDY;
y += start_coal;
)";
}



public:

KernelString get_forall_kernelstring(){

  std::stringstream ss;

  ss << genutil::get_time_string(forall_type);
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

  if (forall_type.compare("copya") == 0 || forall_type.compare("copyb") == 0){
    append_positioning_y_string(ss);
  }
  
  append_work_string(ss);
  
  ss << "\n}\n\n\n";




  return {{uses_a, uses_b, uses_c, uses_workspace, uses_alpha, uses_beta}, ss.str(), kernelname, get_global_work_size(), get_local_work_size()};
}


};




KernelString get_beta_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 ForallGenerator fg(hp, gg, dp, "betac"); //haha hehe.
 return fg.get_forall_kernelstring();
}

KernelString get_copya_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 ForallGenerator fg(hp, gg, dp, "copya");
 return fg.get_forall_kernelstring();
}

KernelString get_copyb_kernelstring(const tinygemm::hyperparams::HyperParams & hp, const tinygemm::TinyGemmGeometry & gg, const tinygemm::derivedparams::DerivedParams & dp){
 ForallGenerator fg(hp, gg, dp, "copyb");
 return fg.get_forall_kernelstring();
}


}
}
