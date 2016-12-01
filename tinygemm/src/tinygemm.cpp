#include "tinygemmerror.hpp"
#include <thread>
#include <sstream>
#include <limits>
#include <chrono>
#include <limits>
#include <vector> 

#include "tinygemm.hpp"
#include "consistencychecks.hpp"
#include "redirection.hpp"
#include "outputwriter.hpp"
#include "sizingup.hpp"
#include "kernelsnips.hpp"
#include "kernelchecks.hpp"
#include "openclutil.hpp"
#include "betackernelutil.hpp"
#include "floattostring.hpp"
#include "tinygemmsolution.hpp"
#include "hyperparams.hpp"
#include "makekernelsource.hpp"
#include "defaultoutpath.hpp"

namespace tinygemm{

/* Make these user defined parameters ? This is where all kernels will be written, overwriting was was previously there.  */
static std::string kerneldir = defpaths::scratchpadfinddir + "/";
static std::string kernelname = "atinygemmkernel";
static std::string kernelfilename = kerneldir + kernelname + ".cl";


void confirm_cl_status(cl_int ret, std::string function, std::string argument){
  if (ret != CL_SUCCESS){
    std::stringstream errms;
    errms << "failure to set cl kernel arg, " << argument << ", in function " << function << "." << " error!";
    throw tinygemm_error(errms.str());
  }
}

class MultiFloatType{

  private:
    double v_d;
    float v_f;  
    
  public:
    MultiFloatType(double v): v_d(v), v_f(static_cast<float>(v)) {}  
    void * operator [] (char floattype){
      return floattype == 'd' ? (void *)(& v_d) : (void * )(& v_f);
    }
};


/* Return the int parameters required by make_kernel, other than the hyper parameters */
std::map<std::string, unsigned> get_bare_bones_int_parms(const tinygemm::TinyGemmGeometry & gg){
  std::map<std::string, unsigned> all_int_parms;
  all_int_parms["is_col_major"] = gg.isColMajor;
  all_int_parms["a_transposed"] = gg.tA;
  all_int_parms["b_transposed"] = gg.tB;
  all_int_parms["c_transposed"] = gg.tC;
  all_int_parms["use_edge_trick"] = 1;
  return all_int_parms;
}


size_t get_floatsize(char floattype){
  size_t floatsize;
  if (floattype == 'f'){
    floatsize = sizeof(float);
  }
  else if (floattype == 'd'){
    floatsize = sizeof(double);
  }
  else{
    throw tinygemm_error("Unrecognised floattype char. Currently, only 'f' (single precision) and 'd' (double precision) are supported");
  }
  return floatsize;
}

class OpenCLGemmEncapsulator{
public: 
  /* references to constructor parameters */
  cl_command_queue command_queue;
  std::string & outputfilename;
  const char & floattype; 
  const tinygemm::TinyGemmGeometry & gg;
  const double & alpha;
  const double & beta;
  cl_mem & a_gpu;
  cl_mem & b_gpu; 
  cl_mem & c_gpu;


private:
  size_t floatsize;
  MultiFloatType m_alpha;
  MultiFloatType m_beta;
  outputwriting::OutputWriter mowri;
  /* parameters which will be needed to determine number of work groups and work items */
  unsigned macro_tile_width, macro_tile_height, n_workitems_per_workgroup, n_work_items_per_c_elm;
  /* parameters relevent for the case(s) where the beta scaling is performed by a separate kernel (such as when splitting-in-k) */
  unsigned does_betac_inc;
  size_t n_work_groups;
  size_t local_work_size;
  size_t global_work_size;
  cl_int ret;
  cl_program program = NULL;
  cl_kernel kernel = NULL;    
  std::string kernel_function_name;
  cl_program betac_program = NULL;
  cl_kernel betac_kernel = NULL;
  std::string betac_kernel_function_name;
  /* parameters related to the scaling (beta C) kernel */
  unsigned dim_coal, dim_uncoal;
  size_t betac_global_work_size, betac_local_work_size;
  /* used for getting performance of main and (possibly) betac kernels */
  cl_event event_main_kernel;
  cl_event event_betac_kernel;
  /* stores (the most recent of n_runs) execution time */
  size_t t_start_betac_kernel, t_end_betac_kernel, t_start_main_kernel, t_end_main_kernel;
  float t_total_with_both, t_just_scaling, t_just_main;
  /* vector times over all n_runs runs */
  std::vector<float> v_t_total_with_both, v_t_just_scaling, v_t_just_main;
  /* These are set in the constructor by querying command_queue */
  cl_context context;
  cl_device_id device_id_to_use;
  bool nobenching;


public:
  OpenCLGemmEncapsulator(
  cl_command_queue command_queue, 
  const char & floattype,
  const tinygemm::TinyGemmGeometry & gg,  
  const double & alpha,
  const double & beta,
  cl_mem & a_gpu,
  cl_mem & b_gpu, 
  cl_mem & c_gpu,
  std::string & outputfilename,
  bool verbose, 
  bool nobenching):

  command_queue(command_queue), 
  outputfilename(outputfilename),
  floattype(floattype), 
  gg(gg),
  alpha(alpha),
  beta(beta),
  a_gpu(a_gpu),
  b_gpu(b_gpu), 
  c_gpu(c_gpu),
  m_alpha(alpha),
  m_beta(beta),
  mowri(verbose, outputfilename.compare("") != 0, outputfilename),
  nobenching(nobenching)
  {
    set_floatsize();
    run_checks();
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
    if (ret != CL_SUCCESS){
      throw tinygemm_error("Problem using clGetCommandQueueInfo, trying to get CL_QUEUE_CONTEXT (in constructor in openclgemmapi.cpp");
    }
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id_to_use, NULL);
    if (ret != CL_SUCCESS){
      throw tinygemm_error("Problem using clGetCommandQueueInfo, trying to get CL_QUEUE_DEVICE (in constructor in openclgemmapi.cpp");        
    }      
    if (nobenching == false){ // || (std::abs(beta - 1.) < 1e-9)){
      setup_betac_kernel(); 
    }
  }
 
 
  ~OpenCLGemmEncapsulator(){
    if (nobenching == false){
      release_betac_kernel_and_program();
    }
  }
  
  void release_betac_kernel_and_program(){
    ret = clReleaseKernel(betac_kernel);
    if (ret != CL_SUCCESS){
      throw tinygemm_error("failed in clReleaseKernel, in release_betac_kernel_and_program");
    }
    
    ret = clReleaseProgram(betac_program);
    if (ret != CL_SUCCESS){
      throw tinygemm_error("failed in clReleaseProgram, in release_betac_kernel_and_program");
    }
  }


  void release_main_kernel_and_program(){
    ret = clReleaseKernel(kernel);
    if (ret != CL_SUCCESS){
      throw tinygemm_error("failed in clReleaseKernel, in release_main_kernel_and_program");
    }
    
    ret = clReleaseProgram(program);
    if (ret != CL_SUCCESS){
      throw tinygemm_error("failed in clReleaseProgram, in release_main_kernel_and_program");
    }
  }
  
  void run_checks(){
    if (c_gpu == a_gpu || c_gpu == b_gpu){
      throw tinygemm_error("c should be distinct from a and b for gemm, otherwise race condition arises (one thread writes its result to c before another one has finished reading from c)");
    }
    /* Confirm that the input parameters make sense */
    consistencychecks::check_ldx_mnk_consistent(gg);
    sizingup::check_sizes_ok_for_unsigned(gg);
    if (floattype != 'd' and floattype != 'f'){
      throw tinygemm_error("floattype should be one of 'f' and 'd'");
    }
  }


  void setup_betac_kernel(){
    /* jit compile the betackernel source  */
    openclutil::set_program_and_kernel(betac_program, betac_kernel, betac_kernel_function_name, context, device_id_to_use, betac::get_cl_file_path(floattype));
    betac::set_betackernel_sizes(floattype, gg.isColMajor, gg.tC, gg.m, gg.n, dim_coal, dim_uncoal, betac_global_work_size, betac_local_work_size);
    mowri << "in setup_betac_kernel, global_work_size : " << betac_global_work_size << Endl; 
    ret = clSetKernelArg(betac_kernel, 0, sizeof(unsigned), &dim_coal);
    confirm_cl_status(ret, "clSetKernelArg", "dim_coal");
    ret = clSetKernelArg(betac_kernel, 1, sizeof(unsigned), &dim_uncoal);
    confirm_cl_status(ret, "clSetKernelArg", "dim_uncoal");
    ret = clSetKernelArg(betac_kernel, 2, sizeof(unsigned), &gg.ldc);
    confirm_cl_status(ret, "clSetKernelArg", "ldc");
    ret = clSetKernelArg(betac_kernel, 3, sizeof(unsigned), &gg.c_offset);
    confirm_cl_status(ret, "clSetKernelArg", "ldc");
    ret = clSetKernelArg(betac_kernel, 4, sizeof(cl_mem), (void *)&c_gpu);
    confirm_cl_status(ret, "clSetKernelArg", "c_gpu");
    ret = clSetKernelArg(betac_kernel, 5, floatsize, m_beta[floattype]);// 
    confirm_cl_status(ret, "clSetKernelArg", "m_beta[floattype]");
  }



  void set_floatsize(){
    floatsize = get_floatsize(floattype);    
  }

  void check_file_and_set_from(std::string kernelfn){
    
    /* check that the parameters in the kernelfile look reasonable */
    kernelutil::check_gpu_kernel_preprocessor_parameters(kernelfn, gg.tA, gg.tB, gg.tC, gg.isColMajor, gg.m, gg.n, floattostring::get_float_string(floattype));
    
    /* extract the parameters which are needed to determine the number of work groups and work items to launch, directly from kernel source */
    kernelutil::set_sizes_from_kernel_source(macro_tile_width, macro_tile_height, n_workitems_per_workgroup, n_work_items_per_c_elm, does_betac_inc, kernelfn);
  }


  
  void set_workforce(){
    sizingup::set_workforce(n_work_groups, local_work_size, global_work_size, gg.m, gg.n, n_work_items_per_c_elm, macro_tile_height, macro_tile_width, n_workitems_per_workgroup);
  }


  void set_main_kernel_arguments(){
    /* set the alpha (alpha and beta) kernel parameters */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&c_gpu);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&a_gpu);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_gpu);
    ret = clSetKernelArg(kernel, 3, floatsize, m_alpha[floattype]);
    ret = clSetKernelArg(kernel, 4, floatsize, m_beta[floattype]);
    ret = clSetKernelArg(kernel, 5, sizeof(unsigned), &gg.lda);
    ret = clSetKernelArg(kernel, 6, sizeof(unsigned), &gg.ldb);
    ret = clSetKernelArg(kernel, 7, sizeof(unsigned), &gg.ldc);
    ret = clSetKernelArg(kernel, 8, sizeof(unsigned), &gg.m);
    ret = clSetKernelArg(kernel, 9, sizeof(unsigned), &gg.n);
    ret = clSetKernelArg(kernel, 10, sizeof(unsigned), &gg.k);
    ret = clSetKernelArg(kernel, 11, sizeof(unsigned), &gg.a_offset);
    ret = clSetKernelArg(kernel, 12, sizeof(unsigned), &gg.b_offset);
    ret = clSetKernelArg(kernel, 13, sizeof(unsigned), &gg.c_offset);
  }

  void enqueue_betac_kernel(){
    ret = clEnqueueNDRangeKernel(command_queue, betac_kernel, 1, NULL, &betac_global_work_size, &betac_local_work_size, 0, NULL, &event_betac_kernel);
    if (ret != CL_SUCCESS){
      throw tinygemm_error("Error in clEnqueueNDRangeKernel (in the scaling kernel)");
    }
  }


  
  int enqueue_main_kernel(){
    
    if(does_betac_inc == 0){
      ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 1, &event_betac_kernel, &event_main_kernel);
    }
    else{
      ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0,NULL, &event_main_kernel);
    }

    /* Handle the failure case which is not CL_OUT_OF_RESOURCES */ 
    if (ret != CL_SUCCESS && ret != CL_OUT_OF_RESOURCES){
      std::string errm("Error in clEnqueueNDRangeKernel (in the main kernel), The CL_STATUS value is ( ");
      errm += std::to_string(ret);
      errm += " )";
      errm += "\n";
      throw tinygemm_error(errm);
    }
    
    /* Either returning CL_SUCCESS or CL_OUT_OF_RESOURCES.  */
    return ret;
  }


  void update_run_times(){
    clGetEventProfilingInfo(event_main_kernel, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start_main_kernel, nullptr);
    clGetEventProfilingInfo(event_main_kernel, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end_main_kernel, nullptr);
    t_just_main = 1e-6*(t_end_main_kernel-t_start_main_kernel);
    v_t_just_main.push_back(t_just_main);
    if (does_betac_inc == 0){
      clGetEventProfilingInfo(event_betac_kernel, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start_betac_kernel, nullptr);
      clGetEventProfilingInfo(event_betac_kernel, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end_betac_kernel, nullptr);
      t_total_with_both = 1e-6*(t_end_main_kernel-t_start_betac_kernel);
      t_just_scaling = 1e-6*(t_end_betac_kernel-t_start_betac_kernel);
      v_t_total_with_both.push_back(t_total_with_both);
      v_t_just_scaling.push_back(t_just_scaling);
    }
    else{
      v_t_total_with_both.push_back(t_just_main);
    }
  }
  
  /* Will be used when a kernel failed to be enqueued, so just set the time really big */
  void update_run_times_really_bad(){
    t_just_main = std::numeric_limits<float>::max();
    v_t_total_with_both.push_back(std::numeric_limits<float>::max());
    v_t_total_with_both.push_back(std::numeric_limits<float>::max());
  }

  void print_run_times(){
    if (does_betac_inc == 0){
      mowri << "elapsed time : " <<  t_total_with_both << "\t (scaling : " << t_just_scaling << "\t main : " <<  t_just_main << " [ms])  " << 
      "\tGflops/s : " << 2.0 * gg.m * gg.n * gg.k / (t_total_with_both * 1e6) << Endl;
    }
  
    else{
      mowri << "elapsed time : " <<  t_just_main << "    ";
      mowri << "Gflops/s : " << 2.0 * gg.m * gg.n * gg.k / (t_just_main * 1e6) << Endl;
    }
  }


  void reset_v_times(){
    v_t_total_with_both.resize(0);
    v_t_just_main.resize(0);
    v_t_just_scaling.resize(0);
  }
  
  void core_gemm_loop(size_t n_runs){
    
    //auto core_loop_start = std::chrono::high_resolution_clock::now();


    reset_v_times();
    for (size_t kqq = 0; kqq < n_runs; ++kqq){

      /* This pause should have zero effect, but mysteriously it smooths out the run times between runs when working with certain gpu drivers
       * It has something to do with overheating.  */        
      if (n_runs > 1){
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
      }
  
      /* ***************************************************************************************
       *  Note on timing : the same results have been obtained whth timespec and std::chrono.  *
       *  **************************************************************************************/

      /* Enqueue the beta c kernel if nec */
      if (does_betac_inc == 0){
        enqueue_betac_kernel();
      }


      /* At this point, the main kernel has been succesfully compiled, 
       * but it is still possible that the resources necessary (LDS etc) are
       * not sufficient on this machine. We catch this case here. */
      /* Attempt to enqueue the main kernel */      
      int status = enqueue_main_kernel();
      
      /* I'm not really sure when to use cl Flush TODO : find out. */
      clFlush(command_queue);
      
      if (status == CL_OUT_OF_RESOURCES){
        
        /* Set the run time(s) and append to vectors */
        mowri << "This kernel could not be enqueued. the status returned from clEnqueueNDRangeKernel was CL_OUT_OF_RESOURCES (=-5). So just setting time to be as large as possible, and moving along " << Endl;
        update_run_times_really_bad();      
        print_run_times();
      }
      
      else if (status == CL_SUCCESS){
        /* Wait for kernels to complete */
        clWaitForEvents(1, &event_main_kernel);

        /* Set the run time(s) and append to vectors */
        update_run_times();
        print_run_times();
      }
      
      else{
        throw std::logic_error("How can this not be CL_SUCCESS or CL_OUT_OF_RESOURCES? Algo prob, come fix");
      }
    }
    
    
    //auto core_loop_end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<float> fp_ms = core_loop_end - core_loop_start;
    //float elapsed_seconds = fp_ms.count();
    //mowri << "time in core_loop : " << elapsed_seconds*1000 << Endl;

  }


  void full_kernel_setup_from_source_file(std::string kernelfilename){

    /* set 5 parameters from given file: 
     * macro_tile_width, macro_tile_height, n_workitems_per_workgroup, n_work_items_per_c_elm, does_betac_inc */
    check_file_and_set_from(kernelfilename);

    /* Here we set the numbers of work groups (n_work_groups) and work items (global_work_size) for the alpha (alpha and beta) aka main kernel */  
    set_workforce();
    mowri << "main kernel global work size : " << global_work_size <<  " (recommended ~ 4*64*40*64 = 655360)" << Endl; 
    mowri << "Entering setting of program and kernel, compiling ..." << Flush;
    /* jit compile the cl source to get `kernel' (and `program') */
    openclutil::set_program_and_kernel(program, kernel, kernel_function_name, context, device_id_to_use, kernelfilename);
    mowri << "... done" << Endl;
    /* set kernel's arguments */
    set_main_kernel_arguments();
  }
  




  
        
  /* try-wrapped in gemini */
  void benchgemm(std::string kernelfilename, unsigned n_runs){

    if (n_runs == 0){
      throw tinygemm_error("n_runs to benchgemm should be a positive integer");
    }
    
    mowri << "INPUT_CALL   \t:" << gg.get_string() << Endl;
    full_kernel_setup_from_source_file(kernelfilename);
    mowri << "Entering the core gemm loops" << Endl;
    core_gemm_loop(n_runs);
    release_main_kernel_and_program();
    
  }
  
  
  
  
  
  tinygemm::TinyGemmSolution find(float allotted_time, bool enforce_deterministic, unsigned n_runs_in_find_per_kernel){  
    
    if (allotted_time > 0 && nobenching == true){
      throw tinygemm_error("allotted_time > 0 and nobencking == false, which does not make sense. Algo problem, come fix ");
    }
    
    else if (allotted_time <= 0 && nobenching == false){
      throw tinygemm_error("allotted_time <= 0 and nobenching == false. This makes no sense : algo problem, come fix ");
    }
    
    if (allotted_time <= 0){
      mowri << "Allotted time insufficient for benchmarking, returning default TinyGemmSolution" << Endl;
      return get_default(enforce_deterministic, floattype, gg);      
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fp_ms = end - start;
    float elapsed_seconds = fp_ms.count();

    /* we will count how many kernels are successfully generated AND compiled AND benchmarked */
    unsigned global_counter = 0;
        
    /* we track the best TinyGemmSolution found during the search  */    
    std::vector<tinygemm::TinyGemmSolution> path_of_best_solns;
    
    /* In here, we will store all previously considered HyperParams, used to check and ensure that we do not consider a HyperParam more than once */
    std::vector<hyperparams::HyperParams> hyper_front_history;
    
    /* we set some initial parameters: the ones which are not in HyperParam object but eventually needed by the python script to generate kernel */
    std::map<std::string, unsigned> all_int_parms = get_bare_bones_int_parms(gg);
    
    /* while generating, compiling and benchmarking kernels, we will keep track of the fastest found thus far */
    float best_time = std::numeric_limits<float>::max();
    hyperparams::HyperParams best_hyper_params;
    

    /* we initialise the `hyper-front' with a single HyperParams, selected based on problem dimensions (TODO : should be based on cache table look-up) */
    std::vector<hyperparams::HyperParams> hyper_front = { hyperparams::get_default(gg, enforce_deterministic) };    
    auto hyper_param_start = hyper_front[0];

    bool improvement_found_on_front = true;
    
    /* a hyper front consists of all kernels within a certain "distance of the current best. We start with a front
     * distance of 1, and when this gets into a local mimimum we switch to front distance 2. Front distance 2 kernels
     * are defined is terms of front distance 1 kernels : front distance 2 kernels are just the concatenatition of 
     * the distance 1 kernels from all distance 1 kernels */
    unsigned front_search_horizon = 1;
    
    mowri << "allotted time : " << allotted_time << Endl;
    while (improvement_found_on_front == true){
      improvement_found_on_front = false;
      mowri << "\nnew hyper front size : " << hyper_front.size() << Endl;

      unsigned hfi = 0;
      while (hfi < hyper_front.size() && improvement_found_on_front == false && elapsed_seconds < allotted_time){        
        hyperparams::HyperParams hp = hyper_front[hfi];
        /* certain kernels will not be generated, for diverse reasons */
        /* reason 0 : it's already been considered */
        if (std::find(hyper_front_history.begin(), hyper_front_history.end(), hp) != hyper_front_history.end()){
          /* this kernel has already been considered */
        }
        
        else{
          hyper_front_history.push_back(hp);
        
          /* reason 1 : the macro tile is too tall */
          if (gg.m < hp.params.at("macro_tile_height")){
            mowri << "m < macro_tile_height, not considering this kernel" << Endl;
          }
          
          /* reason 2 : the macro tile is too wide */
          else if (gg.n < hp.params.at("macro_tile_width")){
            mowri << "m < macro_tile_width, not considering this kernel" << Endl;
          }
          
          /* reason 3 : the user requests a deterministic kernel, which cannot be guaranteed */
          else if (enforce_deterministic == true && hp.params.at("n_work_items_per_c_elm") != 1){
            mowri << "not considering kernels which may be non-deterministic" << Endl;
          }
          /* ************************************************************************ */
  
          /* We will now attempt to generate the kernel */
          else {
            
            for (auto & x : hp.params){
              all_int_parms[x.first] = x.second;
            }
            /* attempt to generate the kernel. Certain `bad' kernels are only caught at this stage, with tests for hyper-parameter compatibilty in the 
             * python script which I don't want to recode here. The main compatibility issue caught here is that load sizes from global are multiples
             * of the number of work items. In the case of `stupid' mistakes (bad paths, run time problems) the python output from the first kernel 
             * generated (global_counter = 0) will be printed explicitly, so that hopefully the user will see what the problem is.  */
            bool verbose_report_from_python = global_counter == 0 ? true : false;
            int kernel_write_status = tinygemm::mkkern::make_kernel_via_python(kerneldir,  floattostring::get_float_string(floattype), all_int_parms, kernelname, verbose_report_from_python);
            
            /* the kernel was succesfully generated, we now compile and benchmark it */
            if (kernel_write_status == 0){

              mowri << "\n" << hp.get_string() << Endl;
              ++global_counter;
              mowri << "global gen-com-bench : " << global_counter  <<  "." << Endl;

              benchgemm(kernelfilename, n_runs_in_find_per_kernel);
              std::sort(v_t_total_with_both.begin(), v_t_total_with_both.end());

              /* Taking the fastest or median? */ 
              float median_time = v_t_total_with_both[v_t_total_with_both.size()/2]; //[0]; 
              
              if (std::abs(v_t_total_with_both.back() - median_time) / median_time > 0.2) {
                mowri << "tinygemm_warning: large variance in times. " <<  Endl;
              }
              
              end = std::chrono::high_resolution_clock::now();
              fp_ms = end - start;
              elapsed_seconds = fp_ms.count();
                              
              mowri << "median time  : " << median_time << "\t m-Gflops/s : " << 2.0 * gg.m * gg.n * gg.k / (median_time * 1e6) << Endl;
              mowri << "elapsed seconds : " << elapsed_seconds << Endl;
      
              /* A new best kernel !!! we're only interested in an improvement if it's 0.5% or more */
              if (median_time < 0.995*best_time){
                improvement_found_on_front = true;
                best_time = median_time;
                best_hyper_params = hp;
                mowri << "---------- NEW BEST TIME FOUND --------- : " << best_time << Endl << "breaking from current hyper front, creating new hyper front " << Endl;

                end = std::chrono::high_resolution_clock::now();
                fp_ms = end - start;
                elapsed_seconds = fp_ms.count();

                float median_benchmark_gflops = (2. * gg.m * gg.n * gg.k) / (median_time * 10e5);                
                tinygemm::TinyGemmSolutionStatistics tgss(median_time, median_benchmark_gflops, gg, elapsed_seconds);
                
                //set kernel files
                //TODO : should not be determining whether a kernel does betac or not based on this, find true parameter
                std::string soln_betac_kernel = all_int_parms.at("n_work_items_per_c_elm") == 1 ?  ""  : kernelutil::get_as_single_string(betac::get_cl_file_path(floattype));
                std::string soln_betac_kernel_function_name = all_int_parms.at("n_work_items_per_c_elm") == 1 ? "" : betac_kernel_function_name;
                std::string soln_main_kernel = kernelutil::get_as_single_string(kernelfilename);
                std::string soln_main_kernel_function_name = kernel_function_name;                
                tinygemm::TinyGemmSolution tgs(soln_betac_kernel, soln_main_kernel, soln_betac_kernel_function_name, soln_main_kernel_function_name, all_int_parms, floattype, tgss);

                path_of_best_solns.push_back(tgs); 
              }
            }
          
            else{
              mowri << "\nSkipping this kernel, hyper-parameters incompatible" << Endl;
            }
          }
        }
        
        ++hfi;
        
        end = std::chrono::high_resolution_clock::now();
        fp_ms = end - start;
        elapsed_seconds = fp_ms.count();
        
      }

      /* TODO: maybe. add another level of loop here. get_one_aways, then get_two_aways, etc. 
       * what we will have here is that `one' is just rough tile shape, important stuff.*/
      if (improvement_found_on_front == true && front_search_horizon == 1){        
        /* getting all `one-away's */
        hyper_front = best_hyper_params.get_one_aways(gg);
      }
      
      if (improvement_found_on_front == false && front_search_horizon == 1 && elapsed_seconds < allotted_time){
        ++front_search_horizon;

        /* TODO : if you WANT to go onto front 2, uncomment the following. This should be finalised TODO TODO TODO  */        
        const bool jump_to_front_horizon_size_2 = true;
        if (jump_to_front_horizon_size_2 == true){
          improvement_found_on_front = true;
          mowri << "\nSWITCHING TO FRONT HORIZON SIZE 2\n" << Endl;
        }
      }
      
      if (improvement_found_on_front == true && front_search_horizon == 2){        
        /* getting all `two-aways' */
        hyper_front = best_hyper_params.get_two_aways(gg);
      }
      
      if (improvement_found_on_front == false && front_search_horizon == 2){
        /* this is going to cause the end of the search */
      }
      
      if (front_search_horizon != 1 && front_search_horizon != 2){
        throw std::logic_error("front_search_horizon is neither 1 nor 2. This is currently not possible, Broken algorithm, come fix.");        
      }
    }
    
    if (allotted_time <= elapsed_seconds){
      mowri << "stopping the search because the allotted time has been surpassed" << Endl;
    }
    
    else{
      mowri << "stopping the search because a locally minimal kernel has been found" << Endl;
    }
    
    if (path_of_best_solns.size() == 0){
      throw tinygemm_error("\nUser should never see this error, this is an internal problem. Possibly, there were no solutions found. Which is strange, as at least the initial kernel (the initial hyper front) should have been a solution. Either, the initial kernel was not valid (which should not happen unless my filters are broken) or for whatever reason the kernel was not generated or did not compile. Maybe there is some preceding warning printed which sheds light on this? Other possibilities are bad dir_name or kernelname in make_kernel.py. set verbose_report to True in makekernelsource if you (james) can't figure this out ");
    }
  
    mowri << "best time : " << best_time << Endl;
    mowri << "best kernel : " << best_hyper_params.get_string() << Endl;
    mowri << "start kernel : " << hyper_param_start.get_string() << Endl;
    mowri << "the kernels along the path the final solution :  " << Endl; 
    mowri << "hyper parameter string                                     \t time when found\t median gflop/s" << Endl;

    for (auto & x : path_of_best_solns){
      mowri <<  x.get_hyper_param_string() << "\t " << x.statistics.solution_discovery_time << "\t\t " << x.statistics.median_benchmark_gflops  << Endl;
    }
     
    mowri <<  path_of_best_solns.back().get_hyper_param_string() << "\t " << elapsed_seconds << "\t\t " << path_of_best_solns.back().statistics.median_benchmark_gflops  << Endl;
    return path_of_best_solns.back();
  }
}; 

/* functions for the end-user */ 
tinygemm::TinyGemmSolution
nonconst_find(
float allotted_time,
cl_command_queue command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
const bool enforce_deterministic,
const char floattype, 
const tinygemm::TinyGemmGeometry & gg,
const double alpha,
const double beta,
bool verbose, 
std::string logfile){
  /* The number of times each kernel is run in find. 
   * consider adding this parameter to user API. */
  unsigned n_runs_in_find_per_kernel = 3;
  bool nobenching = allotted_time <= 0 ?  true : false ;  
  OpenCLGemmEncapsulator oger(command_queue, floattype, gg, alpha, beta, a, b, c, logfile, verbose, nobenching);
  return oger.find(allotted_time, enforce_deterministic, n_runs_in_find_per_kernel);
}

/* functions for the end-user */ 
tinygemm::TinyGemmSolution
find(
float allotted_time,
cl_command_queue command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
const bool enforce_deterministic,
const char floattype, 
const tinygemm::TinyGemmGeometry & gg,
const double alpha,
const double beta,
bool verbose, 
std::string logfile){
  
  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);
  
  cl_mem c_copied;
  cl_event c_copy_event; 
  size_t n_c = gg.ldc * (gg.tC == gg.isColMajor ? gg.m : gg.n) + gg.c_offset;
  size_t c_memsize = get_floatsize(floattype)*n_c;
  cl_int ret;
  
  cl_context context;
  ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
  if (ret != CL_SUCCESS){
    throw tinygemm_error("Problem using clGetCommandQueueInfo, trying to get CL_QUEUE_CONTEXT (in find, in openclgemmapi.cpp)");
  }
  
  c_copied = clCreateBuffer (context, CL_MEM_READ_WRITE, c_memsize, NULL, &ret);  
  if (ret != CL_SUCCESS) {
    std::string errm ("in tinygemm.cpp, find function. Attempting to create cl_mem c_copy with clCreateBuffer. Failed to do so, with exit status ");
    errm += std::to_string(ret);
    throw tinygemm_error(errm);
  }
  
  ret = clEnqueueCopyBuffer (command_queue, c, c_copied, 0, 0, c_memsize, 0, NULL, &c_copy_event);
  if (ret != CL_SUCCESS){
    std::string errm("in tinygemm.cpp, find function. Attempting to copy cl_mem c with clEnqueueCopyBuffer. Failed to do so, with exit status ");
    errm += std::to_string(ret);
    throw tinygemm_error(errm);
  }
  
  clWaitForEvents(1, &c_copy_event);
  auto soln =  nonconst_find(allotted_time, command_queue, a, b, c_copied, enforce_deterministic, floattype, gg, alpha, beta, verbose, logfile);
  ret = clReleaseMemObject(c_copied);
  if (ret != CL_SUCCESS){
    std::string errm("in tinygemm.cpp, find function. Attempting to release memory with clReleaseMemObject. Failed to do so, with exit status ");
    errm += std::to_string(ret);
    throw tinygemm_error(errm);
  }  
  return soln;
}


/* takes 0.05 seconds. 0.035 of this is spent in python script generating kernel and writing it, the rest is spent finding default. */
/* I can get through about 5 benchmarks per second, that is 1 every 0.2 seconds, so the python script is something to be concerned about... */
tinygemm::TinyGemmSolution
get_default(
const bool enforce_deterministic, 
const char floattype, 
const tinygemm::TinyGemmGeometry & gg
){
  std::map<std::string, unsigned> all_int_parms = get_bare_bones_int_parms(gg);
  hyperparams::HyperParams hp = hyperparams::get_default(gg, enforce_deterministic);
  for (auto & x : hp.params){
    all_int_parms[x.first] = x.second;
  }
  bool verbose_report_from_python = true;
  int kernel_write_status = tinygemm::mkkern::make_kernel_via_python(kerneldir,  floattostring::get_float_string(floattype), all_int_parms, kernelname, verbose_report_from_python);
  if (kernel_write_status != CL_SUCCESS){
    throw tinygemm_error("A problem arose in the python script to generate the kernel. Throwing from get_default.");
  }
  float median_time = std::numeric_limits<float>::max();
  float elapsed_seconds = 0.;
  float median_benchmark_gflops = 0.;                
  tinygemm::TinyGemmSolutionStatistics tgss(median_time, median_benchmark_gflops, gg, elapsed_seconds);
  std::string soln_betac_kernel = all_int_parms.at("n_work_items_per_c_elm") == 1 ?  ""  : kernelutil::get_as_single_string(betac::get_cl_file_path(floattype));
  std::string soln_betac_kernel_function_name = all_int_parms.at("n_work_items_per_c_elm") == 1 ? "" : kernelutil::get_kernel_function_name(betac::get_cl_file_path(floattype));
  std::string soln_main_kernel = kernelutil::get_as_single_string(kernelfilename);
  std::string soln_main_kernel_function_name = kernelutil::get_kernel_function_name(kernelfilename);                
  tinygemm::TinyGemmSolution tgs(soln_betac_kernel, soln_main_kernel, soln_betac_kernel_function_name, soln_main_kernel_function_name, all_int_parms, floattype, tgss);

  return tgs;
}




void benchgemm(
  cl_command_queue command_queue,
  std::string kernelfilename,
  unsigned n_runs,
  const char floattype, 
  const tinygemm::TinyGemmGeometry & gg,
  const double alpha,
  const double beta,
  cl_mem a_gpu,
  cl_mem b_gpu, 
  cl_mem c_gpu,
  bool verbose,
  std::string logfile){
  bool nobenching = false;
  OpenCLGemmEncapsulator oger(command_queue, floattype, gg, alpha, beta, a_gpu, b_gpu, c_gpu, logfile, verbose, nobenching);
  oger.benchgemm(kernelfilename, n_runs);
}
  
} //namespace




