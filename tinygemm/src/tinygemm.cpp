#include <thread>
#include <sstream>
#include <limits>
#include <chrono>
#include <limits>
#include <vector> 
#include <algorithm>

#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/consistencychecks.hpp>
#include <tinygemm/redirection.hpp>
#include <tinygemm/outputwriter.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/openclutil.hpp>
#include <tinygemm/betackernelutil.hpp>
#include <tinygemm/floattostring.hpp>
#include <tinygemm/tinygemmsolution.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/kernelstringgenerator.hpp>

namespace tinygemm{
  
static const std::string generickernelname = "atinygemmkernel";
static const std::string betackernelname = "heeltemal";


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



size_t get_floatbytes(char floattype){
  size_t floatbytes;
  if (floattype == 'f'){
    floatbytes = sizeof(float);
  }
  else if (floattype == 'd'){
    floatbytes = sizeof(double);
  }
  else{
    throw tinygemm_error("Unrecognised floattype char. Currently, only 'f' (single precision) and 'd' (double precision) are supported");
  }
  return floatbytes;
}


class TinyGemmKernel{
  
  public:
    cl_command_queue command_queue;
    std::string kernstr;
    std::string fname;
  
  private:
    cl_program clprog;

  public:
    cl_kernel clkern;
  
  private:
    std::string hash;

  public:
    
    TinyGemmKernel(cl_command_queue command_queue_, const std::string & hash_): command_queue(command_queue_), kernstr(""), fname(""), clprog(nullptr), clkern(nullptr), hash(hash_) {}
    
    void try_release(){
      if (clprog != nullptr){
        openclutil::cl_release_program(clprog, "TinyGemmKernel Destructor");
      }
      if (clkern != nullptr){
        openclutil::cl_release_kernel(clkern, "TinyGemmKernel Destructor");
      }
    }
    
    void update(const std::string & new_kernstr){
      try_release();
      kernstr = new_kernstr;      
      openclutil::set_program_and_kernel(command_queue, clprog, clkern, fname, kernstr);
    }
    
    ~TinyGemmKernel(){
      try_release();
    }
    
    bool is_set(){
      return (clprog != nullptr && clkern != nullptr);
    }

    
    void set_kernel_arg(cl_uint arg_index, size_t arg_size, const void *arg_value){
      
      if (clkern == nullptr){
        throw tinygemm_error("Attempt to set kernel argument of uninitialised kernel");
      }
      openclutil::cl_set_kernel_arg(clkern, arg_index, arg_size, arg_value, "in set_kernel_arg of TinyGemmKernel, " + hash + " index : " + std::to_string(arg_index));
    }
    
    void set_kernel_args(std::vector<std::pair<size_t, const void *> > arg_sizes_values){
      for (cl_uint arg_index = 0; arg_index < arg_sizes_values.size(); ++arg_index){
        set_kernel_arg(arg_index, arg_sizes_values[arg_index].first, arg_sizes_values[arg_index].second); 
      }
    }
      

};


class OpenCLGemmEncapsulator{
public: 
  /* references to constructor parameters */
  cl_command_queue command_queue;
  std::string outputfilename;
  const char floattype; 
  const tinygemm::TinyGemmGeometry gg;
  cl_mem a_gpu;
  cl_mem b_gpu; 
  cl_mem c_gpu;


private:

  size_t floatbytes;
  MultiFloatType m_alpha;
  MultiFloatType m_beta;
  outputwriting::OutputWriter mowri;
   
  /* parameters related to the main kernel */
  unsigned does_betac_inc;
  size_t main_n_work_groups;
  size_t main_local_work_size;
  size_t main_global_work_size;
  
  /* parameters related to the scaling (betac) kernel */
  unsigned dim_coal;
  unsigned dim_uncoal;
  size_t betac_global_work_size;
  size_t betac_local_work_size;

  /* stores (the most recent of n_runs) execution time */
  size_t t_start_betac_kernel, t_end_betac_kernel, t_start_main_kernel, t_end_main_kernel;

  /* vector of times over a set of runs on core loop */
  std::vector<float> v_t_total_with_both, v_t_just_scaling, v_t_just_main;

  /* used for getting performance of main and (possibly) betac kernels */
  cl_event event_main_kernel;
  cl_event event_betac_kernel;


  
  TinyGemmKernel tk_main;
  TinyGemmKernel tk_betac;
  

  

public:
  OpenCLGemmEncapsulator(
  cl_command_queue command_queue_, 
  const char floattype_,
  const tinygemm::TinyGemmGeometry gg_,  
  const double alpha_,
  const double beta_,  
  cl_mem a_gpu_,
  cl_mem b_gpu_, 
  cl_mem c_gpu_,
  std::string outputfilename_,
  bool verbose_):

  command_queue(command_queue_), 
  outputfilename(outputfilename_),
  floattype(floattype_), 
  gg(gg_),
  a_gpu(a_gpu_),
  b_gpu(b_gpu_), 
  c_gpu(c_gpu_),
  m_alpha(alpha_),
  m_beta(beta_),
  mowri(verbose_, outputfilename.compare("") != 0, outputfilename_),
  
  tk_main(command_queue, "tk_main"),
  tk_betac(command_queue, "tk_betac"){
    
    
    floatbytes = get_floatbytes(floattype);
    run_checks();
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

  
  void set_betac_kernel_arguments(){
    tk_betac.set_kernel_args( {
      {sizeof(unsigned),    &dim_coal},
      {sizeof(unsigned),    &dim_uncoal},
      {sizeof(unsigned),    &gg.ldc},
      {sizeof(unsigned),    &gg.c_offset},
      {sizeof(cl_mem),      (void *)&c_gpu},
      {floatbytes,           m_beta[floattype]}    
    } );              
  }
  
  void setup_betac_kernel(){
    tk_betac.update(betac::get_betac_kernel_string(floattype, betackernelname));
    betac::set_betackernel_sizes(floattype, gg.isColMajor, gg.tC, gg.m, gg.n, dim_coal, dim_uncoal, betac_global_work_size, betac_local_work_size);
    mowri << "in setup_betac_kernel, betac_global_work_size : " << betac_global_work_size << Endl; 
    set_betac_kernel_arguments();

  }
  
  void enqueue_betac_kernel(){
    openclutil::cl_enqueue_ndrange_kernel(command_queue, tk_betac.clkern, 1, NULL, &betac_global_work_size, &betac_local_work_size, 0, NULL, &event_betac_kernel, "in enqueue_betac_kernel");
  }






  void set_main_kernel_arguments(){
    /* set the main kernel parameters */
    tk_main.set_kernel_args( {
      {sizeof(cl_mem),      (void *)&c_gpu},
      {sizeof(cl_mem),      (void *)&a_gpu},
      {sizeof(cl_mem),      (void *)&b_gpu},
      {floatbytes,           m_alpha[floattype]},
      {floatbytes,           m_beta[floattype]},
      {sizeof(unsigned),    &gg.lda},
      {sizeof(unsigned),    &gg.ldb},
      {sizeof(unsigned),    &gg.ldc},
      {sizeof(unsigned),    &gg.m},
      {sizeof(unsigned),    &gg.n},
      {sizeof(unsigned),    &gg.k},
      {sizeof(unsigned),    &gg.a_offset},
      {sizeof(unsigned),    &gg.b_offset},
      {sizeof(unsigned),  &gg.c_offset} 
    } ); 

  }

  
  

  void setup_main_kernel(const std::string & kernel_string, const hyperparams::HyperParams & hp, const derivedparams::DerivedParams & dp){
    
     /* Here we set the numbers of work groups (main_n_work_groups) and work items (main_global_work_size) for the main kernel */  
    sizingup::set_workforce(main_n_work_groups, main_local_work_size, main_global_work_size, gg.m, gg.n, hp.n_work_items_per_c_elm, hp.macro_tile_height, hp.macro_tile_width, dp.n_workitems_per_workgroup);
    
    mowri << "main kernel global work size : " << main_global_work_size <<  " (recommended ~ 4*64*40*64 = 655360)" << Endl; 
    tk_main.update(kernel_string);
    /* set main kernel's arguments */
    set_main_kernel_arguments();    
  }


  //int enqueue(bool throw_on_oor, const size_t * global_work_size, const size_t * local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list){
    //cl_int ret = clEnqueueNDRangeKernel(command_queue, clkern, 1, NULL, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    //if (if throw_on_oor == true || ret != CL_OUT_OF_RESOURCES){
      //openclutil::confirm_cl_status(ret, "in enqueue:  " + hash);
    //}
    //return ret;
  //}

  
  
  int enqueue_main_kernel(){
    cl_int ret;
    if(does_betac_inc == 0){
      ret = clEnqueueNDRangeKernel(command_queue, tk_main.clkern, 1, NULL, &main_global_work_size, &main_local_work_size, 1, &event_betac_kernel, &event_main_kernel);
    }
    else{
      ret = clEnqueueNDRangeKernel(command_queue, tk_main.clkern, 1, NULL, &main_global_work_size, &main_local_work_size, 0, NULL, &event_main_kernel);
    }
    if (ret != CL_OUT_OF_RESOURCES){
      openclutil::confirm_cl_status(ret, "in enqueue_main_kernel");
    }
    /* Either returning CL_SUCCESS or CL_OUT_OF_RESOURCES, anything has been thrown */
    return ret;
  }
  
  
  
  
  void setup_tinykernels(const std::string & kernstr, const hyperparams::HyperParams & hp, const derivedparams::DerivedParams & dp){
    
    does_betac_inc = dp.does_beta_c_inc;
    
    setup_main_kernel(kernstr, hp, dp);    
    if (tk_betac.is_set() == false && does_betac_inc == false){
      setup_betac_kernel();
    }
  }
  


  void update_run_times(){
    
    clGetEventProfilingInfo(event_main_kernel, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start_main_kernel, nullptr);
    clGetEventProfilingInfo(event_main_kernel, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end_main_kernel, nullptr);
    float t_just_main = 1e-6*(t_end_main_kernel-t_start_main_kernel);
    v_t_just_main.push_back(t_just_main);
    
    
    if (does_betac_inc != 0){
      v_t_total_with_both.push_back(t_just_main);
      v_t_just_scaling.push_back(0);
    }
      
    else{
      clGetEventProfilingInfo(event_betac_kernel, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start_betac_kernel, nullptr);
      clGetEventProfilingInfo(event_betac_kernel, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end_betac_kernel, nullptr);
      v_t_total_with_both.push_back(1e-6*(t_end_main_kernel-t_start_betac_kernel));
      v_t_just_scaling.push_back(1e-6*(t_end_betac_kernel-t_start_betac_kernel));
    }
    

  }
  
  /* Will be used when a kernel failed to be enqueued, so just set the time really big */
  void update_run_times_really_bad(){
    v_t_just_main.push_back(std::numeric_limits<float>::max());
    v_t_total_with_both.push_back(std::numeric_limits<float>::max());
    v_t_just_scaling.push_back(std::numeric_limits<float>::max());
  }

  void print_run_times(){
    if (does_betac_inc == 0){
      mowri << "elapsed time : " <<  v_t_total_with_both.back() << "\t (scaling : " << v_t_just_scaling.back() << "\t main : " <<  v_t_just_main.back() << " [ms])  " << 
      "\tGflops/s : " << 2.0 * gg.m * gg.n * gg.k / (v_t_total_with_both.back() * 1e6) << Endl;
    }
  
    else{
      mowri << "elapsed time : " <<  v_t_just_main.back() << "    ";
      mowri << "Gflops/s : " << 2.0 * gg.m * gg.n * gg.k / (v_t_just_main.back() * 1e6) << Endl;
    }
  }


  void reset_v_times(){
    v_t_total_with_both.resize(0);
    v_t_just_main.resize(0);
    v_t_just_scaling.resize(0);
  }
  
  void core_gemm_loop(size_t n_runs){
    
    reset_v_times();
    
    for (size_t kqq = 0; kqq < n_runs; ++kqq){

      /* This pause should have zero effect, but mysteriously it smooths out the run times between runs when working with certain gpu drivers
       * something to do with overheating  */        
      if (n_runs > 1){
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
      }
  
      /* ***************************************************************************************
       *  Note on timing : the same results have been obtained whth timespec and std::chrono   *
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
      openclutil::cl_flush(command_queue, "cl flushing in core gemm loop");
      
      if (status == CL_OUT_OF_RESOURCES){
        /* Set the run time(s) and append to vectors */
        mowri << "kernel could not be enqueued, status returned from clEnqueueNDRangeKernel was CL_OUT_OF_RESOURCES, setting time to be x-large and moving on " << Endl;
        update_run_times_really_bad();
        print_run_times();
      }
      
      else if (status == CL_SUCCESS){
        /* Wait for kernels to complete */
        openclutil::cl_wait_for_events(1, &event_main_kernel, "with status == CL_SUCCESS in core gemm loops");

        /* Set the run time(s) and append to vectors */
        update_run_times();
        print_run_times();
      }
      
      else{
        throw std::logic_error("How can this not be CL_SUCCESS or CL_OUT_OF_RESOURCES? Algo prob, come fix");
      }
    }
  }



  tinygemm::kerngen::KernelStringBundle get_ksb(const hyperparams::HyperParams & hp){ //, std::string & kernel_string){
    
    //tinygemm::kerngen::KernelStringBundle bundle =
    
    return  tinygemm::kerngen::get_kernel_string_bundle(
    hp,
    //kernel_string,
    generickernelname,
    floattype ==  'f' ? 32 : 64,
    gg.tA,
    gg.tB,
    gg.tC,
    gg.isColMajor);
    
    //kernel_string = std::move(bundle.kernel_string)
    
    //return set_status;
    
  }

  void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs){


    if (n_runs == 0){
      throw tinygemm_error("n_runs to benchgemm should be a positive integer");
    }

    //unsigned n_kernels_processed = 0;
    for ( unsigned i = 0; i < hps.size(); ++i){ //auto & hp : hps){
      
      mowri << "\nSource kernel " << "(" << i + 1 << "/" << hps.size() << ") "  << Endl;      
            

      auto bundle = get_ksb(hps[i]);
      
      if (bundle.set_status.is_good() != true){
        throw tinygemm_error("the hyper parameters in benchgemm are not consistent, specifically : \n" + bundle.set_status.message);
      }
      
  
      setup_tinykernels(bundle.kernel_string, hps[i], bundle.dp);    
      mowri << "(benchgemm) geometry  \t:" << gg.get_string()  << "\nEntering the core gemm loops" << Endl;
      core_gemm_loop(n_runs);
    }
  }
  
  
  /* takes 0.05 seconds. 0.035 of this is spent in python script generating kernel and writing it, the rest is spent finding default. */
  /* I can get through about 5 benchmarks per second, that is 1 every 0.2 seconds, so the python script is something to be concerned about... */
  tinygemm::TinyGemmSolution
  get_default(
  const bool enforce_deterministic
//  const char floattype, 
//  const tinygemm::TinyGemmGeometry & gg
  ){
    
    hyperparams::HyperParams hp = hyperparams::get_default(gg, enforce_deterministic);
    
    std::string soln_main_kernel_string;
    
    auto bundle = get_ksb(hp);//, soln_main_kernel_string);
    if (bundle.set_status.is_good() != true){
      throw tinygemm_error("the hyper parameters in get_default are not consistent, specifically : \n" + bundle.set_status.message);
    }
    
    
    tinygemm::TinyGemmSolutionStatistics tgss(std::numeric_limits<float>::max(), 0, 0);    
    std::string soln_betac_kernel_string = hp.n_work_items_per_c_elm == 1 ?  ""  : betac::get_betac_kernel_string(floattype, betackernelname);
    std::string soln_betac_kernel_function_name = hp.n_work_items_per_c_elm == 1 ? "" : betackernelname; //kernelutil::get_kernel_function_name(soln_betac_kernel_string);
    std::string soln_main_kernel_function_name = bundle.kernel_function_name;
    
    
    
    return { soln_betac_kernel_string, soln_betac_kernel_function_name, bundle.kernel_string, soln_main_kernel_function_name, hp, bundle.dp, gg, floattype, tgss };


  }
  
  
  tinygemm::TinyGemmSolution find(float allotted_time, bool enforce_deterministic, unsigned n_runs_per_kernel){
    
    
    
    if (gg.m < 8 || gg.n < 8){
      mowri << "really skinny/thin matrix, returning a default kernel (to be improved) " << Endl;
      return get_default(enforce_deterministic);//, gg); //floattype, gg);
    }
      
    
    if (allotted_time <= 0){
      mowri << "Allotted time insufficient for benchmarking, returning default TinyGemmSolution" << Endl;
      return get_default(enforce_deterministic);//, gg);//floattype, gg);      
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
        
    /* while generating, compiling and benchmarking kernels, we will keep track of the fastest found thus far */
    float best_time = std::numeric_limits<float>::max();
    
    hyperparams::HyperParams best_hp = hyperparams::get_default(gg, enforce_deterministic);
    
    
    



    /* we initialise the `hyper-front' with a single HyperParams, selected based on problem dimensions  */
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
          if (gg.m < hp.macro_tile_height){
            mowri << "m < macro_tile_height, not considering this kernel" << Endl;
          }
          
          /* reason 2 : the macro tile is too wide */
          else if (gg.n < hp.macro_tile_width){
            mowri << "m < macro_tile_width, not considering this kernel" << Endl;
          }
          
          /* reason 3 : the user requests a deterministic kernel, which cannot be guaranteed */
          else if (enforce_deterministic == true && hp.n_work_items_per_c_elm != 1){
            mowri << "not considering kernels which may be non-deterministic" << Endl;
          }
          /* ************************************************************************ */
  
          /* We will now attempt to generate the kernel */
          else {

            /* attempt to generate the kernel. Certain `bad' kernels are only caught at this stage (set_status.is_good() == false)
             * with tests for hyper-parameter compatibilty in the python script which I don't want to recode here. The main compatibility 
             * issue caught here is that load sizes from global are multiples of the number of work items.  */
            
            mowri << "\n" << hp.get_string() << Endl;
            
            
            auto bundle = get_ksb(hp);
            tk_main.kernstr = std::move(bundle.kernel_string);
            
            if (bundle.set_status.is_good() == true){
              
                            
              /* the kernel was succesfully generated, we now compile and benchmark it */
              
              ++global_counter;
              mowri << "global gen-com-bench : " << global_counter  <<  "." << Endl;
              
              setup_tinykernels(tk_main.kernstr, hp, bundle.dp);    
              mowri << "(find) geometry  \t:" << gg.get_string()  << "\nEntering the core gemm loops" << Endl;
              core_gemm_loop(n_runs_per_kernel);
  
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
                best_hp = hp;
                mowri << "---------- NEW BEST TIME FOUND --------- : " << best_time << Endl << "breaking from current hyper front, creating new hyper front " << Endl;
  
                end = std::chrono::high_resolution_clock::now();
                fp_ms = end - start;
                elapsed_seconds = fp_ms.count();
  
                float median_benchmark_gflops = (2. * gg.m * gg.n * gg.k) / (median_time * 10e5);                
                
                tinygemm::TinyGemmSolutionStatistics tgss(median_time, median_benchmark_gflops, elapsed_seconds);
                
                
                /* set kernel files */
                std::string soln_betac_kernel = does_betac_inc == 1 ?  ""  : betac::get_betac_kernel_string(floattype, betackernelname);
                std::string soln_betac_kernel_function_name = does_betac_inc == 1 ? "" : tk_betac.fname;
                std::string soln_main_kernel_function_name = bundle.kernel_function_name;                 
                                
                path_of_best_solns.emplace_back (soln_betac_kernel, soln_betac_kernel_function_name, tk_main.kernstr, soln_main_kernel_function_name, hp, bundle.dp, gg, floattype, tgss); 

                
              }
           
            }
          
            else{
              mowri << "\nSkipping this kernel, hyper-parameters incompatible. " << Endl;
              mowri << "Specifically, the message from the kernel string setting function was --- " << bundle.set_status.message << "\n";
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
        hyper_front = best_hp.get_one_aways(gg);
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
        hyper_front = best_hp.get_two_aways(gg);
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
      throw tinygemm_error("\nUser should never see this error, this is an internal problem. Possibly, there were no solutions found. Which is strange, as at least the initial kernel (the initial hyper front) should have been a solution. Either, the initial kernel was not valid (which should not happen unless my filters are broken) or for whatever reason the kernel was not generated or did not compile. Maybe there is some preceding warning printed which sheds light on this? Another possibility is that there was an error in the kernel string generation which I did not think of. ");
    }
  
    mowri << "best time : " << best_time << Endl;
    mowri << "best kernel : " << best_hp.get_string() << Endl;
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

openclutil::SafeClMem get_copy(
cl_command_queue command_queue,
cl_mem c,   
const char floattype, 
const tinygemm::TinyGemmGeometry & gg,
const std::string & hash
){
  openclutil::SafeClMem c_copied(hash);
  cl_event c_copy_event; 
  size_t n_c = gg.ldc * (gg.tC == gg.isColMajor ? gg.m : gg.n) + gg.c_offset;
  size_t c_memsize = get_floatbytes(floattype)*n_c;
  c_copied.clmem = openclutil::cl_create_buffer_from_command_queue(command_queue, CL_MEM_READ_WRITE, c_memsize, NULL, hash + ", in function get_copy of tinygemm");
  openclutil::cl_enqueue_copy_buffer(command_queue, c, c_copied.clmem, 0, 0, c_memsize, 0, NULL, &c_copy_event, hash + ", in function get_copy of tinygemm");
  openclutil::cl_wait_for_events(1, &c_copy_event, "in function find of tinygemm");
  return c_copied;
}


/////* functions for the end-user */ 
////tinygemm::TinyGemmSolution
////nonconst_find(
////float allotted_time,
////cl_command_queue command_queue,
////cl_mem a,   
////cl_mem b,
////cl_mem c,
////const bool enforce_deterministic,
////const char floattype, 
////const tinygemm::TinyGemmGeometry & gg,
////const double alpha,
////const double beta,
////bool verbose, 
////std::string logfile){

  
////}




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
std::string logfile, 
bool c_is_const){
  
  
  /* The number of times each kernel is run in find. 
   * consider adding this parameter to user API. */
  unsigned n_runs_per_kernel = 3;

  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);  

  if (c_is_const == true){
  
    openclutil::SafeClMem c_copied = get_copy(command_queue, c, floattype, gg, "copy of c in find");
    OpenCLGemmEncapsulator oger(command_queue, floattype, gg, alpha, beta, a, b, c_copied.clmem, logfile, verbose);
    return oger.find(allotted_time, enforce_deterministic, n_runs_per_kernel);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, floattype, gg, alpha, beta, a, b, c, logfile, verbose);
    return oger.find(allotted_time, enforce_deterministic, n_runs_per_kernel);
  }
  
  
  
  
  
}




void benchgemm(
  cl_command_queue command_queue,
  const std::vector<hyperparams::HyperParams> & hps,
  unsigned n_runs,
  const char floattype, 
  const tinygemm::TinyGemmGeometry & gg,
  const double alpha,
  const double beta,
  cl_mem a_gpu,
  cl_mem b_gpu, 
  cl_mem c_gpu,
  bool verbose,
  std::string logfile,
  bool c_is_const){
    
  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);
  if (c_is_const == true){
    openclutil::SafeClMem c_copied = get_copy(command_queue, c_gpu, floattype, gg, "copy of c in benchgemm");
    OpenCLGemmEncapsulator oger(command_queue, floattype, gg, alpha, beta, a_gpu, b_gpu, c_copied.clmem, logfile, verbose);
    oger.benchgemm(hps, n_runs);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, floattype, gg, alpha, beta, a_gpu, b_gpu, c_gpu, logfile, verbose);
    oger.benchgemm(hps, n_runs);
  }
}
  
} //namespace





