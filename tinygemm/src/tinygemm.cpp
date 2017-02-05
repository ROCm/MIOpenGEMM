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
#include <tinygemm/tinygemmkernel.hpp>

namespace tinygemm{
  

static const std::string betackernelname = "tg_betac";



class MultiFloatType{

  private:
    double v_d;
    float v_f;  
    
  public:
    MultiFloatType(double v): v_d(v), v_f(static_cast<float>(v)) {}
    void * operator [] (char floattype) const{
      return floattype == 'd' ? (void *)(& v_d) : (void * )(& v_f);
    }
};



static const MultiFloatType m_alpha(default_alpha);
static const MultiFloatType m_beta(default_beta);
  

class OpenCLGemmEncapsulator{
public: 
  /* references to constructor parameters */
  cl_command_queue command_queue;
  std::string outputfilename;

  const tinygemm::TinyGemmGeometry gg;
  const tinygemm::TinyGemmOffsets toff;
  
  cl_mem a_gpu;
  cl_mem b_gpu; 
  cl_mem c_gpu;
  cl_mem workspace_gpu;


private:

  outputwriting::OutputWriter mowri;
   
  /* parameters related to the main kernel */
  unsigned does_betac_inc;
  size_t main_n_work_groups;
  size_t main_local_work_size;
  size_t main_global_work_size;
  
  /* parameters related to the scaling (betac) kernel */
  size_t betac_global_work_size;
  
  /* stores (the most recent of n_runs) execution time */
  size_t t_start_betac_kernel, t_end_betac_kernel, t_start_main_kernel, t_end_main_kernel;

  /* vector of times over a set of runs on core loop */
  std::vector<float> v_t_total_with_both, v_t_just_scaling, v_t_just_main;

  /* used for getting performance of main and (possibly) betac kernels */
  cl_event event_main_kernel;
  cl_event event_betac_kernel;
  
  TinyGemmKernel tk_main;
  TinyGemmKernel tk_betac;
  

private:


  void address_check_valid(){
  
    if (c_gpu == a_gpu || c_gpu == b_gpu){
      throw tinygemm_error("c should be distinct from a and b for gemm, otherwise race condition arises (one thread writes its result to c before another one has finished reading from c)");
    }
    
    if (c_gpu == nullptr){
      throw tinygemm_error("c should not be nullptr");
    }
    
    if (workspace_gpu == nullptr && gg.workspace_size != 0){
      throw tinygemm_error("pointer to workspace memory is the nullptr, but workspace_size is not zero");
    }
    
    if (workspace_gpu != nullptr && gg.workspace_size == 0){
      throw tinygemm_error("pointer to workspace memory is not the nullptr, but workspace_size is zero. if workspace_size is zero please set workspace_gpu to the nullptr to make super clear that there will be no workspace used ");
      
    }
    
    if (workspace_gpu != nullptr && (workspace_gpu == a_gpu || workspace_gpu == b_gpu || workspace_gpu == c_gpu ) ){
      throw tinygemm_error("pointer to workspace memory is not the nullptr, and it is the same as one of the a,b,c pointers ");
    }
  }
  
  
  void address_check_valid_and_reliable(){
    
    address_check_valid();
    
    if (a_gpu == b_gpu){
      throw tinygemm_error( "a and b are the same. this will effect kernel run time, not sure if this should be allowed so throwing"); 
    }
  }


public:
  OpenCLGemmEncapsulator(
  cl_command_queue command_queue_, 

  const tinygemm::TinyGemmGeometry gg_,
  const tinygemm::TinyGemmOffsets toff_,

  cl_mem a_gpu_,
  cl_mem b_gpu_, 
  cl_mem c_gpu_,
  cl_mem workspace_gpu_,
  std::string outputfilename_,
  bool verbose_):

  command_queue(command_queue_), 
  outputfilename(outputfilename_),
  //floattype(floattype_), 
  gg(gg_),
  toff(toff_),
  a_gpu(a_gpu_),
  b_gpu(b_gpu_), 
  c_gpu(c_gpu_),
  workspace_gpu(workspace_gpu_),
  
  mowri(verbose_, outputfilename.compare("") != 0, outputfilename_),
  
  tk_main(command_queue, "tk_main"),
  tk_betac(command_queue, "tk_betac"){
    
    run_checks();
  }
  

  void run_checks(){
        
    /* Confirm that the input parameters make sense (done in constructor? TODO confirm) */
    //consistencychecks::check_ldx_mnk_consistent(gg);
    
    sizingup::check_sizes_ok_for_unsigned(gg, toff);
  
  }  

  
  void set_betac_kernel_arguments(){
    tk_betac.set_kernel_args( {
      {sizeof(unsigned),                      &gg.derived.dim_c_coal},
      {sizeof(unsigned),                      &gg.derived.dim_c_uncoal},
      {sizeof(unsigned),                      &gg.ldc},
      {sizeof(unsigned),                      &toff.oc},
      {sizeof(cl_mem),                        (void *)&c_gpu},
      {gg.derived.float_size_bytes,           m_beta[gg.floattype]}    
    } );              
  }
  
  void setup_betac_kernel(){
    tk_betac.update(betac::get_betac_kernel_string(gg.floattype, betackernelname), betackernelname);
    betac_global_work_size = betac::get_global_work_size(gg);
    mowri << "in setup_betac_kernel, betac_global_work_size : " << betac_global_work_size << Endl; 
    set_betac_kernel_arguments();
  }
  
  void enqueue_betac_kernel(){
    openclutil::cl_enqueue_ndrange_kernel(command_queue, tk_betac.clkern, 1, NULL, &betac_global_work_size, &betac::n_work_items_per_group, 0, NULL, &event_betac_kernel, "in enqueue_betac_kernel");
  }






  void set_main_kernel_arguments(){
    /* set the main kernel parameters */
    tk_main.set_kernel_args( {
      {sizeof(cl_mem),                        (void *)&c_gpu},
      {sizeof(cl_mem),                        (void *)&a_gpu},
      {sizeof(cl_mem),                        (void *)&b_gpu},
      {gg.derived.float_size_bytes,           m_alpha[gg.floattype]},
      {gg.derived.float_size_bytes,           m_beta[gg.floattype]},
      {sizeof(unsigned),                      &toff.oa},
      {sizeof(unsigned),                      &toff.ob},
      {sizeof(unsigned),                      &toff.oc} 
    } ); 

  }

  
  

  void setup_main_kernel(std::string && kernel_string, const hyperparams::HyperParams & hp, const derivedparams::DerivedParams & dp, const std::string & kern_func_name){
    
     /* Here we set the numbers of work groups (main_n_work_groups) and work items (main_global_work_size) for the main kernel */  
    sizingup::set_workforce(main_n_work_groups, main_local_work_size, main_global_work_size, gg.m, gg.n, hp.n_work_items_per_c_elm, hp.macro_tile_height, hp.macro_tile_width, dp.n_work_items_per_workgroup);
    
    mowri << "main kernel global work size : " << main_global_work_size <<  " (recommended ~ 4*64*40*64 = 655360)" << Endl; 
    tk_main.update(std::move(kernel_string), kern_func_name);
    /* set main kernel's arguments */
    set_main_kernel_arguments();    
  }

  
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
  
  
  
  
  void setup_tinykernels(std::string && kernstr, const hyperparams::HyperParams & hp, const derivedparams::DerivedParams & dp, const std::string & kern_func_name){
    
    does_betac_inc = dp.does_beta_c_inc;
    
    setup_main_kernel(std::move(kernstr), hp, dp, kern_func_name);    
    if (tk_betac.is_set() == false && does_betac_inc == false){
      setup_betac_kernel();
    }
  }
  


  void update_run_times(){
    
    //TODO : wrap clGetEventProfilingInfo in safety layer
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



  tinygemm::kerngen::KernelStringBundle get_ksb(const hyperparams::HyperParams & hp){
    
    return  tinygemm::kerngen::get_kernel_string_bundle(
    hp,gg);
    
    
  }

  void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs){

    address_check_valid();

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
      

  
      setup_tinykernels(std::move(bundle.kernel_string), hps[i], bundle.dp, bundle.kernel_function_name);    
      mowri << "(benchgemm) geometry  \t:" << gg.get_string()  << "\nEntering the core gemm loops" << Endl;
      core_gemm_loop(n_runs);
    }
  }
  
  
  tinygemm::TinyGemmSolution
  get_default(const bool enforce_deterministic){
    
    hyperparams::HyperParams hp = hyperparams::get_default(gg, enforce_deterministic);
    
    
    auto bundle = get_ksb(hp);//, soln_main_kernel_string);
    if (bundle.set_status.is_good() != true){
      throw tinygemm_error("the hyper parameters in get_default are not consistent, specifically : \n" + bundle.set_status.message);
    }
    
    tinygemm::TinyGemmSolutionStatistics tgss(std::numeric_limits<float>::max(), 0, 0);    
    
    std::string soln_betac_kernel_string = hp.n_work_items_per_c_elm == 1 ?  ""  : betac::get_betac_kernel_string(gg.floattype, betackernelname);
    std::string soln_betac_kernel_function_name = hp.n_work_items_per_c_elm == 1 ? "" : betackernelname;
    
    return { soln_betac_kernel_string, soln_betac_kernel_function_name, bundle.kernel_string, bundle.kernel_function_name, hp, bundle.dp, gg, tgss };
  }
  
  
  tinygemm::TinyGemmSolution find(float allotted_time, bool enforce_deterministic, unsigned n_runs_per_kernel){
    
    

    
    if (gg.m < 8 || gg.n < 8){
      mowri << "really skinny/thin matrix, returning a default kernel (to be improved) " << Endl;
      return get_default(enforce_deterministic);
    }
      
    
    if (allotted_time <= 0){
      mowri << "Allotted time insufficient for benchmarking, returning default TinyGemmSolution" << Endl;
      return get_default(enforce_deterministic);      
    }

    address_check_valid_and_reliable();
        
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
            //tk_main.tgk_strings.kernstr = std::move(bundle.kernel_string);
            
            if (bundle.set_status.is_good() == true){
              
                            
              /* the kernel was succesfully generated, we now compile and benchmark it */
              
              ++global_counter;
              mowri << "global gen-com-bench : " << global_counter  <<  "." << Endl;
              
              setup_tinykernels(std::move(bundle.kernel_string), hp, bundle.dp, bundle.kernel_function_name);    
              mowri << "(find) geometry : " << gg.get_string()  << "\nEntering the core gemm loops" << Endl; 
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
                std::string soln_betac_kernel = does_betac_inc == 1 ?  ""  : tk_betac.tgk_strings.kernstr;
                std::string soln_betac_kernel_function_name = does_betac_inc == 1 ? "" : tk_betac.tgk_strings.fname;
                std::string soln_main_kernel_function_name = bundle.kernel_function_name;                 
                
                
                path_of_best_solns.emplace_back (soln_betac_kernel, soln_betac_kernel_function_name, tk_main.tgk_strings.kernstr, soln_main_kernel_function_name, hp, bundle.dp, gg, tgss); 

                
              }
           
            }
          
            else{
              mowri << "Skipping this kernel, hyper-parameters incompatible. " << Endl;
              mowri << "Specifically, the message from the kernel string setting function was \n`````\n" << bundle.set_status.message << "'''''\n";
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
  
    //mowri << "best time : " << best_time << Endl;
    mowri << "\nstart kernel : " << hyper_param_start.get_string() << Endl;
    mowri << "best kernel  : " << best_hp.get_string() << Endl;

    mowri << "the kernels along the path the final solution:  " << Endl; 
    mowri << "hyper parameter string                                          \t time when found\t median gflop/s" << Endl;

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
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
const std::string & hash
){
  openclutil::SafeClMem c_copied(hash);
  cl_event c_copy_event; 
  size_t n_c = gg.ldc * (gg.tC == gg.isColMajor ? gg.m : gg.n) + toff.oc;
  size_t c_memsize = gg.derived.float_size_bytes*n_c;////get_floatbytes(floattype)*n_c;
  c_copied.clmem = openclutil::cl_create_buffer_from_command_queue(command_queue, CL_MEM_READ_WRITE, c_memsize, NULL, hash + ", in function get_copy of tinygemm");
  openclutil::cl_enqueue_copy_buffer(command_queue, c, c_copied.clmem, 0, 0, c_memsize, 0, NULL, &c_copy_event, hash + ", in function get_copy of tinygemm");
  openclutil::cl_wait_for_events(1, &c_copy_event, "in function find of tinygemm");
  return c_copied;
}







tinygemm::TinyGemmSolution
find(
float allotted_time,
cl_command_queue command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
cl_mem workspace,
const bool enforce_deterministic,
//const char floattype, 
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
//const double alpha,
//const double beta,
bool verbose, 
std::string logfile, 
bool c_is_const){
  
  
  /* The number of times each kernel is run in find. 
   * consider adding this parameter to user API. */
  unsigned n_runs_per_kernel = 3;

  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);  

  if (c_is_const == true){
  
    openclutil::SafeClMem c_copied = get_copy(command_queue, c, gg, toff, "copy of c in find");
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a, b, c_copied.clmem, workspace, logfile, verbose); //floattype, alpha, beta, 
    return oger.find(allotted_time, enforce_deterministic, n_runs_per_kernel);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a, b, c, workspace, logfile, verbose); //floattype,  alpha, beta, 
    return oger.find(allotted_time, enforce_deterministic, n_runs_per_kernel);
  }
}

tinygemm::TinyGemmSolution
get_default(
const bool enforce_deterministic,
const tinygemm::TinyGemmGeometry & gg,
bool verbose, 
std::string logfile){
  
 
  OpenCLGemmEncapsulator oger({}, gg, {0,0,0,0}, {}, {}, {}, {}, logfile, verbose);  //3.14, 3.14, floattype, 
  return oger.get_default(enforce_deterministic);
 
}
  
  



void benchgemm(
  cl_command_queue command_queue,
  const std::vector<hyperparams::HyperParams> & hps,
  unsigned n_runs,
  const tinygemm::TinyGemmGeometry & gg,
  const tinygemm::TinyGemmOffsets & toff, 
  cl_mem a_gpu,
  cl_mem b_gpu, 
  cl_mem c_gpu,
  cl_mem workspace_gpu,  
  bool verbose,
  std::string logfile,
  bool c_is_const){
   
  
  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);
  if (c_is_const == true){
    openclutil::SafeClMem c_copied = get_copy(command_queue, c_gpu, gg, toff, "copy of c in benchgemm");
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a_gpu, b_gpu, c_copied.clmem, workspace_gpu, logfile, verbose);
    oger.benchgemm(hps, n_runs);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a_gpu, b_gpu, c_gpu, workspace_gpu, logfile, verbose);
    oger.benchgemm(hps, n_runs);
  }
}
  
} //namespace





