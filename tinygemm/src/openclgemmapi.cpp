#include <stdexcept>
#include <thread>
#include <sstream>
#include <limits>
#include <chrono>
#include <limits>

#include "openclgemmapi.hpp"
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

namespace clgemm{



//class CloorException: public std::runtime_error {
  //virtual const char* what() const throw()
  //{
    //return "My exception happened";
  //}
//} cloorexception;


void confirm_cl_status(cl_int ret, std::string function, std::string argument){
  if (ret != CL_SUCCESS){
    std::stringstream errms;
    errms << "failure to set cl kernel arg, " << argument << ", in function " << function << "." << " error!";
    throw std::runtime_error(errms.str());
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



    

class OpenCLGemmEncapsulator{
public: 
  /* references to all constructor parameters */
  cl_context & context;
  cl_command_queue & command_queue;
  cl_device_id & device_id_to_use;
  std::string & outputfilename;
  const char & floattype; 
  const gemmgeometry::Geometry & gg;
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
  unsigned does_beta_c_inc;
  size_t n_work_groups;
  size_t local_work_size;
  size_t global_work_size;
  cl_int ret;
  cl_program program = NULL;
  cl_kernel kernel = NULL;    
  std::string kernel_function_name;
  cl_program betac_program = NULL;
  cl_kernel beta_c_kernel = NULL;
  std::string beta_c_kernel_function_name;
  /* parameters related to the scaling (beta C) kernel */
  unsigned dim_coal, dim_uncoal;
  size_t betac_global_work_size, betac_local_work_size;
  /* used for getting performance of main and (possibly) betac kernels */
  cl_event event_main_kernel;
  cl_event event_beta_c_kernel;
  /* stores (the most recent of n_runs) execution time */
  size_t t_start_beta_c_kernel, t_end_beta_c_kernel, t_start_main_kernel, t_end_main_kernel;
  float t_total_with_both, t_just_scaling, t_just_main;
  /* vector times over all n_runs runs */
  std::vector<float> v_t_total_with_both, v_t_just_scaling, v_t_just_main;


public:
      
  OpenCLGemmEncapsulator(
    cl_context & context,
    cl_command_queue & command_queue, 
    cl_device_id & device_id_to_use,
    const char & floattype,
    const gemmgeometry::Geometry & gg,  
    const double & alpha,
    const double & beta,
    cl_mem & a_gpu,
    cl_mem & b_gpu, 
    cl_mem & c_gpu,
    std::string & outputfilename,
    bool verbose):
      
    context(context),
    command_queue(command_queue), 
    device_id_to_use(device_id_to_use),
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
    mowri(verbose, outputfilename.compare("") != 0, outputfilename)
    {
      
      set_floatsize();
      run_checks();
      setup_beta_c_kernel(); //TODO : release when done.

    }
    

  void run_checks(){
    if (c_gpu == a_gpu || c_gpu == b_gpu){
      throw std::runtime_error("c should be distinct from a and b for gemm, otherwise race condition arises (one thread writes its result to c before another one has finished reading from c)");
    }
    /* Confirm that the input parameters make sense */
    consistencychecks::check_ldx_mnk_consistent(gg);
    sizingup::check_sizes_ok_for_unsigned(gg);
    if (floattype != 'd' and floattype != 'f'){
      throw std::runtime_error("floattype should be one of 'f' and 'd'");
    }
  }


  void setup_beta_c_kernel(){
    /* jit compile the betackernel source : change to loading a binary (?) */
    openclutil::set_program_and_kernel(betac_program, beta_c_kernel, beta_c_kernel_function_name, context, device_id_to_use, betac::get_cl_file_path(floattype));
    betac::set_betackernel_sizes(floattype, gg.isColMajor, gg.tC, gg.m, gg.n, dim_coal, dim_uncoal, betac_global_work_size, betac_local_work_size);
    mowri << "in setup_beta_c_kernel, global_work_size : " << betac_global_work_size << Endl; 
    //if (betac_global_work_size > 4*64*40*64){
      //mowri << "The number of threads running the scaling kernel looks very large (greater than 4 64 40 64). Could increase WORK_PER_THREAD to bring down... " << Endl;
    //}
    
    //mowri << "global work size (scaling kernel) is " << betac_global_work_size <<  " (recommended is ~ 4*64*40*64 =  655360)" <<  Endl;
 
    ret = clSetKernelArg(beta_c_kernel, 0, sizeof(unsigned), &dim_coal);
    confirm_cl_status(ret, "clSetKernelArg", "dim_coal");
    ret = clSetKernelArg(beta_c_kernel, 1, sizeof(unsigned), &dim_uncoal);
    confirm_cl_status(ret, "clSetKernelArg", "dim_uncoal");
    ret = clSetKernelArg(beta_c_kernel, 2, sizeof(unsigned), &gg.ldc);
    confirm_cl_status(ret, "clSetKernelArg", "ldc");
    ret = clSetKernelArg(beta_c_kernel, 3, sizeof(cl_mem), (void *)&c_gpu);
    confirm_cl_status(ret, "clSetKernelArg", "c_gpu");
    ret = clSetKernelArg(beta_c_kernel, 4, floatsize, m_beta[floattype]);// 
    confirm_cl_status(ret, "clSetKernelArg", "m_beta[floattype]");
  }


  void set_floatsize(){
    if (floattype == 'f'){
      floatsize = sizeof(float);
    }
    else if (floattype == 'd'){
      floatsize = sizeof(double);
    }
    
    else{
      throw std::runtime_error("Unrecognised floattype char. Currently, only 'f' (single precision) and 'd' (double precision) are supported");
    }
  }

  void check_file_and_set_from(std::string kernelfn){
    /* check that the parameters in the kernelfile look reasonable */
        
    kernelutil::check_gpu_kernel_preprocessor_parameters(kernelfn, gg.tA, gg.tB, gg.tC, gg.isColMajor, gg.m, gg.n, floattostring::get_float_string(floattype));
    
    /* extract the parameters which are needed to determine the number of work groups and work items to launch, directly from kernel source */
    kernelutil::set_sizes_from_kernel_source(macro_tile_width, macro_tile_height, n_workitems_per_workgroup, n_work_items_per_c_elm, does_beta_c_inc, kernelfn);
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
  }

  void enqueue_beta_c_kernel(){
    ret = clEnqueueNDRangeKernel(command_queue, beta_c_kernel, 1, NULL, &betac_global_work_size, &betac_local_work_size, 0, NULL, &event_beta_c_kernel);
    if (ret != CL_SUCCESS){
      throw std::runtime_error("Error in clEnqueueNDRangeKernel (in the scaling kernel)");
    }
  }


  
  int enqueue_main_kernel(){
    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0,NULL, &event_main_kernel);
    /* Handle the failure case which is not CL_OUT_OF_RESOURCES */ 
    if (ret != CL_SUCCESS && ret != CL_OUT_OF_RESOURCES){
      std::string errm("Error in clEnqueueNDRangeKernel (in the main kernel), The CL_STATUS value is ( ");
      errm += std::to_string(ret);
      errm += " )";
      errm += "\n";
      throw std::runtime_error(errm);
    }
    
    /* Either returning CL_SUCCESS or CL_OUT_OF_RESOURCES.  */
    return ret;
  }


  void update_run_times(){
    clGetEventProfilingInfo(event_main_kernel, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start_main_kernel, nullptr);
    clGetEventProfilingInfo(event_main_kernel, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end_main_kernel, nullptr);
    t_just_main = 1e-6*(t_end_main_kernel-t_start_main_kernel);
    v_t_just_main.push_back(t_just_main);
    if (does_beta_c_inc == 0){
      clGetEventProfilingInfo(event_beta_c_kernel, CL_PROFILING_COMMAND_START, sizeof(size_t), &t_start_beta_c_kernel, nullptr);
      clGetEventProfilingInfo(event_beta_c_kernel, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end_beta_c_kernel, nullptr);
      t_total_with_both = 1e-6*(t_end_main_kernel-t_start_beta_c_kernel);
      t_just_scaling = 1e-6*(t_end_beta_c_kernel-t_start_beta_c_kernel);
      v_t_total_with_both.push_back(t_total_with_both);
      v_t_just_scaling.push_back(t_just_scaling);
    }
    else{
      v_t_total_with_both.push_back(t_just_main);
    }
  }
  
  /* Will be used when a kernel failed to be enqueued, so just set then time really big */
  void update_run_times_really_bad(){
    t_just_main = std::numeric_limits<float>::max();
    v_t_total_with_both.push_back(std::numeric_limits<float>::max());
    v_t_total_with_both.push_back(std::numeric_limits<float>::max());
  }

  void print_run_times(){
    if (does_beta_c_inc == 0){
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
    
    reset_v_times();
    for (size_t kqq = 0; kqq < n_runs; ++kqq){

      /* This pause should have zero effect, but mysteriously it smooths out the run times between runs when working with certain gpu drivers
       * It has something to do with overheating.  */        
      if (n_runs > 1){
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
      }
  
      /* ***************************************************************************************
       *  Note on timing : the same results have been obtained whth timespec and std::chrono.  *
       *  **************************************************************************************/

      /* 20 Nov removed cl finish */

      /* Enqueue the beta c kernel if nec */
      if (does_beta_c_inc == 0){
        enqueue_beta_c_kernel();
      }


      /* At this point, the main kernel has been succesfully compiled, 
       * but it still possible that the resources necessary (LDS etc) are
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
  }


  void full_kernel_setup_from_source_file(std::string kernelfilename){

    /* set 5 parameters from given file: 
     * macro_tile_width, macro_tile_height, n_workitems_per_workgroup, n_work_items_per_c_elm, does_beta_c_inc */
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

    mowri << "INPUT_CALL   \t:" << gg.get_string() << Endl;
    full_kernel_setup_from_source_file(kernelfilename);
    mowri << "Entering the core gemm loops" << Endl;
    
    core_gemm_loop(n_runs);
    
    
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
  }//gemm
  
  
  
  
  
  tinygemm::TinyGemmSolution find(float allotted_time, bool enforce_deterministic){
    
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fp_ms = end - start;
    float elapsed_seconds = fp_ms.count();

    if (allotted_time <= 0){
      throw std::runtime_error("Should return the nearest cached : TODO develop caching scheme");
    }
    
    /* we will count how many kernels are successfully generated AND compiled AND benchmarked */
    unsigned global_counter = 0;
    
    /* we initialise the `hyper-front' with a single HyperParams, selected based on problem dimensions (TODO : should be based on cache table look-up) */
    std::vector<hyperparams::HyperParams> hyper_front = hyperparams::get_initial_front(gg, enforce_deterministic);
    auto hyper_param_start = hyper_front[0];
    
    /* we track the best TinyGemmSolution found during the search  */    
    std::vector<tinygemm::TinyGemmSolution> path_of_best_solns;
    
    /* In here, we will store all previously considered HyperParams, used to check and ensure that we do not consider a HyperParam more than once */
    std::vector<hyperparams::HyperParams> hyper_front_history;
    
    /* we set some initial parameters: the ones which are not in HyperParam object but eventually needed by the python script to generate kernel */
    std::map<std::string, unsigned> all_int_parms;
    all_int_parms["is_col_major"] = gg.isColMajor;
    all_int_parms["a_transposed"] = gg.tA;
    all_int_parms["b_transposed"] = gg.tB;
    all_int_parms["c_transposed"] = gg.tC;
    all_int_parms["use_edge_trick"] = 1;
    
    /* while generating, compiling and benchmarking kernels, we will keep track of the fastest found thus far */
    float best_time = std::numeric_limits<float>::max();
    hyperparams::HyperParams best_hyper_params;
    
    
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
      
      
      //std::abort();
      while (hfi < hyper_front.size() && improvement_found_on_front == false && elapsed_seconds < allotted_time){
        
        hyperparams::HyperParams hp = hyper_front[hfi];

        /* certain kernels will not be generated, for diverse reasons : */
        /* reason 0 : it's already been considered */
        if (std::find(hyper_front_history.begin(), hyper_front_history.end(), hp) != hyper_front_history.end()){
          /* this kernel has already been considered */
        }
        
        else{
          hyper_front_history.push_back(hp);
        
        
        
        
        
          //for (unsigned blaa = 0; blaa < 1000000; ++blaa){
        
        
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
              //mowri << x.first << "  : " << x.second  << Endl;
            }
             
            /* User defined parameters ? */
            std::string kerneldir = defpaths::scratchpadfinddir + "/";
            std::string kernelname = "atinygemmkernel";// + std::to_string(global_counter); // + ".cl";
          
            /* attempt to generate the kernel. Certain `bad' kernels are only caught at this stage, with tests for hyper-parameter compatibilty in the 
             * python script which I don't want to recode here. The main compatibility issue caught here is that load sizes from global are multiples
             * of the number of work items. */
            int kernel_write_status = mkkern::make_kernel_via_python(kerneldir,  floattostring::get_float_string(floattype), all_int_parms, kernelname);
            
            /* the kernel was succesfully generated, we now compile and benchmark it */
            if (kernel_write_status == 0){
              
              mowri << "\n" << hp.get_string() << Endl;
              ++global_counter;
              mowri << "global gen-com-bench : " << global_counter  <<  "." << Endl;
  
              std::string kernelfilename = kerneldir + kernelname + ".cl";
              
              /* bench 2 times. TODO: unhardwire this. */
              benchgemm(kernelfilename, 2);
              std::sort(v_t_total_with_both.begin(), v_t_total_with_both.end());
              float median_time = v_t_total_with_both[0]; //[v_t_total_with_both.size()/2]; taking the fastest. with 3 runs, this is reasonable :) fans, overheating, ergh.

              end = std::chrono::high_resolution_clock::now();
              fp_ms = end - start;
              elapsed_seconds = fp_ms.count();
                              
              mowri << "median time  : " << median_time << "\t m-Gflops/s : " << 2.0 * gg.m * gg.n * gg.k / (median_time * 1e6) << Endl;
              mowri << "elapsed seconds : " << elapsed_seconds << Endl;
      
              /* A new best kernel! we're only interested in an improvement if it's 0.5% or more */
              if (median_time < 0.995*best_time){
                improvement_found_on_front = true;
                best_time = median_time;
                best_hyper_params = hp;
                mowri << "---------- NEW BEST TIME FOUND --------- : " << best_time << Endl << "breaking from current hyper front, creating new hyper front " << Endl;
                
                tinygemm::TinyGemmSolution tgs;
                tgs.statistics.median_benchmark_time = median_time;
                tgs.statistics.median_benchmark_gflops = (2. * gg.m * gg.n * gg.k) / (median_time * 10e5);
                tgs.statistics.benchmarked_m = gg.m; 
                tgs.statistics.benchmarked_n = gg.n;
                tgs.statistics.benchmared_k = gg.k;


                end = std::chrono::high_resolution_clock::now();
                fp_ms = end - start;
                elapsed_seconds = fp_ms.count();
    
                tgs.statistics.time_when_benchmarked = elapsed_seconds;
                tgs.allparams = all_int_parms;
                tgs.floattype = floattype;
                
                //set kernel files
                //TODO : should not be determining whether a kernel does betac or not based on this, find true parameter
                if (all_int_parms.at("n_work_items_per_c_elm") == 1){
                  tgs.betac_kernel = "";
                }
                else{
                  tgs.betac_kernel = kernelutil::get_as_single_string(betac::get_cl_file_path(floattype));
                  tgs.betac_kernel_function_name = beta_c_kernel_function_name;
                }
                
                tgs.main_kernel = kernelutil::get_as_single_string(kernelfilename);
                tgs.main_kernel_function_name = kernel_function_name;

                path_of_best_solns.push_back(tgs); 
                
                
                
              }
            }
          
            else{
              //std::cout << kernel_write_status << std::endl;
              mowri << "\nSkipping this kernel, hyper-parameters incompatible" << Endl;
            }
          }
        }
        
        
        //} //delme
        
        ++hfi;
        
        end = std::chrono::high_resolution_clock::now();
        fp_ms = end - start;
        elapsed_seconds = fp_ms.count();
        
        //std::chrono::system_clock::now()-start;
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
      throw std::runtime_error("\nUser should never see this error, this is an internal problem. Possibly, there were no solutions found. Which is strange, as at least the initial kernel (the initial hyper front) should have been a solution. Either, the initial kernel was not valid (which should not happen unless my filters are broken) or for whatever reason the kernel was not generated or did not compile. Maybe there is some preceding warning printed which sheds light on this? Other possibilities are bad dir_name or kernelname in make_kernel.py. set verbose_report to True in makekernelsource if you (james) can't figure this out ");
    }
  
    mowri << "best time : " << best_time << Endl;
    mowri << "best kernel : " << best_hyper_params.get_string() << Endl;
    mowri << "start kernel : " << hyper_param_start.get_string() << Endl;
    
    mowri << "the kernels along the path the final solution :  " << Endl; 
    mowri << "hyper parameter string                                     \t time when found\t median gflop/s" << Endl;



    for (auto & x : path_of_best_solns){
      mowri <<  x.get_hyper_param_string() << "\t " << x.statistics.time_when_benchmarked << "\t\t " << x.statistics.median_benchmark_gflops  << Endl;
    } 
    mowri <<  path_of_best_solns.back().get_hyper_param_string() << "\t " << elapsed_seconds << "\t\t " << path_of_best_solns.back().statistics.median_benchmark_gflops  << Endl;
    
    
    
    return path_of_best_solns.back();
  }
  
}; 






/* functions for the end-user */ 
tinygemm::TinyGemmSolution
find(
float allotted_time,
cl_context & context,
cl_command_queue & command_queue,
cl_device_id & device_id_to_use,
cl_mem a,   
cl_mem b,
cl_mem c,
const bool enforce_deterministic,
const char floattype, 
const gemmgeometry::Geometry & gg,
const double alpha,
const double beta,
bool verbose, 
std::string logfile){
  
  //TODO : add bool verbose and std::string logfile
  


  /*user defined filename ?*/
  
  //defpaths::scratchpadfinddir
  
//  std::string outputfilename("");
  OpenCLGemmEncapsulator oger(context, command_queue, device_id_to_use, floattype, gg, alpha, beta, a, b, c, logfile, verbose);
  return oger.find(allotted_time, enforce_deterministic);

}





void benchgemm(
  cl_context & context,
  cl_command_queue & command_queue, 
  cl_device_id & device_id_to_use,
  std::string kernelfilename,
  unsigned n_runs,
  const char floattype, 
  const gemmgeometry::Geometry & gg,
  const double alpha,
  const double beta,
  cl_mem a_gpu,
  cl_mem b_gpu, 
  cl_mem c_gpu,
  bool verbose,
  std::string logfile){
    
  OpenCLGemmEncapsulator oger(context, command_queue, device_id_to_use, floattype, gg, alpha, beta, a_gpu, b_gpu, c_gpu, logfile, verbose);
  oger.benchgemm(kernelfilename, n_runs);
}
  
} //namespace




//Y256_X256_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2
//Y256_X256_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE4
