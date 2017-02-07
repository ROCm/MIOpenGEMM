#ifndef BASICFIND_HPP
#define BASICFIND_HPP

#include <vector>
#include <thread>
#include <chrono>

#include <map>
#include <chrono>
#include <sstream>

/* Required for basic error check performed */
#include <algorithm>

/* Required header for using tinygemm */
#include <tinygemm/tinygemm.hpp>

/* The following two header files define functions which help with opencl boilerplating on my systems, they are not necessary */
#include <tinygemm/outputwriter.hpp>
#include <tinygemm/openclutil.hpp>
#include <tinygemm/accuracytests.hpp>

/* This is needed for accuracy testing */
#include <tinygemm/slowcpugemm.hpp>

/* used to populate data with random values */
#include "setabcw.hpp"
  
template <typename TFloat> 

void basicfind(const tinygemm::TinyGemmGeometry & geometry, const tinygemm::TinyGemmOffsets & toff, float allotted_time, bool verbose, std::string logfile, bool enforce_deterministic, unsigned n_postfind_runs, bool do_cpu_test){
  
  /* just checking that geometry floattype is correct */
  if (!((geometry.floattype == 'f' && sizeof(TFloat) == 4) || (geometry.floattype == 'd' && sizeof(TFloat) == 8))) {
    throw tinygemm::tinygemm_error("disagreement between geometry.floattype and sizeof(TFloat) in basicfind.hpp");
  }

  double alpha = tinygemm::default_alpha;
  double beta = tinygemm::default_beta;
  
  /* generating cpu copies of data */
  std::vector<TFloat> v_a;
  std::vector<TFloat> v_b;
  std::vector<TFloat> v_c;
  std::vector<TFloat> v_workspace;
  setabcw::set_abcw(v_a, v_b, v_c, v_workspace, geometry, toff);
  
  size_t n_a = v_a.size();
  size_t n_b = v_b.size();  
  size_t n_c = v_c.size(); 
  size_t n_w = v_workspace.size();
  
  
  /* On OpenCL boilerplate. 
   * This might be different depending on your system. 
   * tinygemm does not help in setting up OpenCL boilerplate, 
   * and assumes you can allocate memory buffers, find devices etc. 
   * Here we use our own boilerplate setup functions, which might not 
   * work on your system, but you can give it a try and see. Otherwise, 
   * this section can be changed.  
   * ******************************************************************
   * ******************************************************************
   * ******************************************************************/
    
  cl_context context;
  cl_device_id device_id_to_use;
  cl_command_queue command_queue;

  cl_platform_id platform = nullptr;
  cl_uint num_platforms;
  tinygemm::outputwriting::OutputWriter mowri(verbose, logfile.compare("") != 0, logfile);
  tinygemm::openclutil::set_platform_etc(platform, num_platforms, context, device_id_to_use, mowri);
  
  /* we use are own version of clCreateCommandQueue (and other opencl functions), which has an added layer of error detection */
  command_queue = tinygemm::openclutil::cl_create_command_queue(context, device_id_to_use, CL_QUEUE_PROFILING_ENABLE, "in basicfind.hpp");  
  
  /* writing cpu arrays to gpu */
  cl_mem a_gpu = tinygemm::openclutil::cl_create_buffer(context, CL_MEM_READ_ONLY,  sizeof(TFloat)*n_a, NULL, "a_gpu in basicfind.hpp");
  cl_mem b_gpu = tinygemm::openclutil::cl_create_buffer(context, CL_MEM_READ_ONLY,  sizeof(TFloat)*n_b, NULL, "b_gpu in basicfind.hpp");
  cl_mem c_gpu = tinygemm::openclutil::cl_create_buffer(context, CL_MEM_READ_WRITE, sizeof(TFloat)*n_c, NULL, "c_gpu in basicfind.hpp");      
  cl_mem workspace_gpu = tinygemm::openclutil::cl_create_buffer(context, CL_MEM_READ_WRITE, sizeof(TFloat)*n_w, NULL, "workspace_gpu in basicfind.hpp");

  tinygemm::openclutil::cl_enqueue_write_buffer(command_queue, a_gpu, CL_TRUE, 0, sizeof(TFloat)*n_a, v_a.data(), 0, NULL, NULL, "a_gpu in basicfind.hpp");
  tinygemm::openclutil::cl_enqueue_write_buffer(command_queue, b_gpu, CL_TRUE, 0, sizeof(TFloat)*n_b, v_b.data(), 0, NULL, NULL, "b_gpu in basicfind.hpp");
  tinygemm::openclutil::cl_enqueue_write_buffer(command_queue, c_gpu, CL_TRUE, 0, sizeof(TFloat)*n_c, v_c.data(), 0, NULL, NULL, "c_gpu in basicfind.hpp");
  tinygemm::openclutil::cl_enqueue_write_buffer(command_queue, workspace_gpu, CL_TRUE, 0, sizeof(TFloat)*n_w, v_workspace.data(), 0, NULL, NULL, "workspace_gpu in basicfind.hpp");

  /* ******************************************************************
  * ******************************************************************
  * ******************************************************************/
  
  

  /* *****************
   * Find a solution *
   * *****************/
  tinygemm::TinyGemmSolution soln = tinygemm::find(allotted_time, command_queue, a_gpu, b_gpu, c_gpu, workspace_gpu, enforce_deterministic, geometry, toff, verbose, logfile);
  
    
  if (do_cpu_test == true && n_postfind_runs < 1){
    throw tinygemm::tinygemm_error("(in basicfind.hpp, part of example/test suite) do_cpu_test is true, and n_postfind_runs < 1. If you wish to run the cpu test, n_postfind_runs should take a positive integral value");
  }
  
  /* ****************************************************
   * how to proceed after the kernel(s) have been found *
   * ****************************************************/
  if (n_postfind_runs > 0){

    /* We now show how to use the kernel(s) in soln.
     * We use the kernel on the same problem as benchmarked. 
     * We expect this to be the standard workflow with tinygemm.  */
        
    /* Note that the alpha and beta used in the `find' step are always double, even if the data type is float.  
     * This above design choice was made to reduce the amount of object code. 
     * However, when the kernel is to be used, alpha and beta must corrspond the type of the data. */
    TFloat beta_true_type = static_cast<TFloat>(beta);
    TFloat alpha_true_type = static_cast<TFloat>(alpha);

    std::string buildOptions_11 = "-cl-std=CL2.0 -Werror";
    auto buildOptions = buildOptions_11.c_str();
    
    
    std::vector<cl_program> clprograms;
    std::vector<cl_kernel> clkernels;
    std::vector<cl_event> clevents;

    for (auto & ks : soln.v_tgks){
            
      auto kernel_cstr = ks.kernstr.c_str();
      auto fname_cstr = ks.fname.c_str();
      size_t source_size = ks.kernstr.size();

      clprograms.emplace_back ( 
        tinygemm::openclutil::cl_create_program_with_source(context, 1, &kernel_cstr, &source_size, ks.type + ": creating program (in basicfind.hpp)")
      );
      
      tinygemm::openclutil::cl_build_program(clprograms.back(), 1, &device_id_to_use, buildOptions, NULL, NULL, ks.type + ": building program (in basicfind.hpp)");     
      
      clkernels.emplace_back (
        tinygemm::openclutil::cl_create_kernel(clprograms.back(), fname_cstr, ks.type + "creating kernel (in basicfind.hpp)")
      );
      
      clevents.emplace_back ();
      
      if (ks.type.compare("betac") == 0){
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 0, sizeof(unsigned), &geometry.derived.dim_c_coal, "betac 0");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 1, sizeof(unsigned), &geometry.derived.dim_c_uncoal, "betac 1");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 2, sizeof(unsigned), &geometry.ldc, "betac 2");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 3, sizeof(unsigned), &toff.oc, "betac 3");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 4, sizeof(cl_mem), (void *)&c_gpu, "betac 4");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 5, sizeof(TFloat), &beta_true_type, "betac 5"); 
      }
      
      else if (ks.type.compare("alphaonso") == 0){
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 0, sizeof(cl_mem), (void *)&c_gpu, "main kernel 0 (basicfind.hpp)");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 1, sizeof(cl_mem), (void *)&a_gpu, "main kernel 1 (basicfind.hpp)");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 2, sizeof(cl_mem), (void *)&b_gpu, "main kernel 2 (basicfind.hpp)");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 3, sizeof(TFloat), &alpha_true_type, "main kernel 3 (basicfind.hpp)");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 4, sizeof(TFloat), &beta_true_type, "main kernel 4 (basicfind.hpp)");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 5, sizeof(unsigned), &toff.oa, "main kernel 5 (basicfind.hpp)");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 6, sizeof(unsigned), &toff.ob, "main kernel 6 (basicfind.hpp)");
        tinygemm::openclutil::cl_set_kernel_arg(clkernels.back(), 7, sizeof(unsigned), &toff.oc, "main kernel 7 (basicfind.hpp)");  
      }
      
      else{
        throw tinygemm::tinygemm_error("unrecogised kernel type, " + ks.type);
      }
    }
    
    
    /* Enqueueing the kernel(s)  */
    auto enqueue_kernels_serial = [&soln, &clevents, &command_queue, &clkernels] (std::string hash) {

      for (unsigned ki = 0; ki < soln.v_tgks.size(); ++ki){
        size_t n_events_to_wait_on = ki == 0 ? 0 : 1;
        cl_event * events_to_wait_on = ki == 0 ? nullptr : &clevents[ki - 1];
        tinygemm::openclutil::cl_enqueue_ndrange_kernel(
        command_queue, clkernels[ki], 1, NULL, &soln.v_tgks[ki].global_work_size, &soln.v_tgks[ki].local_work_size, n_events_to_wait_on, events_to_wait_on, &clevents[ki], "Error in basicfind.hpp, enqueueing " + soln.v_tgks[ki].type + " in call to enqueue_kernels with hash : " + hash);
      }
    };
    
    enqueue_kernels_serial("first enqueue");
    
    tinygemm::openclutil::cl_wait_for_events(1, &clevents.back(), "basicfind.hpp, waiting for " + soln.v_tgks.back().type + " in call to enqueue_kernels after postfind first enq.");
    
    if (do_cpu_test == true){

      /* We do a check with cpu */
      std::vector<std::string> algs {"3fors"};
      auto c_cpu_final = v_c;
      tinygemm::slowcpugemm::gemms_cpu<TFloat>(geometry, toff, v_a.data(), v_b.data(), c_cpu_final.data(), alpha, beta, algs, mowri);
      auto c_copied_from_gpu = std::vector<TFloat>(v_c.size(), 0);
      cl_event event_read_c_back;
      tinygemm::openclutil::cl_enqueue_read_buffer(command_queue, c_gpu, CL_TRUE, 0, sizeof(TFloat)*c_copied_from_gpu.size(), c_copied_from_gpu.data(), 0, NULL, &event_read_c_back, "read in basicfind.hpp s.");  
      clWaitForEvents(1, &event_read_c_back);

      bool old_to_terminal = mowri.to_terminal;
      mowri.to_terminal = true;
      tinygemm::accuracytests::elementwise_compare<TFloat>(v_c.data(), beta, c_copied_from_gpu.data(), c_cpu_final.data(), v_c.size(), mowri);
      mowri.to_terminal = old_to_terminal;      
    }

    /* That's all you need to know, and don't forget those clReleases! */

    if (n_postfind_runs > 1){
      /* We now take a look at how the times reported in benchmarking, 
       * which use cl_events to get accurate gpu times, compare to times
       * obtained here on the host side.  */ 
      std::map<unsigned, float> host_times;
      for (unsigned npr : std::vector<unsigned> {1, n_postfind_runs + 1}){
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned pfr = 0; pfr < npr; ++pfr){
          enqueue_kernels_serial("in postfind runs");
        }

        /* Wait for the final kernel to complete, then record the elapsed time */
        tinygemm::openclutil::cl_wait_for_events(1, &clevents.back(), "basicfind.hpp, waiting for " + soln.v_tgks.back().type + " in call to enqueue_kernels after postfind runs");
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> fp_ms = end - start;
        float elapsed_seconds = fp_ms.count();
        host_times[npr] = elapsed_seconds;  
        std::cout << "Time to complete " << npr << " run(s) : " << elapsed_seconds << " [s]." << " This corresponds to " << 1e-9*(2.*geometry.m*geometry.n*geometry.k*npr) / elapsed_seconds << " gflop/s. " << std::endl;
      }
      
      float difference_in_times = host_times[n_postfind_runs + 1] - host_times[1];
      std::cout << "Difference in times : " << difference_in_times << " [s]. This corresponds to  " << 1e-9*(2.*geometry.m*geometry.n*geometry.k*n_postfind_runs) / difference_in_times << " gflop/s. " << std::endl;
      /* std::cout << "Note to self and MLOpeners : we need to decide how to proceed with the DeepBench benchmarking, baidu's approach  (run 10 times after warm-up, no subtraction as above) *might* underestimate cudnn performance. For many problems (generally the large ones), tinygemm and host times are the same. Occasionally (for small problems it seems), the host time is 20% slower." << std::endl; */
      std::cout << soln.get_hyper_param_string() << std::endl;
    }
  }
  
  /* Cleaning up, closing shop. */
  tinygemm::openclutil::cl_release_mem_object(c_gpu, "c_gpu in basicfind.hpp");
  tinygemm::openclutil::cl_release_mem_object(a_gpu, "a_gpu in basicfind.hpp");
  tinygemm::openclutil::cl_release_mem_object(b_gpu, "b_gpu  in basicfind.hpp");
  tinygemm::openclutil::cl_release_mem_object(workspace_gpu, "workspace_gpu  in basicfind.hpp");

  tinygemm::openclutil::cl_release_command_queue(command_queue, "command queue in basicfind.hpp");
  tinygemm::openclutil::cl_release_context(context, "context in basicfind.hpp");
  

}


#endif
