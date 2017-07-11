/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <miopengemm/architests.hpp>
#include <miopengemm/bundle.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/openclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>

namespace MIOpenGEMM
{

class MultiFloatType
{

  private:
  double v_d;
  float  v_f;

  public:
  MultiFloatType(double v) : v_d(v), v_f(static_cast<float>(v)) {}
  void* operator[](char floattype) const
  {
    return floattype == 'd' ? (void*)(&v_d) : (void*)(&v_f);
  }
};

static const MultiFloatType m_alpha(default_alpha);
static const MultiFloatType m_beta(default_beta);

class GPUMems
{
  private:
  
  std::array<cl_mem, Mem::E::N> cl_mems;

  public:
  GPUMems(cl_mem a_gpu_, cl_mem b_gpu_, cl_mem c_gpu_, cl_mem workspace_gpu_){
  cl_mems[Mem::E::A] = a_gpu_;
  cl_mems[Mem::E::B] = b_gpu_;
  cl_mems[Mem::E::C] = c_gpu_;    
  cl_mems[Mem::E::W] = workspace_gpu_;      
  }

  cl_mem& operator[](Mem::E x)
  {
    // TODO : bound check here if in debug mode.
    return cl_mems[x];
  }
};

class OpenCLGemmEncapsulator
{

  public:
  cl_command_queue                   command_queue;
  std::string                        outputfilename;
  const Geometry                     gg;
  const Offsets                      toff;
  GPUMems                            gpum;
  const openclutil::OpenCLDeviceInfo devinfo;
  std::string                        constraints_string;
  hyperparams::Graph                 graph;

  private:
  outputwriting::OutputWriter& mowri;
  // special purpose output writer
  // vector of times over a set of runs on core loop
  std::vector<float> v_t_total;
  float              median_time;
  float              median_gflops;
  // (for find) while generating, compiling and
  // benchmarking kernels, we will keep track of the
  // fastest found thus far
  std::vector<Solution>                                       best_solns_path;
  std::vector<Kernel>                                         tk_kernels;
  std::vector<Kernel*>                                        tk_kernels_active;
  std::vector<std::vector<size_t>>                          v_wait_indices;
  bool                                                        bundle_verbose{false};
  float                                                       total_elapsed_seconds{0};
  size_t                                                    total_elapsed_descents{0};
  size_t                                                    total_kernels_tested{0};
  std::chrono::time_point<std::chrono::high_resolution_clock> find_start;
  std::string                                                 old_comment_string;
  std::string                                                 new_comment_string;

  public:
  OpenCLGemmEncapsulator(cl_command_queue             command_queue_,
                         const Geometry               gg_,
                         const Offsets                toff_,
                         cl_mem                       a_gpu_,
                         cl_mem                       b_gpu_,
                         cl_mem                       c_gpu_,
                         cl_mem                       workspace_gpu_,
                         std::string                  constraints_string_,
                         bool                         full_constraints_expected,
                         outputwriting::OutputWriter& mowri_):
                         
 
      command_queue(command_queue_),
      gg(gg_),
      toff(toff_),
      gpum(a_gpu_, b_gpu_, c_gpu_, workspace_gpu_),

      devinfo(command_queue_),
      constraints_string(constraints_string_),
      graph(gg, devinfo, constraints_string_, full_constraints_expected),

      mowri(mowri_)
  {

    tk_kernels.resize(BasicKernelType::E::N);
    for (size_t i = 0; i < BasicKernelType::E::N; ++i)
    {
      tk_kernels[i] = Kernel(command_queue, BasicKernelType::M.name[i]);
    }

    run_checks();
  }

  private:
  /* TODO : option for median / max / mean */
  void set_medians()
  {

    auto v_t_total_copy = v_t_total;

    std::sort(v_t_total_copy.begin(), v_t_total_copy.end());
    // Taking the fastest or median? [v_t_total.size()/2]
    median_time   = v_t_total_copy[0];
    median_gflops = get_gflops(median_time);
  }

  float get_gflops(float timems) { return (2. * gg.m * gg.n * gg.k) / (timems * 10e5); }

  void address_check_valid()
  {
    if (gpum[Mem::E::C] == gpum[Mem::E::A] || gpum[Mem::E::C] == gpum[Mem::E::B])
    {
      throw miog_error("in address_check_valid, c should be distinct from a and b for gemm, "
                       "otherwise race condition arises (one thread writes its result to c "
                       "before "
                       "another one has finished reading from c)");
    }

    if (gpum[Mem::E::C] == nullptr)
    {
      throw miog_error("in address_check_valid, c should not be nullptr");
    }

    if (gpum[Mem::E::W] == nullptr && gg.workspace_size != 0)
    {
      throw miog_error("in address_check_valid, pointer to workspace memory is "
                       "the nullptr, but "
                       "workspace_size is not zero");
    }

    if (gpum[Mem::E::W] != nullptr && gg.workspace_size == 0)
    {
      throw miog_error("in address_check_valid, pointer to workspace memory is not the "
                       "nullptr, "
                       "but workspace_size is zero. if workspace_size is zero please set "
                       "workspace_gpu to the nullptr to make super clear that there will be "
                       "no "
                       "workspace used. The workspace offset should be zero too in this case ");
    }

    if (gpum[Mem::E::W] != nullptr &&
        (gpum[Mem::E::W] == gpum[Mem::E::A] || gpum[Mem::E::W] == gpum[Mem::E::B] || gpum[Mem::E::W] == gpum[Mem::E::C]))
    {
      throw miog_error("in address_check_valid, pointer to workspace memory is "
                       "not the nullptr, "
                       "and it is the same as one of the a,b,c pointers ");
    }
  }

  void address_check_valid_and_reliable()
  {
    address_check_valid();
    if (gpum[Mem::E::A] == gpum[Mem::E::B])
    {
      throw miog_error("in address_check_valid_and_reliable, a and b are the "
                       "same. this will "
                       "effect kernel run time, not sure if this should be "
                       "allowed, so throwing");
    }
  }

  void run_checks() { } //sizingup::check_sizes_ok_for_size_t(gg, toff); }

  void set_kern_args(const KernelType& type)
  {

    // parameter order rule: {a, oa, b, ob, c, oc, ws, ows}, alpha, beta
    std::vector<std::pair<size_t, const void*>> arg_sizes_values;

    for (auto x : {Mem::E::A, Mem::E::B, Mem::E::C, Mem::E::W})    
    {
      if (type.uses(x) == true)
      {
        arg_sizes_values.emplace_back(sizeof(cl_mem), (void*)&(gpum[x]));
        arg_sizes_values.emplace_back(sizeof(size_t), &(toff.offsets[x]));
      }
    }

    if (type.uses_alpha)
    {
      arg_sizes_values.emplace_back(gg.derived.float_size_bytes, m_alpha[gg.floattype]);
    }

    if (type.uses_beta)
    {
      arg_sizes_values.emplace_back(gg.derived.float_size_bytes, m_beta[gg.floattype]);
    }

    tk_kernels.at(type.basic_kernel_type).set_kernel_args(arg_sizes_values);
  }

  bool refresh_needed(BasicKernelType::E                                 type,
                      const hyperparams::HyperParams&     new_hp,
                      const derivedparams::DerivedParams& new_dp)
  {

    /* TODO : check (here) hyper parameters to see if needed anew */

    if (type == BasicKernelType::E::BETAC)
    {
      if (tk_kernels.at(BasicKernelType::E::BETAC).is_set() == false && new_dp.main_does_beta_c_inc == 0)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    else if (type == BasicKernelType::E::MAIN)
    {
      return true;
    }

    else if (type == BasicKernelType::E::WSA)
    {
      if (tk_kernels.at(BasicKernelType::E::WSA).is_set() == false && new_hp.at(Mat::E::A).vs[Chi::E::WOS] != Scratch::E::UNUSED)
      {
        return true;
      }
      else
      {
        return false;
      }
    }


    else if (type == BasicKernelType::E::WSB)
    {
      if (tk_kernels.at(BasicKernelType::E::WSB).is_set() == false && new_hp.at(Mat::E::B).vs[Chi::E::WOS] != Scratch::E::UNUSED)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    else
    {
      throw miog_error("what is the type of this kernel? Don't recognise it : " + type);
    }
  }

  openclutil::OpenCLResult refresh_kernel(const KernelString&                 ks,
                                          const hyperparams::HyperParams&     hp,
                                          const derivedparams::DerivedParams& dp)
  {

    openclutil::OpenCLResult oclr;
    auto                     type = ks.type;
    if (refresh_needed(type.basic_kernel_type, hp, dp) == true)
    {

      oclr = tk_kernels.at(type.basic_kernel_type).update(ks, mowri);
      if (oclr.fail() == false)
      {
        set_kern_args(type);
      }

      else
      {
        oclr.message += " (failed from refresh_kernel) ";
      }
    }
    return oclr;
  }

  openclutil::OpenCLResult setup_tinykernels(const hyperparams::HyperParams& hp,
                                             const kerngen::Bundle&          bundle)
  {

    openclutil::OpenCLResult oclr;

    v_wait_indices = bundle.v_wait_indices;
    tk_kernels_active.resize(0);

    for (size_t ksi = 0; ksi < bundle.v_tgks.size(); ++ksi)
    {

      BasicKernelType::E basic = bundle.v_tgks[ksi].type.basic_kernel_type;
      oclr      = refresh_kernel(bundle.v_tgks[ksi], hp, bundle.dp);
      if (oclr.fail() == false)
      {
        tk_kernels_active.push_back(&tk_kernels[basic]);
      }
      else
      {
        oclr.message += " (failed in setup_tinykernels) ";
        return oclr;
      }
    }

    return oclr;
  }

  void update_run_times(cl_int status)
  {

    if (status == CL_SUCCESS)
    {
      for (auto& ptr_tk_kernel : tk_kernels_active)
      {
        ptr_tk_kernel->update_times();
      }
      // end time of last kernel - start time of first kernel
      v_t_total.push_back(1e-6 * (tk_kernels_active.back()->t_end - tk_kernels_active[0]->t_start));
    }

    else
    {
      throw miog_error("in update_run_times, status is not CL_SUCCESS. The "
                       "logic has changed, this "
                       "logic branch should be impossible");
    }
  }

  std::string get_run_times_heading()
  {
    std::stringstream ss;
    ss << "tt: \t";
    for (size_t k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind)
    {
      ss << " k" << k_ind << ":\t";
    }
    ss << " Gflops/s:\n";
    return ss.str();
  }

  std::string get_run_time_string(cl_int status)
  {
    std::stringstream ss;
    if (status == CL_SUCCESS)
    {
      ss << std::fixed << std::setprecision(3) << v_t_total.back() << "\t";
      for (size_t k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind)
      {
        ss << " " << tk_kernels_active[k_ind]->v_times.back() << "\t";
      }
      ss << " " << 2.0 * gg.m * gg.n * gg.k / (v_t_total.back() * 1e6) << std::setprecision(6);
    }

    else
    {
      ss << "(failed run)";
    }
    return ss.str();
  }

  void reset_v_times()
  {
    v_t_total.resize(0);
    for (auto& ptr_tk_kernel : tk_kernels_active)
    {
      ptr_tk_kernel->reset_times();
    }
  }

  void mowri_tracker_print()
  {
    std::stringstream comment_string_ss;
    comment_string_ss << "[ TOTAL TIME:"
                      << stringutil::get_padded(static_cast<int>(total_elapsed_seconds), 7);
    comment_string_ss << "  #RESTARTS:" << stringutil::get_padded(total_elapsed_descents, 7);
    comment_string_ss << "  #GEMMS CONSIDERED:" << stringutil::get_padded(total_kernels_tested, 7)
                      << "]       ";
    old_comment_string     = new_comment_string;
    new_comment_string     = comment_string_ss.str();
    std::string backspaces = std::string(old_comment_string.size(), '\b');
    /* TODO : determine where to use mowri_tracker,
     * and enable to path to here */
    mowri.tracker << backspaces << new_comment_string << Flush;
  }

  openclutil::OpenCLResult core_gemm_loop(size_t max_n_runs, double max_time, bool print_asap)
  {

    update_total_elapsed_seconds();

    reset_v_times();
    mowri_tracker_print();

    std::vector<std::string> indi_run_strings;
    if (print_asap == true)
    {
      mowri << get_run_times_heading();
    }

    size_t runi = 0;
    double local_elapsed_seconds = 0;
    auto start_kernel_run0 = std::chrono::high_resolution_clock::now();
    
    

    while (runi < max_n_runs && local_elapsed_seconds < max_time)
    {

      // see `oeverheat' comment at bottome
      if (max_n_runs > 1)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
      }

      openclutil::OpenCLResult oclr;

      for (size_t k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind)
      {
        // At this point, the kernel has been succesfully compiled,
        // but it is still possible that the resources necessary (LDS etc) are
        // not sufficient on this machine. We catch this case here.

        /* TODO : architests can go some way to catching these before */

        std::vector<cl_event> clevent_waits;
        for (auto& evi : v_wait_indices[k_ind])
        {
          // see `cl-events' comment at bottom
          clevent_waits.emplace_back(tk_kernels_active[evi]->clevent);
        }

        size_t          num_events_int_wait_list = clevent_waits.size();
        const cl_event* event_wait_list =
          num_events_int_wait_list == 0 ? nullptr : clevent_waits.data();
        oclr = tk_kernels_active[k_ind]->enqueue(num_events_int_wait_list, event_wait_list);
        // see `in-series' comment at bottom

        // Set the run time(s) and append to vectors
        if (oclr.success == CL_OUT_OF_RESOURCES)
        {
          openclutil::cl_flush(command_queue, "cl flushing in core gemm loop", true);
          oclr.message += " (CL_OUT_OF_RESOURCES in core_gemm_loop) ";
          return oclr;
        }

        else if (oclr.success != CL_SUCCESS)
        {
          std::stringstream ss;
          ss << "OpenCL error status : " << oclr.success << ". ";
          ss << "This seems like a logic error. How can this not be CL_SUCCESS "
                "or "
                "CL_OUT_OF_RESOURCES? Maybe an algo prob, come fix. Is status "
                "10111? Maybe there "
                "are no kernels? Maybe there's a runtime/compiler issue? One "
                "option is to lump "
                "this error with CL_OUT_OF_RESOURCES (ie throw oclr) ";
          ss << "The error from opencl was " << oclr.message;
          throw miog_error(ss.str());
        }

        else
        {
          // CL_SUCCESS
        }
      }

      openclutil::cl_flush(command_queue, "cl flushing in core gemm loop", true);

      // Wait for kernels to complete
      openclutil::cl_wait_for_events(1,
                                     &(tk_kernels_active.back()->clevent),
                                     "with status == CL_SUCCESS in core gemm loops",
                                     true);

      update_run_times(oclr.success);
      indi_run_strings.push_back(get_run_time_string(oclr.success));
      if (print_asap == true)
      {
        mowri << indi_run_strings[runi] << '\n';
      }
      ++runi;
      
      
      auto t_now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> fp_ms = t_now - start_kernel_run0;
      local_elapsed_seconds = fp_ms.count();


    }
    size_t runs_completed = runi;

    set_medians();

  
    if (print_asap == false)
    {
      mowri << get_run_times_heading();
      for (size_t ir = 0; ir < runs_completed; ++ir)
      {
        mowri << indi_run_strings[ir];
        if (median_time == v_t_total[ir])
        {
          mowri << " (median) ";
          if (best_solns_path.size() > 0 &&
              (best_solns_path.back().statistics.median_benchmark_time >= median_time))
          {
            mowri << " (NEW BEST) ";
          }
        }
        mowri << '\n';
      }
    }

    ++total_kernels_tested;
    return {};
  }

  void deriveability_test(const hyperparams::HyperParams& hp, const std::string& hash)
  {
    auto deriveability = derivedparams::get_deriveability(hp, gg);
    if (std::get<0>(deriveability) == false)
    {
      throw miog_error(hash + ": the hyper parameters in benchgemm are not consistent, "
                              "specifically, from get_deriveability \n" +
                       std::get<1>(deriveability));
    }
  }

  public:
  void benchgemm(size_t max_n_runs, double max_time)
  {

    address_check_valid();

    if (max_n_runs == 0)
    {
      throw miog_error("max_n_runs to benchgemm should be a positive integer");
    }

    hyperparams::HyperParams hp(graph);
    deriveability_test(hp, "in benchgemm");

    auto bundle = kerngen::get_bundle(hp, gg, mowri, bundle_verbose);
    auto atr    = architests::architecture_specific_tests(command_queue, bundle.dp, gg, hp);
    if (std::get<0>(atr) == false)
    {
      throw miog_error(std::get<1>(atr));
    }

    auto oclr = setup_tinykernels(hp, bundle);
    if (oclr.fail())
    {
      throw miog_error(oclr.message);
    }

    mowri << "(benchgemm) hp   :" << hp.get_string() << Endl;
    mowri << "(benchgemm) geometry  \t:" << gg.get_string() << "\nEntering the core gemm loops"
          << Endl;
    oclr = core_gemm_loop(max_n_runs, max_time, true);

    if (oclr.fail())
    {
      throw miog_error(oclr.message);
    }
  }

  hyperparams::HyperParams get_hyper_param_start()
  {

    hyperparams::HyperParams hyper_param_start(graph);
    hyper_param_start.checks();

    bool              found_a_deriveable_goodarchi_hp = false;
    size_t          d_and_g_search_iteration        = 0;
    std::stringstream d_and_g_ss;

    // the number of attempts at finding a
    // deriveable HyperParams given the
    // constraint string
    const size_t n_trials = 100000;

    while (found_a_deriveable_goodarchi_hp == false && d_and_g_search_iteration < n_trials)
    {
      hyper_param_start = hyperparams::HyperParams(graph);
      hyper_param_start.checks();
      auto deriveability = derivedparams::get_deriveability(hyper_param_start, gg);
      if (std::get<0>(deriveability) == false)
      {
        d_and_g_ss << hyper_param_start.get_string()
                   << " is not deriveable, because : " << std::get<1>(deriveability) << "\n\n";
      }

      else
      {
        auto dp = derivedparams::DerivedParams(hyper_param_start, gg);
        auto atr =
          architests::architecture_specific_tests(command_queue, dp, gg, hyper_param_start);
        if (std::get<0>(atr) == false)
        {
          d_and_g_ss << hyper_param_start.get_string()
                     << "failed archotests because : " << std::get<1>(atr);
        }
        else
        {
          found_a_deriveable_goodarchi_hp = true;
        }
      }
      ++d_and_g_search_iteration;
    }

    mowri << "n trials looking for a viable starting node in the graph : "
          << d_and_g_search_iteration << Endl;

    // force the graph starting parameters
    if (found_a_deriveable_goodarchi_hp == false)
    {
      std::stringstream base_ss;
      base_ss << "\n\nStruggling to find a deriveable set of hyper parameters "
                 "which satisfy the "
                 "geometry and constraints. ";
      base_ss << "The number of attempts made is " << n_trials
              << ".  To view the full output of the hyper parameters tried and "
                 "their reasons for "
                 "not being derivable, modify the code here (add "
                 "d_and_g_ss.str()). Attempting to "
                 "obtain hyper parameters using get_generic_solution. \n";
      mowri << base_ss.str() << "\n\n";

      auto cached_soln = get_generic_cached_solution(graph.constraints_string_in, gg);
      graph.force_start_node(cached_soln.hyperstring);
      hyper_param_start = hyperparams::HyperParams(graph);
      hyper_param_start.checks();
      auto deriveability = derivedparams::get_deriveability(hyper_param_start, gg);
      if (std::get<0>(deriveability) == false)
      {
        mowri << "NOW, THE FALLBACK SOLUTION IS NOT EVEN DERIVEABLE: "
              << hyper_param_start.get_string()
              << " is not deriveable, because : " << std::get<1>(deriveability) << "\n\n";
        throw miog_error("\nfallback solution failed deriveability test in "
                         "get_hyper_param_start/\n");
      }
    }

    return hyper_param_start;
  }

  void update_total_elapsed_seconds()
  {
    auto                         end   = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fp_ms = end - find_start;
    total_elapsed_seconds              = fp_ms.count();
  }

  Solution find(const FindParams& find_params)
  {

    /* TODO : use sumstat */
    float    allotted_time     = find_params.allotted_time;
    size_t allotted_descents = find_params.allotted_descents;

    if (allotted_time <= 0 || allotted_descents == 0)
    {
      std::string k_comment("");
      mowri << "in find with allotted time = " << allotted_time
            << " and allotted_descents = " << allotted_descents << ", returning default" << Endl;
      return get_default(command_queue, constraints_string, gg, k_comment, mowri);
    }

    find_start = std::chrono::high_resolution_clock::now();

    std::vector<Solution> v_tgsolns;

    std::string stars("");

    while (total_elapsed_seconds <= allotted_time && total_elapsed_descents < allotted_descents)
    {

      std::stringstream sss;
      sss << "Entering single_descent_find, at t = " << total_elapsed_seconds << " [s] ( < "
          << allotted_time << " [s]) and iteration = " << total_elapsed_descents << " ( < "
          << allotted_descents << " )";
      std::string titlestring = sss.str();

      stars.resize(titlestring.size(), '*');
      mowri << '\n' << stars << '\n' << titlestring << '\n' << stars << Endl;

      v_tgsolns.emplace_back(
        single_descent_find(allotted_time - total_elapsed_seconds, find_params)); // fst,

      update_total_elapsed_seconds();
      ++total_elapsed_descents;
    }

    float              best_gflops     = 0;
    size_t           best_soln_index = 0;
    std::vector<float> soln_gflops;
    for (size_t si = 0; si < v_tgsolns.size(); ++si)
    {

      float gflops = v_tgsolns[si].statistics.median_benchmark_gflops;
      soln_gflops.push_back(gflops);
      if (gflops > best_gflops)
      {
        best_gflops     = gflops;
        best_soln_index = si;
      }
    }

    std::string header("The gflops found by single descents:");
    stars.resize(header.size(), '*');

    mowri << '\n'
          << "(find finished) elapsed seconds: " << total_elapsed_seconds
          << "    elapsed descents: " << total_elapsed_descents << Endl;
    mowri << header << '\n' << stars << '\n';
    std::sort(soln_gflops.begin(), soln_gflops.end());
    for (auto& x : soln_gflops)
    {
      mowri << x << "  ";
    }
    mowri << "\n\n";
    
    mowri << " -- snip -- -- -- snip --\n" << Endl;
    mowri << v_tgsolns[best_soln_index].get_cache_entry_string();
    mowri << " -- snip -- -- -- snip --" << Endl;


    return v_tgsolns[best_soln_index];
  }

  Solution single_descent_find(float allotted_time, const FindParams& find_params)
  {

    mowri << "geometry : " << gg.get_string() << Endl;

    address_check_valid_and_reliable();

    // we will count how many kernels are successfully generated
    // AND compiled AND benchmarked
    size_t global_counter = 0;

    hyperparams::HyperParams hyper_param_current = get_hyper_param_start(); // fst);

    if (allotted_time <= 0)
    {
      throw miog_error("in single_descent_find with allotted_time <= 0, this "
                       "should never happen (logic error)");
    }

    // In here, we will store all previously considered
    // HyperParams strings, used to check and
    // ensure that we do not consider a HyperParam more than once
    std::vector<std::string> hyper_front_history;

    best_solns_path.clear();

    std::vector<hyperparams::HyperParams> hyper_front = {hyper_param_current};

    bool improvement_found_on_front = true;
    mowri << "allotted time : " << allotted_time << Endl;

    float elapsed_seconds;
    auto  start = std::chrono::high_resolution_clock::now();

    auto update_elapsed_seconds = [&elapsed_seconds, &start]() {
      auto                         end   = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> fp_ms = end - start;
      elapsed_seconds                    = fp_ms.count();
    };

    while (improvement_found_on_front == true)
    {
      update_elapsed_seconds();
      improvement_found_on_front = false;
      size_t hfi               = 0;
      while (hfi < hyper_front.size() && improvement_found_on_front == false &&
             elapsed_seconds < allotted_time)
      {
        hyper_param_current   = hyper_front[hfi];
        std::string hp_string = hyper_param_current.get_string();
        hyper_front_history.push_back(hp_string);

        // extra precaution, should be able to remove this
        deriveability_test(hyper_param_current, "in find loop");

        auto bundle = kerngen::get_bundle(hyper_param_current, gg, mowri, bundle_verbose);
        // the OpenCL string was succesfully generated,
        // we can now attempt to compile and benchmark it
        ++global_counter;

        mowri << "\n[" << global_counter << ", " << std::fixed << std::setprecision(2)
              << elapsed_seconds;
        mowri << std::setprecision(6) << "s]\t" << hyper_param_current.get_string() << Endl;

        auto atr = architests::architecture_specific_tests(
          command_queue, bundle.dp, gg, hyper_param_current);
        if (std::get<0>(atr) == false)
        {
          mowri << "Archtests failed : " << std::get<1>(atr) << Endl;
        }

        else
        {
          // kernel compilation
          auto oclr = setup_tinykernels(hyper_param_current, bundle);

          if (oclr.fail())
          {

            std::stringstream ss;
            ss << "** failed in setup : " << hyper_param_current.get_string()
               << " message : " << oclr.message;
            ss << "error status : " << oclr.success;
            ss << "Is this error related to this thread : "
                  "https://github.com/BVLC/caffe/issues/5610  ? ";
            ss << "  did you see errors about local memory exceeded? That is "
                  "coming from runtime, "
                  "bailing";
            throw miog_error(ss.str());
          }

          else
          {
            // run kernels
            oclr = core_gemm_loop(find_params.max_n_runs_per_kernel, find_params.max_time_per_kernel, false);

            if (oclr.fail())
            {
              mowri << "** failed in core_gemm_loop : " << hyper_param_current.get_string() << Endl;
            }

            else
            {

              // A new best kernel found
              if (best_solns_path.size() == 0 ||
                  median_time < 1.000 * best_solns_path.back().statistics.median_benchmark_time)
              {
                update_elapsed_seconds();
                improvement_found_on_front = true;

                std::time_t generation_time =
                  std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                auto sstats = SolutionStatistics(median_time,
                                                 median_gflops,
                                                 elapsed_seconds,
                                                 std::ctime(&generation_time),
                                                 find_params);
                best_solns_path.emplace_back(gg,
                                             sstats,
                                             bundle.v_tgks,
                                             hyper_param_current.get_string(),
                                             devinfo,
                                             constraints_string);
              }
            }
          }
        }

        ++hfi;
        update_elapsed_seconds();
      }

      if (improvement_found_on_front == true && allotted_time > elapsed_seconds)
      {

        // getting all `one-away's
        auto one_aways = hyper_param_current.get_one_aways();

        // refreshing hyper front
        hyper_front.clear();

        for (auto& hp : one_aways)
        {

          auto hp_string = hp.get_string();

          auto in_graph_tuple = hp.in_graph();
          if (std::count(one_aways.begin(), one_aways.end(), hp) > 1)
          {
            throw miog_error("duplicates in one_aways not allowed, should have already been "
                             "filtered. Could filter out here, but less efficient ");
          }

          else if (std::get<0>(in_graph_tuple) == false)
          {
            std::stringstream errmss;
            errmss << "constraint violators not allowed, should have already been "
                      "filtered. Could "
                      "filter out here, but less efficient. \nThe hyperstring is\n"
                   << hp.get_string();
            errmss << "\nrecall the geometry is\n" << gg.get_string();
            errmss << "\nthe constraint violations string is:\n" << std::get<1>(in_graph_tuple);
            throw miog_error(errmss.str());
          }

          // filtering out if it has already been considered
          else if (std::find(hyper_front_history.begin(), hyper_front_history.end(), hp_string) !=
                   hyper_front_history.end())
          {
          }

          // filtering out non-deriveables
          else if (std::get<0>(derivedparams::get_deriveability(hp, gg)) == false)
          {
          }

          // looks ok, adding it to the hyper-front
          else
          {
            hyper_front.push_back(hp);
          }
        }
      }
    }

    if (allotted_time <= elapsed_seconds)
    {
      mowri << "stopping the search because the allotted time has been surpassed" << Endl;
    }

    else if (improvement_found_on_front == false)
    {
      mowri << "stopping the search because a locally minimal kernel has been "
               "found"
            << Endl;
    }

    else
    {
      throw miog_error("why did the algorithm stop ? ");
    }

    if (best_solns_path.size() == 0)
    {
      throw miog_error("\nThere were no solutions found. This suggests that "
                       "the initial kernel did "
                       "not work (could not derive hyper parameters, required "
                       "too much memory, or "
                       "did not compile. Maybe there is some preceding warning "
                       "printed which sheds "
                       "light on this? Probably with a modification to the "
                       "FindStartType or the "
                       "constraints_string, this should be resolved. For "
                       "example, the unroll UNR "
                       "can be reduced if the problem is memory. jn should "
                       "catch certain problems "
                       "in architests ");
    }

    auto leading_size = best_solns_path.back().hyper_param_string.size() + 2;

    std::string startstring = "hyper parameter string:";
    startstring.resize(leading_size, ' ');
    mowri << '\n' << startstring << "\t time when found:\t median Gflops/s:" << Endl;

    for (auto& x : best_solns_path)
    {
      std::string solnstring = x.get_hyper_param_string();
      solnstring.resize(leading_size, ' ');
      mowri << std::fixed << solnstring << "\t " << x.statistics.solution_discovery_time << "\t\t "
            << x.statistics.median_benchmark_gflops << Endl;
    }

    return best_solns_path.back();
  }
};

cl_mem get_copy(cl_command_queue   command_queue,
                cl_mem             c,
                const Geometry&    gg,
                const Offsets&     toff,
                const std::string& hash)
{
  cl_mem   c_copied;
  cl_event c_copy_event;

  size_t n_c = gg.ldX[Mat::E::C] * (gg.tX[Mat::E::C] == gg.isColMajor ? gg.m : gg.n) + toff.offsets[Mem::E::C];
  size_t c_memsize = gg.derived.float_size_bytes * n_c;
  openclutil::cl_set_buffer_from_command_queue(c_copied,
                                               command_queue,
                                               CL_MEM_READ_WRITE,
                                               c_memsize,
                                               NULL,
                                               hash +
                                                 ", in function get_copy which returns a cl_mem",
                                               true);
  openclutil::cl_enqueue_copy_buffer(command_queue,
                                     c,
                                     c_copied,
                                     0,
                                     0,
                                     c_memsize,
                                     0,
                                     NULL,
                                     &c_copy_event,
                                     hash + ", in function get_copy which returns a cl_mem",
                                     true);

  openclutil::cl_wait_for_events(1, &c_copy_event, "in function find", true);
  return c_copied;
}

cl_mem get_single(cl_command_queue command_queue, const std::string& hash)
{
  size_t c_memsize = 1;
  cl_mem single;
  openclutil::cl_set_buffer_from_command_queue(
    single,
    command_queue,
    CL_MEM_READ_WRITE,
    c_memsize,
    NULL,
    hash + ", in function cl_mem get_single which returns a cl_mem",
    true);
  return single;
}

Solution find(cl_command_queue             command_queue,
              const FindParams&            find_params,
              cl_mem                       a,
              cl_mem                       b,
              cl_mem                       c,
              cl_mem                       workspace,
              const std::string            constraints_string,
              const Geometry&              gg,
              const Offsets&               toff,
              outputwriting::OutputWriter& mowri,
              bool                         c_is_const)
{

  gg.check_ldx_consistent();
  bool full_constraints_expected = false;

  cl_mem                c_to_use(nullptr);
  openclutil::SafeClMem c_copied("to be used in the case that c_is_const");
  if (c_is_const == true)
  {
    c_to_use       = get_copy(command_queue, c, gg, toff, "c_is_const is true, making the copy");
    c_copied.clmem = c_to_use;
  }

  else
  {
    c_to_use = c;
  }

  OpenCLGemmEncapsulator oger(command_queue,
                              gg,
                              toff,
                              a,
                              b,
                              c_to_use,
                              workspace,
                              constraints_string,
                              full_constraints_expected,
                              mowri);
  return oger.find(find_params);
}

std::tuple<bool, std::string> check_for_default(cl_command_queue command_queue,
                                                std::string     constraints_string,
                                                const Geometry& gg,
                                                std::string     k_comment)
{

  openclutil::OpenCLDeviceInfo devinfo(command_queue);
  std::string                  k_dev = devinfo.identifier;
  std::string                  k_con = constraints_string;
  std::string                  k_geo = gg.get_string();

  std::stringstream ss;
  ss << "\nfailed to find cache entry from keys:\n";
  ss << get_cache_keys_string(k_dev, k_con, k_geo, k_comment);

  std::string final_comment(
    "(see tests/gencache.cpp for an example of generating a cache entry)\n");

  if (kernel_cache.count(k_dev) == 0)
  {
    ss << "Unrecognised device identifier in cache.\nMaybe the cache needs to "
          "be built for this "
          "device? \n"
       << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).count(k_con) == 0)
  {
    ss << "Unrecognised constraints_string in cache.\nMaybe the cache needs to "
          "be built with these "
          "constraints? \n"
       << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).at(k_con).count(k_geo) == 0)
  {
    ss << "Unrecognised geometry key (gg.get_string()) in cache.\nMaybe a "
          "cache entry needs to be "
          "generated with this geometry? \n"
       << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).at(k_con).at(k_geo).count(k_comment) == 0)
  {
    ss << "Unrecognised k_comment in cache\n";
    return std::make_tuple(false, ss.str());
  }

  return std::make_tuple(true, "");
}

// fall back solution
Solution get_default(const Geometry& gg)
{
  std::string constraints_string = "";

  auto                         cached_soln = get_generic_cached_solution(constraints_string, gg);
  openclutil::OpenCLDeviceInfo devinfo;
  outputwriting::OutputWriter  mowri(Ver::E::SILENT, "");
  hyperparams::Graph           graph(gg, devinfo, cached_soln.hyperstring, false);
  hyperparams::HyperParams     hp(graph);

  bool bundle_verbose_get_default = true;
  auto bundle                     = kerngen::get_bundle(hp, gg, mowri, bundle_verbose_get_default);

  return {gg, cached_soln.stats, bundle.v_tgks, hp.get_string(), devinfo, constraints_string};
}

Solution get_default(cl_command_queue             command_queue,
                     std::string                  constraints_string,
                     const Geometry&              gg,
                     std::string                  k_comment,
                     outputwriting::OutputWriter& mowri)
{

  openclutil::OpenCLDeviceInfo devinfo(command_queue);

  std::string k_dev = devinfo.identifier;
  std::string k_con = constraints_string;
  std::string k_geo = gg.get_string();

  CachedSolution cached_soln;
  auto           pair = check_for_default(command_queue, constraints_string, gg, k_comment);
  if (std::get<0>(pair) == false)
  {
    miog_warning(std::get<1>(pair));
    mowri << std::get<1>(pair);
    cached_soln = get_generic_cached_solution(constraints_string, gg);
  }

  else
  {
    cached_soln = kernel_cache.at(k_dev).at(k_con).at(k_geo).at(k_comment);
  }

  // generating source files from cache
  hyperparams::Graph       graph(gg, devinfo, cached_soln.hyperstring, false);
  hyperparams::HyperParams hp(graph);
  bool                     bundle_verbose_get_default = true;
  auto                     bundle = kerngen::get_bundle(hp, gg, mowri, bundle_verbose_get_default);

  return {gg, cached_soln.stats, bundle.v_tgks, hp.get_string(), devinfo, constraints_string};
}

void benchgemm(cl_command_queue             command_queue,
               const std::string&           hyperstring,
               size_t                     max_n_runs,
               double                     max_time,
               const Geometry&              gg,
               const Offsets&               toff,
               cl_mem                       a_gpu,
               cl_mem                       b_gpu,
               cl_mem                       c_gpu,
               cl_mem                       workspace_gpu,
               outputwriting::OutputWriter& mowri,
               bool                         c_is_const)
{

  bool full_constraints_expected = true;

  gg.check_ldx_consistent();
  if (c_is_const == true)
  {

    cl_mem c_cop = get_copy(command_queue, c_gpu, gg, toff, "copy of c in benchgemm");
    openclutil::SafeClMem c_copied("copy of c in find");
    c_copied.clmem = c_cop;

    OpenCLGemmEncapsulator oger(command_queue,
                                gg,
                                toff,
                                a_gpu,
                                b_gpu,
                                c_copied.clmem,
                                workspace_gpu,
                                hyperstring,
                                full_constraints_expected,
                                mowri);
    oger.benchgemm(max_n_runs, max_time);
  }

  else
  {

    OpenCLGemmEncapsulator oger(command_queue,
                                gg,
                                toff,
                                a_gpu,
                                b_gpu,
                                c_gpu,
                                workspace_gpu,
                                hyperstring,
                                full_constraints_expected,
                                mowri);

    oger.benchgemm(max_n_runs, max_time);
  }
}

Solution find(float            allotted_time,
              cl_command_queue command_queue,
              cl_mem           a,
              cl_mem           b,
              cl_mem           c,
              bool             enforce_determinism,
              const Geometry&  tgg,
//              bool             verbose,
              bool             with_warnings)
{

  Solution solution = get_default(tgg);

  /* TODO : where is a good place to set this ? */
  float min_time_without_cache = 100.00;

  SummStat::E sumstat(SummStat::E::MEDIAN);
  size_t    allotted_descents = 30;
  size_t    max_n_runs_per_kernel = 3;
  double    max_time_per_kernel = 1000.; // 1000 seconds. 
  
  FindParams  find_params(allotted_time, allotted_descents, max_n_runs_per_kernel, max_time_per_kernel, sumstat);

  cl_mem workspace = nullptr;

  std::string constraints_string = "A_WOS0__B_WOS0"; // no workspace
  if (enforce_determinism == true)
  {
    constraints_string += "__C_ICE1";
  }

  Offsets toff(0, 0, 0, 0, 0, 0, 0, 0);

  outputwriting::OutputWriter mowri(Ver::E::TERMINAL, "");

  bool c_is_const = true;

  std::string k_comment = "";
  auto        pair      = check_for_default(command_queue, constraints_string, tgg, k_comment);

  bool is_custom_cache_entry = std::get<0>(pair);

  mowri << "is_custom_cache_entry = " << is_custom_cache_entry << Endl;
  if (allotted_time < min_time_without_cache)
  {
    mowri << "allotted_time < min_time_without_cache, will not search\n";

    if (is_custom_cache_entry == false)
    {

      std::stringstream ss;
      ss << "\n\n"
         << "In find (version without workspace), and "
         << "\n(1) allotted_time (" << allotted_time << ") is less than min_time_without_cache ("
         << min_time_without_cache << ")  "
         << "\n(2) there is no custom cache entry. The message returned when "
         << "attempting to obtain "
         << "a custom cache entry was,"
         << '\n'
         << std::get<1>(pair) << '\n'
         << "Either "
         << "\n(1) set allotted_time to be greater than "
         << "min_time_without_cache, or "
         << "\n(2) generate a custom cache entry (see tests/gencache.cpp for "
         << "an example)."
         << "\n\nReturing a generic cache entry\n";

      mowri << ss.str();

      if (with_warnings)
        miog_warning("\nvery limited search with no custom cache : expect a "
                     "sub-optimal kernel(s) \n");

      solution = get_default(tgg);
    }

    else
    {
      solution = get_default(command_queue, constraints_string, tgg, k_comment, mowri);
    }
  }

  // we have time to search
  else
  {
    auto found_soln = find(command_queue,
                           find_params,
                           a,
                           b,
                           c,
                           workspace,
                           constraints_string,
                           tgg,
                           toff,
                           mowri,
                           c_is_const);

    if (std::get<0>(pair) == false)
    {
      solution = found_soln;
    }

    else
    {
      auto cached_soln = get_default(command_queue, constraints_string, tgg, k_comment, mowri);
      if (cached_soln.statistics.median_benchmark_gflops >
          found_soln.statistics.median_benchmark_gflops)
      {
        mowri << "cached solution has better glops: "
              << cached_soln.statistics.median_benchmark_gflops << ", returning cached soln"
              << Endl;
        solution = cached_soln;
      }
      else
      {
        mowri << "cached solution has worse glops: "
              << cached_soln.statistics.median_benchmark_gflops
              << ", the new soln will be returned\n"
              << "consider adding this new solution to the cache, it's entry "
              << "string is\n"
              << cached_soln.get_cache_entry_string() << Endl;
        solution = found_soln;
      }
    }
  }

  return solution;
}

}

// comment `overheat'
// This pause should have zero effect but
// mysteriously it smooths out the run times
// between runs when working with certain
// drivers, something to do with overheating

// comment `cl-events'
// copying cl_events is dangerous.
// I have seen that copying them before passed to enqueue
// (last parameter) causes problems,
// this is my idea of what is going on, to confirm:
// from cl.h, we see that
// typedef struct _cl_event *          cl_event,
// that is cl_event is a pointer to a _cl_event.
// when a cl_event address is passed to enqueue,
// the value of it changes. that is it points to a different _cl_event.
// thus ev_a = ev_b, enqueue(..., ev_b)
// leaves ev_a pointing to the wrong (old) place
// checking the event is safe:
// clGetEventInfo takes cl_events by value.
// So the moral of the story is :
// don't copy cl_events before passing their address
// as non-const pointers somewhere!
// paruse cl.h, sometimes *cl_event is passed as const, sometimes not

// comment `in-series'
// if (k_ind == 0){ status = tk_kernels_active[k_ind]->enqueue();}
// else{ status = tk_kernels_active[k_ind]->enqueue(1, &(tk_kernels_active[k_ind -1]->clevent)); }
