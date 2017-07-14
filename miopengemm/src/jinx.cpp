/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>
#include <miopengemm/architests.hpp>
#include <miopengemm/bundle.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/findparams.hpp>
#include <miopengemm/jinx.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/kernelstring.hpp>
#include <miopengemm/oclutil.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/solution.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{


// TODO : find a new home for Timer
class Timer{
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;
  
  public:
  
  void start(){
    t0 = std::chrono::high_resolution_clock::now();
  }
  
  double get_elapsed(){
    std::chrono::duration<double> fp_ms = std::chrono::high_resolution_clock::now() - t0;
    return fp_ms.count();    
  }
  
};


MFType::MFType(double v) : v_d(v), v_f(static_cast<float>(v)) {}
void* MFType::operator[](char floattype) const
{
  return floattype == 'd' ? (void*)(&v_d) : (void*)(&v_f);
}

GpuMms::GpuMms(cl_mem           a_gpu_,
               cl_mem           b_gpu_,
               cl_mem           c_gpu_,
               bool             c_is_const,
               cl_mem           workspace_gpu_,
               size_t           c_nbytes,
               cl_command_queue cq)
{

  cl_mems[Mem::E::A] = a_gpu_;
  cl_mems[Mem::E::B] = b_gpu_;
  cl_mems[Mem::E::W] = workspace_gpu_;

  if (c_is_const == false)
  {
    cl_mems[Mem::E::C] = c_gpu_;
  }

  else
  {
    cl_mems[Mem::E::C] =
      oclutil::get_copy(cq, c_gpu_, c_nbytes, "c_is_const is true, making copy in GpuMms");
    c_copy.clmem = c_gpu_;  // for correct destruction
  }
}

cl_mem& GpuMms::operator[](Mem::E x)
{
  // TODO : bound check here if in debug mode.
  return cl_mems[x];
}

Jinx::Jinx(cl_command_queue command_queue_,
           const Geometry   gg_,
           const Offsets    toff_,
           cl_mem           a_gpu_,
           cl_mem           b_gpu_,
           cl_mem           c_gpu_,
           bool             c_is_const,
           cl_mem           workspace_gpu_,
           owrite::Writer& mowri_)
  :

    command_queue(command_queue_),
    gg(gg_),
    toff(toff_),

    gpum(a_gpu_,
         b_gpu_,
         c_gpu_,
         c_is_const,
         workspace_gpu_,
         get_mat_memsize(gg, toff, Mem::E::C),
         command_queue_),
    devinfo(command_queue_),
    mowri(mowri_)
{

  tk_kernels.resize(BasicKernelType::E::N);
  for (size_t i = 0; i < BasicKernelType::E::N; ++i)
  {
    tk_kernels[i] = Kernel(command_queue, BasicKernelType::M.name[i]);
  }
}

/* TODO : option for median / max / mean */

float Jinx::get_gflops(float timems) { return (2. * gg.m * gg.n * gg.k) / (timems * 10e5); }

void Jinx::address_check_valid()
{
  for (auto x : {Mem::E::A, Mem::E::B})
  {
    if (gpum[Mem::E::C] == gpum[x])
    {
      std::stringstream ss;
      ss << "in address_check_valid, " << Mem::M.name[Mem::E::C] << " and " << Mem::M.name[x]
         << " should have distinct memories, "
         << "otherwise race condition arise (one thread writes its result to "
         << Mem::M.name[Mem::E::C] << "before another one has finished reading from "
         << Mem::M.name[Mem::E::C] << ')';
      throw miog_error(ss.str());
    }
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
      (gpum[Mem::E::W] == gpum[Mem::E::A] || gpum[Mem::E::W] == gpum[Mem::E::B] ||
       gpum[Mem::E::W] == gpum[Mem::E::C]))
  {
    throw miog_error("in address_check_valid, pointer to workspace memory is "
                     "not the nullptr, "
                     "and it is the same as one of the a,b,c pointers ");
  }
}

void Jinx::address_check_valid_and_reliable()
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


void Jinx::set_kern_args(const KernelType& type)
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

bool Jinx::refresh_needed(BasicKernelType::E type, const HyPas& new_hp, const DerivedParams& new_dp)
{

  /* TODO : check (here) hyper parameters to see if needed anew */

  if (type == BasicKernelType::E::BETAC)
  {
    if (tk_kernels.at(BasicKernelType::E::BETAC).is_set() == false &&
        new_dp.main_does_beta_c_inc == 0)
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
    if (tk_kernels.at(BasicKernelType::E::WSA).is_set() == false &&
        new_hp.sus[Mat::E::A].vs[Chi::E::WOS] != Scratch::E::UNUSED)
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
    if (tk_kernels.at(BasicKernelType::E::WSB).is_set() == false &&
        new_hp.sus[Mat::E::B].vs[Chi::E::WOS] != Scratch::E::UNUSED)
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

oclutil::Result
Jinx::refresh_kernel(const KernelString& ks, const HyPas& hp, const DerivedParams& dp)
{

  oclutil::Result oclr;
  auto            type = ks.type;
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

oclutil::Result Jinx::setup_tinykernels(const HyPas& hp, const kerngen::Bundle& bundle)
{

  oclutil::Result oclr;

  // TODO : make enum so accessible to end-users
  v_wait_indices = bundle.v_wait_indices;
  tk_kernels_active.resize(0);

  for (size_t ksi = 0; ksi < bundle.v_tgks.size(); ++ksi)
  {

    BasicKernelType::E basic = bundle.v_tgks[ksi].type.basic_kernel_type;
    oclr                     = refresh_kernel(bundle.v_tgks[ksi], hp, bundle.dp);
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


std::string Jinx::get_run_times_heading()
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



void Jinx::mowri_tracker_print()
{

  std::cout << "in mowri_tracker_print. " << total_elapsed_seconds << "  " << total_kernels_tested
            << "." << std::endl;
  std::stringstream comment_string_ss;
  comment_string_ss << "[ TOTAL TIME:"
                    << stringutil::get_padded(static_cast<int>(total_elapsed_seconds), 7);
  comment_string_ss << "  #RESTARTS:" << stringutil::get_padded(total_elapsed_descents, 7);
  comment_string_ss << "  #GEMMS CONSIDERED:" << stringutil::get_padded(total_kernels_tested, 7)
                    << "]       ";

  old_comment_string = new_comment_string;
  new_comment_string = comment_string_ss.str();

  std::string backspaces = std::string(old_comment_string.size(), '\b');

  /* TODO : determine where to use mowri_tracker,
   * and enable to path to here */

  // mowri.tracker << backspaces << new_comment_string << Flush;
}



std::string Jinx::get_run_time_string(cl_int status, double extime){
  std::stringstream ss;
  if (status == CL_SUCCESS)
  {
    ss << std::fixed << std::setprecision(3) << extime << '\t';
    for (size_t k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind)
    {
      ss << " " << tk_kernels_active[k_ind]->v_times.back() << "\t";
    }
    ss << " " << 2.0 * gg.m * gg.n * gg.k / (extime * 1e6) << std::setprecision(6);
  }

  else
  {
    ss << "(failed run)";
  }
  return ss.str();
}  

oclutil::Result Jinx::true_core(std::function<void(double, std::string)> acton, const Halt & hl){
  
  
  size_t runi{0};
  oclutil::Result oclr;
  
  Timer timer;
  timer.start();
  
  while (!hl.halt(runi, timer.get_elapsed()))
  {
    // see `overheat' comment at bottom

    for (size_t k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind)
    {
      // At this point, the kernel has been succesfully compiled,
      // but it is still possible that it does not run. We catch that here.
      // if anything is caught here, consider testing for it in architests.

      std::vector<cl_event> clevent_waits;
      for (auto& evi : v_wait_indices[k_ind]) 
      {
        // see `cl-events' comment at bottom
        clevent_waits.emplace_back(tk_kernels_active[evi]->clevent);
      }

      const cl_event* event_wait_list = clevent_waits.size() == 0 ? nullptr : clevent_waits.data();
      oclr = tk_kernels_active[k_ind]->enqueue(clevent_waits.size(), event_wait_list);

      // see `in-series' comment at bottom

      if (oclr.success == CL_SUCCESS){
        // good 
      }
      
      else if (oclr.success == CL_OUT_OF_RESOURCES)
      {
        oclutil::cl_flush(command_queue, "cl flushing in core gemm loop", true);
        oclr.message += " (CL_OUT_OF_RESOURCES in true_core) ";
        return oclr;
      }
      

      else
      {
        std::stringstream ss;
        ss << "OpenCL error status : " << oclr.success << ". "
           << "Neither CL_SUCCESS nor CL_OUT_OF_RESOURCES.  "
           << "Maybe there are no kernels? Internal logic error. "
           << "could catch with CL_OUT_OF_RESOURCES (ie throw oclr) "
           << "The error from opencl was " << oclr.message;
        throw miog_error(ss.str());
      }
    }

    oclutil::cl_flush(command_queue, "cl flush in core gemm loop", true);

    // Wait for kernels to complete
    oclutil::cl_wait_for_events(1, &(tk_kernels_active.back()->clevent), "core gemm loops", true);



    for (auto& ptr_tk_kernel : tk_kernels_active)
    {
      ptr_tk_kernel->update_times();
    }
    //// end time of last kernel - start time of first kernel
    //v_t_total.push_back
    
    double extime  = (1e-6 * (tk_kernels_active.back()->t_end - tk_kernels_active[0]->t_start));
  

    
    //act on the results string. 
    acton(extime, get_run_time_string(oclr.success, extime));
    ++runi;
  }
  return {};
} 
   
     


void Jinx::benchgemm(const HyPas& hp, const Halt & hl)
{

  address_check_valid();
  Derivabilty dblt(hp, gg);
  if (dblt.is_derivable == false)
  {
    throw miog_error("Non-derivable in benchgemm : " + dblt.msg);
  }

  bool bundle_verbose = false;
  auto bundle = kerngen::get_bundle(hp, gg, mowri, bundle_verbose);
  
  architests::Stat atr(command_queue, bundle.dp, gg, hp);
  if (!atr.is_good)
  {
    throw miog_error(atr.msg);
  }

  auto oclr = setup_tinykernels(hp, bundle);
  if (oclr.fail())
  {
    throw miog_error(oclr.message);
  }

  mowri << "(benchgemm) hp   :" << hp.get_string() << '\n'
  << "(benchgemm) geometry  \t:" << gg.get_string() << '\n'
  << "Entering the core gemm loops" << Endl;
        
  mowri << get_run_times_heading();
  true_core([this](double a, std::string x){(void)a, mowri << x << '\n';}, hl);
}

Solution Jinx::find(const Constraints& constraints, const FindParams& fparms)
{
  
  address_check_valid_and_reliable();


  Timer timer;
  timer.start();
  std::vector<Solution> v_solns;
  size_t n_descents = 0;

  while (!fparms.hl_outer.halt(n_descents, timer.get_elapsed())){

    mowri << "\nEntering new descent. \n"
     << fparms.hl_outer.get_status(n_descents, timer.get_elapsed()) << '\n';

    double allotted_sd = std::max(0.1, fparms.hl_outer.max_time - timer.get_elapsed());
    auto soln = single_descent_find(allotted_sd, constraints, fparms.hl_core, fparms); // fparms here hacked on
    v_solns.emplace_back(soln);
    ++n_descents;
  }

  float              best_gflops     = 0;
  size_t             best_soln_index = 0;
  std::vector<float> soln_gflops;
  for (size_t si = 0; si < v_solns.size(); ++si)
  {

    float gflops = v_solns[si].statistics.median_benchmark_gflops;
    soln_gflops.push_back(gflops);
    if (gflops > best_gflops)
    {
      best_gflops     = gflops;
      best_soln_index = si;
    }
  }


  mowri      << "\ntotal elapsed seconds: " << total_elapsed_seconds
        << "\ntotal elapsed descents: " << total_elapsed_descents
  << '\n' << stringutil::get_star_wrapped("The gflops found by single descents:") << '\n';
  
  std::sort(soln_gflops.begin(), soln_gflops.end());
  for (auto& x : soln_gflops)
  {
    mowri << x << "  ";
  }
  mowri << "\n\n";

  mowri << " -- snip -- -- -- snip --\n" << Endl;
  mowri << v_solns[best_soln_index].get_cache_entry_string();
  mowri << " -- snip -- -- -- snip --" << Endl;

  return v_solns[best_soln_index];
}

Solution Jinx::single_descent_find(float              allotted_time,
                                   const Constraints& constraints,
                                   const Halt&  core_halt,
                                   const FindParams & fps) // TODO : fp only needed to push onto solution path: should not be needed here
{


  Timer timer;
  timer.start();

  mowri << "geometry : " << gg.get_string() << "\nallotted time : " << allotted_time << Endl;


  // re-creating the same graph for every single_descent is currently wasteful, 
  // but maybe in the future different constraints will be passed on each run
  const Graph graph(gg, devinfo, constraints, mowri);

  // number of kernels whose strings are generated
  size_t single_descent_counter = 0;

  // We will store all previously considered HyPas, used to check and
  // ensure that we do not consider a HyperParam more than once
  // TODO : maybe this should be in the outer find loop. 
  // Although then the stats between runs are no longer independent... 
  std::vector<HyPas> hyper_front_history;

  // Keep track of the `world records' as they get broken
  std::vector<Solution> best_solns_path;



  std::vector<float> v_t_total;
  float              median_time;
  float              median_gflops;


  // the hyper params to be considered on a single wave 
  // TODO: what if I put 2 or three here ? might help fast escape from bad region
  std::vector<HyPas> hyper_front = {graph.get_random_valid_start()};
  
  
  HyPas hyper_param_current = hyper_front[0]; // NULL initialisation
  
  bool improvement_found_on_front = true;

  while (improvement_found_on_front == true)
  {
    improvement_found_on_front = false;
    size_t hfi                 = 0;
   
    while (hfi < hyper_front.size() && improvement_found_on_front == false &&
           timer.get_elapsed() < allotted_time)
    {
      
      
      hyper_param_current = hyper_front[hfi];
      
      hyper_front_history.push_back(hyper_param_current);

      // extra precaution, should be able to remove this
      Derivabilty dblt(hyper_param_current, gg);
      if (dblt.is_derivable == false)
      {
        throw miog_error("Non-derivable in single descent find : " + dblt.msg);
      }

    // TODO : bundle verbose must go
    bool bundle_verbose = true;
      auto bundle = kerngen::get_bundle(hyper_param_current, gg, mowri, bundle_verbose);
      // the OpenCL string was succesfully generated,
      // we can now attempt to compile and benchmark it
      ++single_descent_counter;

      mowri << "\n[" << single_descent_counter << ", " << std::fixed << std::setprecision(2)
            << timer.get_elapsed() << std::setprecision(6) << "s]\t"
            << hyper_param_current.get_string() << Endl;

      architests::Stat atr(command_queue, bundle.dp, gg, hyper_param_current);
      if (atr.is_good == false)
      {
        mowri << "architest failed: " << atr.msg << Endl;
        ++hfi;
        continue;
      }

      // kernel compilation
      auto oclr = setup_tinykernels(hyper_param_current, bundle);
      if (oclr.fail())
      {
        std::stringstream ss;
        ss << "Failed in setup. " << hyper_param_current.get_string()
           << " Message: " << oclr.message <<  "Error status : " << oclr.success
           << " Maybe related : https://github.com/BVLC/caffe/issues/5610  (?) "
           << " Maybe a runtime error? Bailing, please report if possible ";
        throw miog_error(ss.str());
      }

      mowri_tracker_print();
      v_t_total.resize(0);
      for (auto& ptr_tk_kernel : tk_kernels_active)
      {
        ptr_tk_kernel->reset_times();
      }
      std::vector<std::string> summary;
      oclr = true_core([&summary, &v_t_total](double a, std::string x){
        v_t_total.push_back(a);
        summary.push_back(x);
      }, core_halt);

      if (oclr.fail()){
        mowri << "cl out of resources: " << atr.msg << Endl;
        ++hfi;
        continue;        
      }       



      auto v_t_total_copy = v_t_total;
      
      std::sort(v_t_total_copy.begin(), v_t_total_copy.end());
      // Taking the fastest or median? [v_t_total.size()/2]
      median_time   = v_t_total_copy[0];
      median_gflops = get_gflops(median_time);
      
    


      mowri << get_run_times_heading();
      for (size_t ir = 0; ir < summary.size(); ++ir)
      {
        mowri << summary[ir];
        if (median_time == v_t_total[ir])
        {
          mowri << " (median) ";
          if (best_solns_path.size() > 0 &&
             (best_solns_path.back().statistics.median_benchmark_time >= median_time))
          {
            mowri << " (NEW BEST) ";
            improvement_found_on_front = true;
  
            std::time_t generation_time =
              std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            auto sstats = SolutionStatistics(median_time,
                                             median_gflops,
                                             timer.get_elapsed(),
                                             std::ctime(&generation_time),
                                             fps); //TODO : here is the fps which should be here
            best_solns_path.emplace_back(
              gg,
              sstats,
              bundle.v_tgks,
              hyper_param_current.get_string(),
              devinfo,
              constraints.get_r_str());  // TODO : should rather be constraints NOT string

          }
        }
        mowri << '\n';
      }
      ++total_kernels_tested;
    }

    if (improvement_found_on_front == true && allotted_time > timer.get_elapsed())
    {
      // TODO : change name from one_aways
      auto one_aways = graph.get_neighbors(hyper_param_current);

      // refreshing hyper front
      hyper_front.clear();

      for (auto& hp : one_aways)
      {
        if (std::count(one_aways.begin(), one_aways.end(), hp) > 1)
        {
          throw miog_error("duplicates in one_aways not allowed, should have already been "
                           "filtered. Could filter out here, but less efficient ");
        }

        else if (graph.contains(hp) == false)
        {
          std::stringstream errmss;
          errmss << "constraint violators not allowed, should have already been filtered out."
           << "Could filter out here, but less efficient. The hyperstring is\n" << hp.get_string();
          throw miog_error(errmss.str());
        }

        // filtering out if it has already been considered
        else if (std::find(hyper_front_history.begin(), hyper_front_history.end(), hp) !=
                 hyper_front_history.end())
        {
        }

        // filtering out non-deriveables
        else if (!is_dvble(hp, gg))
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

  if (timer.get_elapsed() >= allotted_time)
  {
    mowri << "stopping the search because allotted time has been surpassed" << Endl;
  }

  else if (improvement_found_on_front == false)
  {
    mowri << "stopping the search because a locally minimal kernel has been found" << Endl;
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
