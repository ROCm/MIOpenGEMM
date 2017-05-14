#include <thread>
#include <limits>
#include <chrono>
#include <sstream>
#include <tuple>
#include <vector> 
#include <algorithm>
#include <map>
#include <iomanip>

#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/consistencychecks.hpp>
#include <tinygemm/outputwriter.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/openclutil.hpp>
#include <tinygemm/tinygemmsolution.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/bundle.hpp>
#include <tinygemm/tinygemmkernel.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/architests.hpp>
#include <tinygemm/kernelstring.hpp>
#include <tinygemm/tinygemmkernelcache.hpp>
#include <tinygemm/tinygemmfindparams.hpp>


namespace tinygemm{

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

class TinyGemmGPUMems{
  private:
    cl_mem a_gpu;
    cl_mem b_gpu;
    cl_mem c_gpu;
    cl_mem workspace_gpu;
  
  public:
    TinyGemmGPUMems(cl_mem a_gpu_, cl_mem b_gpu_, cl_mem c_gpu_, cl_mem workspace_gpu_):a_gpu(a_gpu_), b_gpu(b_gpu_), c_gpu(c_gpu_), workspace_gpu(workspace_gpu_) {}

    cl_mem & operator[](char x){
      if (x == 'a'){
        return a_gpu;
      }
      else if (x == 'b'){
        return b_gpu;
      }
      else if (x == 'c'){
        return c_gpu;
      }
      else if (x == 'w'){
        return workspace_gpu;
      }
      
      else{
        throw tinygemm_error(std::string("unrecognised char passed to operator[] of TinyGemmGPUMems. Should be one of a,b,c,w, not ") + x);
      }
    }
};




class OpenCLGemmEncapsulator{

public:   

  cl_command_queue command_queue;
  std::string outputfilename;
  const tinygemm::TinyGemmGeometry gg;
  const tinygemm::TinyGemmOffsets toff;
  TinyGemmGPUMems gpum;
  const openclutil::OpenCLDeviceInfo devinfo;
  std::string constraints_string;
  const hyperparams::Graph graph;


private:
  outputwriting::OutputWriter & mowri;
  /* vector of times over a set of runs on core loop */
  std::vector<float> v_t_total;
  float median_time;
  float median_gflops;
  /* (for find) while generating, compiling and benchmarking kernels, we will keep track of the fastest found thus far */
  std::vector<tinygemm::TinyGemmSolution> best_solns_path;
  std::vector<TinyGemmKernel> tk_kernels;  
  std::vector<TinyGemmKernel *> tk_kernels_active;  
  std::vector<std::vector <unsigned > > v_wait_indices;
  bool bundle_verbose = false;

  
public:
  OpenCLGemmEncapsulator(
  cl_command_queue command_queue_, 
  const tinygemm::TinyGemmGeometry gg_,
  const tinygemm::TinyGemmOffsets toff_,
  cl_mem a_gpu_,
  cl_mem b_gpu_, 
  cl_mem c_gpu_,
  cl_mem workspace_gpu_,
  std::string constraints_string_,
  bool full_constraints_expected,
  outputwriting::OutputWriter & mowri_):
  
  command_queue(command_queue_), 
  gg(gg_),
  toff(toff_),
  gpum(a_gpu_, b_gpu_, c_gpu_, workspace_gpu_),

  devinfo(command_queue_),
  constraints_string(constraints_string_),
  graph(gg, devinfo, constraints_string_, full_constraints_expected),

  mowri(mowri_)
  {
    
    tk_kernels.resize(nBasicKernelTypes);
    for (unsigned i = 0; i < nBasicKernelTypes; ++ i){
      tk_kernels[i] = TinyGemmKernel(command_queue, basic_kernel_type_strings[i]);
    }
  
    run_checks();    
  }
  
private:

  /* TODO : option for median / max / mean */
  void set_medians(){
    
    auto v_t_total_copy = v_t_total;
    
    std::sort(v_t_total_copy.begin(), v_t_total_copy.end());
    /* Taking the fastest or median? */ 
    median_time = v_t_total_copy[0];//[v_t_total.size()/2]; 
    median_gflops = get_gflops(median_time);
  }
  
  float get_gflops(float timems){
    return  (2. * gg.m * gg.n * gg.k) / (timems * 10e5);  
  }

  void address_check_valid(){
    if (gpum['c'] == gpum['a'] || gpum['c'] == gpum['b']){
      throw tinygemm_error("c should be distinct from a and b for gemm, otherwise race condition arises (one thread writes its result to c before another one has finished reading from c)");
    }
    
    if (gpum['c'] == nullptr){
      throw tinygemm_error("c should not be nullptr");
    }
    
    if (gpum['w'] == nullptr && gg.workspace_size != 0){
      throw tinygemm_error("pointer to workspace memory is the nullptr, but workspace_size is not zero");
    }
    
    if (gpum['w'] != nullptr && gg.workspace_size == 0){
      throw tinygemm_error("pointer to workspace memory is not the nullptr, but workspace_size is zero. if workspace_size is zero please set workspace_gpu to the nullptr to make super clear that there will be no workspace used ");      
    }
    
    if (gpum['w'] != nullptr && (gpum['w'] == gpum['a'] || gpum['w'] == gpum['b'] || gpum['w'] == gpum['c'] ) ){
      throw tinygemm_error("pointer to workspace memory is not the nullptr, and it is the same as one of the a,b,c pointers ");
    }
  }
  
  void address_check_valid_and_reliable(){
    address_check_valid();
    if (gpum['a'] == gpum['b']){
      throw tinygemm_error( "a and b are the same. this will effect kernel run time, not sure if this should be allowed so throwing"); 
    }
  }


  void run_checks(){
    sizingup::check_sizes_ok_for_unsigned(gg, toff);
  }  

    
  void set_kern_args(const KernelType & type){

    /* parameter order rule: {a, oa, b, ob, c, oc, ws, ows}, alpha, beta */
    std::vector<std::pair<size_t, const void *> > arg_sizes_values;
    
    
    for (auto & x : {'a', 'b', 'c', 'w'}){
      if (type.uses(x) == true){
        arg_sizes_values.emplace_back(sizeof(cl_mem), (void *)&(gpum[x]));
        arg_sizes_values.emplace_back(sizeof(unsigned), &(toff[x]));
      }
    }
    
    
    if (type.uses_alpha){
      arg_sizes_values.emplace_back(gg.derived.float_size_bytes, m_alpha[gg.floattype]);
    }
    
    
    if (type.uses_beta){
      arg_sizes_values.emplace_back(gg.derived.float_size_bytes, m_beta[gg.floattype]);      
    }
    
    tk_kernels.at(type.basic_kernel_type).set_kernel_args(arg_sizes_values);
  }


  bool refresh_needed(bkt type, const hyperparams::HyperParams & new_hp, const derivedparams::DerivedParams & new_dp){
    
    /* TODO : check (here) hyper parameters to see if needed anew */
            
    if (type == bkt::betac){
       if (tk_kernels.at(bkt::betac).is_set() == false && new_dp.main_does_beta_c_inc == 0){
         return true;
       }
       else{
         return false;
       }
    }
    
    else if (type == bkt::main){
      return true;
    }
    
    else if (type == bkt::wsa){
      if (tk_kernels.at(bkt::wsa).is_set() == false && new_hp.at(nsHP::matA).vs[nsHP::WOS] != 0){
         return true;
       }
       else{
         return false;
       }
    }

    else if (type == bkt::wsb){
      if (tk_kernels.at(bkt::wsb).is_set() == false && new_hp.at(nsHP::matB).vs[nsHP::WOS] != 0){
         return true;
       }
       else{
         return false;
       }
    }

    else{
      throw tinygemm_error("what is the type of this kernel? Don't recognise it : " + type);
    }
  }


  
  void refresh_kernel(const KernelString & ks, const hyperparams::HyperParams & hp, const derivedparams::DerivedParams & dp){

    auto type = ks.type;
    if (refresh_needed(type.basic_kernel_type, hp, dp) == true){
      tk_kernels.at(type.basic_kernel_type).update(ks, mowri);
      set_kern_args(type);
    }
  }


  void setup_tinykernels(const hyperparams::HyperParams & hp, const kerngen::Bundle & bundle ){
    
    v_wait_indices = bundle.v_wait_indices;
    tk_kernels_active.resize(0);
    
    for (unsigned ksi = 0; ksi < bundle.v_tgks.size(); ++ksi){

      bkt basic = bundle.v_tgks[ksi].type.basic_kernel_type;
      refresh_kernel(bundle.v_tgks[ksi], hp, bundle.dp);
      tk_kernels_active.push_back(&tk_kernels[basic]);
    }
  }
  


  void update_run_times(cl_int status){
    
    if (status == CL_SUCCESS){
      for (auto & ptr_tk_kernel : tk_kernels_active){
        ptr_tk_kernel->update_times();
      }
      /* end time of last kernel - start time of first kernel */
      v_t_total.push_back(1e-6*(tk_kernels_active.back()->t_end - tk_kernels_active[0]->t_start));
    }
    
    else{
      v_t_total.push_back(std::numeric_limits<float>::max());
    }
  }

  std::string get_run_times_heading(){
    std::stringstream ss;
    ss << "tt: \t";
    for (unsigned k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind){
      ss << " k" << k_ind <<  ":\t";
    }
    ss << " Gflops/s:\n";    
    return ss.str();
  }
  
  std::string get_run_time_string(cl_int status){
    std::stringstream ss;
    if (status == CL_SUCCESS){      
      ss << std::fixed << std::setprecision(3) << v_t_total.back() << "\t";
      for (unsigned k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind){
        ss << " " << tk_kernels_active[k_ind]->v_times.back() << "\t";
      }
      ss  << " " << 2.0 * gg.m * gg.n * gg.k / (v_t_total.back() * 1e6) << std::setprecision(6); 
    }

    else{
      ss << "(failed run)";
    }
    return ss.str();
  }


  void reset_v_times(){
    v_t_total.resize(0);
    for (auto & ptr_tk_kernel : tk_kernels_active){
      ptr_tk_kernel->reset_times();
    }
  }
  
  
  void core_gemm_loop(size_t n_runs, bool print_asap){
    
    reset_v_times();
    
    std::vector<std::string> indi_run_strings;

    if (print_asap == true){
      mowri << get_run_times_heading();
    }
      


    for (size_t kqq = 0; kqq < n_runs; ++kqq){

      /* This pause should have zero effect but
       * mysteriously it smooths out the run times 
       * between runs when working with certain 
       * drivers, something to do with overheating  */        
      if (n_runs > 1){
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
      }
  
      /* ***************************************************************************************
       *  Note on timing : the same results have been obtained whth timespec and std::chrono   *
       *  **************************************************************************************/
      
      int status = 10111; 

      for (unsigned k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind){
      /* At this point, the kernel has been succesfully compiled, 
       * but it is still possible that the resources necessary (LDS etc) are
       * not sufficient on this machine. We catch this case here. 
       * TODO : architests can go some way to catching these before compilation */        
        

        std::vector<cl_event> clevent_waits;

        for (auto & evi : v_wait_indices[k_ind]){
          /* copying cl_events is dangerous. 
           * I have seen that copying them before passed to enqueue 
           * (last parameter) causes problems,
           * this is my idea of what is going on, to confirm: 
           * from cl.h, we see that
           * typedef struct _cl_event *          cl_event,
           * that is cl_event is a pointer to a _cl_event. 
           * when a cl_event address is passed to enqueue,
           * the value of it changes. that is it points to a different _cl_event.
           * thus ev_a = ev_b, enqueue(..., ev_b) 
           * leaves ev_a pointing to the wrong (old) place 
           * checking the event is safe:
           * clGetEventInfo takes cl_events by value. 
           * So the moral of the story is : 
           * don't copy cl_events before passing their address  
           * as non-const pointers somewhere!
           * paruse cl.h, sometimes *cl_event is passed as const, sometimes not
           *  */
          clevent_waits.emplace_back(tk_kernels_active[evi]->clevent);
        }
        
        size_t num_events_int_wait_list = clevent_waits.size();
        const cl_event * event_wait_list = num_events_int_wait_list == 0 ? nullptr : clevent_waits.data();
        status = tk_kernels_active[k_ind]->enqueue(num_events_int_wait_list, event_wait_list);

        ////* the in series solution */  
        //if (k_ind == 0){ 
          //status = tk_kernels_active[k_ind]->enqueue();
        //}
        
        //else{
          //status = tk_kernels_active[k_ind]->enqueue(1, &(tk_kernels_active[k_ind - 1]->clevent));
        //}


        if (status == CL_OUT_OF_RESOURCES){
          /* Set the run time(s) and append to vectors */
          mowri << "kernel could not be enqueued, status returned from clEnqueueNDRangeKernel was CL_OUT_OF_RESOURCES (" <<tk_kernels_active[k_ind]->hash << ")" << Endl;
          break;
        }
      }
      
      openclutil::cl_flush(command_queue, "cl flushing in core gemm loop");      
      
      
      if (status == CL_SUCCESS){
        /* Wait for kernels to complete */
        openclutil::cl_wait_for_events(1, &(tk_kernels_active.back()->clevent), "with status == CL_SUCCESS in core gemm loops");
      }

      else if (status == CL_OUT_OF_RESOURCES){
        //
      }
      
      else{
        throw std::logic_error("How can this not be CL_SUCCESS or CL_OUT_OF_RESOURCES? Algo prob, come fix. Is status 10111? Maybe there are no kernels?");
      }
      
      update_run_times(status);
      indi_run_strings.push_back(get_run_time_string(status));
      if (print_asap == true){
        mowri << indi_run_strings[kqq] << "\n";
      }

    }
    
    set_medians();
    
    
    if (print_asap == false){
      mowri << get_run_times_heading();
      for (size_t kqq = 0; kqq < n_runs; ++kqq){
        mowri << indi_run_strings[kqq];
        if (median_time == v_t_total[kqq]){
          mowri << " (median) ";
          if (best_solns_path.size() > 0 && (best_solns_path.back().statistics.median_benchmark_time >= median_time)){
            mowri << " (NEW BEST) ";
          }
        }
        mowri << "\n";
      }
    }
  }

  void deriveability_test(const hyperparams::HyperParams & hp, const std::string & hash){
    auto deriveability = derivedparams::get_deriveability(hp, gg);      
    if (std::get<0>(deriveability) == false){
      throw tinygemm_error(hash + ": the hyper parameters in benchgemm are not consistent, specifically, from get_deriveability \n" + std::get<1>(deriveability));
    }
  }

public:
  void benchgemm(unsigned n_runs){

    address_check_valid();
    
    if (n_runs == 0){
      throw tinygemm_error("n_runs to benchgemm should be a positive integer");
    }
   
    hyperparams::HyperParams hp(graph);
    
    deriveability_test(hp, "in benchgemm");
    
    auto bundle = tinygemm::kerngen::get_bundle(hp,gg, mowri, bundle_verbose); 
    auto atr = architests::architecture_specific_tests(command_queue, hp, bundle.dp);
    
    if (std::get<0>(atr) == false){
      throw tinygemm_error(std::get<1>(atr));
    }

    setup_tinykernels(hp, bundle); 
    
    mowri << "(benchgemm) geometry  \t:" << gg.get_string()  << "\nEntering the core gemm loops" << Endl;
    core_gemm_loop(n_runs, true);

  }
  


  hyperparams::HyperParams get_hyper_param_start(){

  
  hyperparams::HyperParams hyper_param_start(graph);
  hyper_param_start.checks();  
  

    bool found_a_deriveable_hp = false;
    unsigned deriveable_search_iteration = 0;
    std::stringstream deriveablesearch_ss;

    
    /* the number of attempts at finding a deriveable HyperParams given the constraint string */
    const unsigned n_trials = 10000;
    
    while (found_a_deriveable_hp == false && deriveable_search_iteration < n_trials){
      
      hyper_param_start = hyperparams::HyperParams(graph);
      hyper_param_start.checks();  
      
      auto deriveability = derivedparams::get_deriveability(hyper_param_start, gg);
      if (std::get<0>(deriveability) == false){
        deriveablesearch_ss << hyper_param_start.get_string() << " is not deriveable, because " << std::get<1>(deriveability) << "\n";            
      }
      else{
        found_a_deriveable_hp = true;
      }
      ++deriveable_search_iteration;
    }
    
    /* TODO : should rather return null-solution, as throwing an error should not be stochastic */
    if (found_a_deriveable_hp == false){
      std::stringstream base_ss;
      base_ss << "\n\nStruggling to find a deriveable set of hyper parameters which satisfy the geometry and constraints. The number of attempts made is " << n_trials << "\n throwing an error. To view the full output of the hyper parameters tried and their reasons for not being derivable, modify the code here (add deriveablesearch_ss.str()). \n";
      throw tinygemm_error(base_ss.str());
    }
    return hyper_param_start;
  }

  
  
  tinygemm::TinyGemmSolution find(const FindParams & find_params){
  
    /* TODO : use sumstat */
    float allotted_time = find_params.allotted_time;
    unsigned allotted_descents = find_params.allotted_descents;
    
    if (allotted_time <= 0 || allotted_descents == 0){
      std::string k_comment("");
      mowri << "in find with allotted time = " << allotted_time << " and allotted_descents = " << allotted_descents << ", returning default" <<  Endl;
      return get_default(command_queue, constraints_string, gg, k_comment, mowri);
    }
    
    float elapsed_seconds = 0;
    unsigned elapsed_descents = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<tinygemm::TinyGemmSolution> v_tgsolns;

    std::string stars("");    
    while (elapsed_seconds <= allotted_time && elapsed_descents < allotted_descents){
      
      std::stringstream sss;
      sss << "Entering single_descent_find, at t = " << elapsed_seconds << " [s] ( < " << allotted_time << " [s]) and iteration = " << elapsed_descents << " ( < " << allotted_descents << " )";
      std::string titlestring = sss.str();

      stars.resize(titlestring.size(), '*');
      mowri << "\n" << stars << "\n" << titlestring << "\n" << stars << Endl;
      
      v_tgsolns.emplace_back(single_descent_find(allotted_time - elapsed_seconds, find_params)); //fst, 
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> fp_ms = end - start;
      elapsed_seconds = fp_ms.count();          
      ++elapsed_descents;
    }
    
    float best_gflops = 0;
    unsigned best_soln_index = 0;
    std::vector<float> soln_gflops;
    for (unsigned si = 0; si < v_tgsolns.size(); ++si){
      
      float gflops = v_tgsolns[si].statistics.median_benchmark_gflops;
      soln_gflops.push_back(gflops);
      if (gflops > best_gflops){
        best_gflops = gflops;
        best_soln_index = si;
      }
      
    }
    
    std::string header("The gflops found by single descents:");
    stars.resize(header.size(), '*');
    
    mowri << "\n" << "(find finished) elapsed seconds: " << elapsed_seconds << "    elapsed descents: " <<  elapsed_descents << Endl;
    mowri << header << "\n" << stars << "\n"; 
    std::sort(soln_gflops.begin(), soln_gflops.end());
    for (auto & x : soln_gflops){
      mowri << x << "  ";
    }
    mowri << "\n\n";
        
    return v_tgsolns[best_soln_index];
    

  }
  
  tinygemm::TinyGemmSolution single_descent_find(float allotted_time, const FindParams & find_params){ //, FindStartType fst
              

    mowri << "geometry : " << gg.get_string()  << Endl;

    address_check_valid_and_reliable();

    /* we will count how many kernels are successfully generated AND compiled AND benchmarked */
    unsigned global_counter = 0;

    hyperparams::HyperParams hyper_param_current = get_hyper_param_start();//fst);


    if (allotted_time <= 0){
      throw tinygemm_error("in single_descent_find with allotted_time <= 0, this should never happen (logic error)");
    }

    /* In here, we will store all previously considered HyperParams strings, used to check and ensure that we do not consider a HyperParam more than once */
    std::vector<std::string> hyper_front_history;
 
    best_solns_path.clear();
    
    std::vector<hyperparams::HyperParams> hyper_front = { hyper_param_current };

    bool improvement_found_on_front = true;    
    mowri << "allotted time : " << allotted_time << Endl;

    float elapsed_seconds;
    auto start = std::chrono::high_resolution_clock::now();

    auto update_elapsed_seconds  =  [&elapsed_seconds, &start]()  {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> fp_ms = end - start;
      elapsed_seconds = fp_ms.count();          
    };
    
    
    while (improvement_found_on_front == true){
      update_elapsed_seconds();
      improvement_found_on_front = false;
      unsigned hfi = 0;
      while (hfi < hyper_front.size() && improvement_found_on_front == false && elapsed_seconds < allotted_time){
        hyper_param_current = hyper_front[hfi];
        std::string hp_string = hyper_param_current.get_string();        
        hyper_front_history.push_back(hp_string);

        /* extra precaution, should be able to remove this */
        deriveability_test(hyper_param_current, "in find loop");     
        
           
        auto bundle = tinygemm::kerngen::get_bundle(hyper_param_current,gg, mowri, bundle_verbose);
        /* the OpenCL string was succesfully generated, we can now attempt to compile and benchmark it */
        ++global_counter;

        mowri << "\n[" << global_counter  <<  ", " << std::fixed << std::setprecision(2) << elapsed_seconds;
        mowri << std::setprecision(6) << "s]\t" << hyper_param_current.get_string() << Endl;
        
        setup_tinykernels(hyper_param_current, bundle);  

        core_gemm_loop(find_params.n_runs_per_kernel, false);

        /* A new best kernel found */
        if (best_solns_path.size() == 0 || median_time < 1.000*best_solns_path.back().statistics.median_benchmark_time){
          update_elapsed_seconds();
          improvement_found_on_front = true;          
          
          std::time_t generation_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
          auto sstats = TinyGemmSolutionStatistics(median_time, median_gflops, elapsed_seconds, std::ctime(&generation_time), find_params);
          best_solns_path.emplace_back (gg, sstats, bundle.v_tgks, hyper_param_current.get_string(), devinfo, constraints_string );
        }
        ++hfi;
        update_elapsed_seconds();
      }
      
  
      if (improvement_found_on_front == true && allotted_time > elapsed_seconds){
        
        /* getting all `one-away's */        
        auto one_aways = hyper_param_current.get_one_aways();
        
        /* refreshing hyper front */
        hyper_front.clear();

        for (auto & hp : one_aways){
           
          auto hp_string = hp.get_string();

          auto in_graph_tuple = hp.in_graph();
          if (std::count(one_aways.begin(), one_aways.end(), hp) > 1){
            throw tinygemm_error("duplicates in one_aways not allowed, should have already been filtered. Could filter out here, but less efficient ");
          }        

          else if (std::get<0>(in_graph_tuple) == false){
            std::stringstream errmss;
            errmss << "constraint violators not allowed, should have already been filtered. Could filter out here, but less efficient. \nThe hyperstring is\n" << hp.get_string();
            errmss << "\nrecall the geometry is\n" << gg.get_string();
            errmss << "\nthe constraint violations string is:\n" << std::get<1>(in_graph_tuple);
            throw tinygemm_error(errmss.str());
          }
          
          /* filtering out if it has already been considered */
          else if (std::find(hyper_front_history.begin(), hyper_front_history.end(), hp_string) != hyper_front_history.end()){
            //front_insertion_type =  's'; 
          }


          /* filtering out non-deriveables */
          else if (std::get<0>(derivedparams::get_deriveability(hp, gg)) == false){
            
            // << "----------------------------- non derivable ----------------------" << std::endl;
            // << std::get<1>(derivedparams::get_deriveability(hp, gg));
            //front_insertion_type = 'd';
          }
          
          /* looks ok, adding it to the hyper-front */
          else{
            //front_insertion_type = '+';
            hyper_front.push_back(hp);
          }
          //mowri << front_insertion_type;

        }
        //mowri << ")  [+" << hyper_front.size() << "]" << Endl;

      }
    }
    
    if (allotted_time <= elapsed_seconds){
      mowri << "stopping the search because the allotted time has been surpassed" << Endl;
    }
    
    else if (improvement_found_on_front == false){
      mowri << "stopping the search because a locally minimal kernel has been found" << Endl;
    }
    
    else{
      throw tinygemm_error("why did the algorithm stop ? ");
    }
    
    if (best_solns_path.size() == std::numeric_limits<float>::max()){
      throw tinygemm_error("\nThere were no solutions found. This suggests that the initial kernel did not work (could not derive hyper parameters, required too much memory, or did not compile. Maybe there is some preceding warning printed which sheds light on this? Probably with a modification to the FindStartType or the constraints_string, this should be resolved. For example, the unroll UNR can be reduced if the problem is memory. jn should catch certain problems in architests ");
    }
  
    auto leading_size = best_solns_path.back().hyper_param_string.size() + 2;
    
    std::string startstring  = "hyper parameter string:";
    startstring.resize(leading_size, ' ');
    mowri << "\n" << startstring <<  "\t time when found:\t median Gflops/s:" << Endl;

    for (auto & x : best_solns_path) {
      std::string solnstring = x.get_hyper_param_string();
      solnstring.resize(leading_size, ' ');
      mowri <<  std::fixed <<  solnstring << "\t " << x.statistics.solution_discovery_time << "\t\t " << x.statistics.median_benchmark_gflops  << Endl;
    }

    return best_solns_path.back();
  }
}; 

cl_mem get_copy(
cl_command_queue command_queue,
cl_mem c,   
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
const std::string & hash
){  
  cl_mem c_copied;
  cl_event c_copy_event; 
  size_t n_c = gg.ldX[nsHP::matC] * (gg.tX[nsHP::matC] == gg.isColMajor ? gg.m : gg.n) + toff.oc;
  size_t c_memsize = gg.derived.float_size_bytes*n_c;
  c_copied = openclutil::cl_create_buffer_from_command_queue(command_queue, CL_MEM_READ_WRITE, c_memsize, NULL, hash + ", in function get_copy of tinygemm");
  openclutil::cl_enqueue_copy_buffer(command_queue, c, c_copied, 0, 0, c_memsize, 0, NULL, &c_copy_event, hash + ", in function get_copy of tinygemm");
  openclutil::cl_wait_for_events(1, &c_copy_event, "in function find of tinygemm");
  return c_copied;
}

cl_mem get_single(
cl_command_queue command_queue,
const std::string & hash
){
  size_t c_memsize = 1;
  cl_mem single = openclutil::cl_create_buffer_from_command_queue(command_queue, CL_MEM_READ_WRITE, c_memsize, NULL, hash + ", in function cl_mem get_single of tinygemm");
  return single;
}


tinygemm::TinyGemmSolution
find(
cl_command_queue command_queue,
const FindParams & find_params,
cl_mem a,   
cl_mem b,
cl_mem c,
cl_mem workspace,
const std::string constraints_string,
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
outputwriting::OutputWriter & mowri,
bool c_is_const){

  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);  

  bool full_constraints_expected = false;

  cl_mem c_to_use (nullptr);
  openclutil::SafeClMem c_copied("to be used in the case that c_is_const");
  if (c_is_const == true){
    c_to_use = get_copy(command_queue, c, gg, toff, "c_is_const is true, making the copy");
    c_copied.clmem = c_to_use;
  }
  
  else{
    c_to_use = c;
  }

  OpenCLGemmEncapsulator oger(command_queue, gg, toff, a, b, c_to_use, workspace, constraints_string, full_constraints_expected, mowri); 
  return oger.find(find_params);
}


std::tuple<bool, std::string> check_for_default(
cl_command_queue command_queue,
std::string constraints_string,
const tinygemm::TinyGemmGeometry & gg, 
std::string k_comment){

  openclutil::OpenCLDeviceInfo devinfo(command_queue);
  std::string k_dev = devinfo.identifier;
  std::string k_con = constraints_string;
  std::string k_geo = gg.get_string();
  
  std::stringstream ss;
  ss << "\n";
  ss << "\nfailure in check_for_default, with keys\n";
  ss << get_cache_keys_string(k_dev, k_con, k_geo, k_comment);
  
  std::string final_comment("see tests/gencache.cpp for an example of generating a cache entry.\n");
  
  if (kernel_cache.count(k_dev) == 0){
    ss << "\nUnrecognised device identifier in cache, maybe the cache needs to be built for this device? \n" << final_comment;
    return std::make_tuple(false, ss.str());
  }
  
  if (kernel_cache.at(k_dev).count(k_con) == 0){
    ss << "\nUnrecognised constraints_string in cache, maybe the cache needs to be built with these constraints? \n" << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).at(k_con).count(k_geo) == 0){
    ss << "\nUnrecognised gg.get_string() (geometry string) in cache, maybe a cache entry needs to be generated with this geometry? \n" << final_comment;
    return std::make_tuple(false, ss.str());
  }

  if (kernel_cache.at(k_dev).at(k_con).at(k_geo).count(k_comment) == 0){
    ss << "\nUnrecognised k_comment in cache\n";
    return std::make_tuple(false, ss.str());
  }  
  
  return std::make_tuple(true, "");
}
  
  

tinygemm::TinyGemmSolution
get_default(
cl_command_queue command_queue,
std::string constraints_string,
const tinygemm::TinyGemmGeometry & gg, 
std::string k_comment,
outputwriting::OutputWriter & mowri){

  openclutil::OpenCLDeviceInfo devinfo(command_queue);
  
  std::string k_dev = devinfo.identifier;
  std::string k_con = constraints_string;
  std::string k_geo = gg.get_string();
  
  auto pair = check_for_default(command_queue, constraints_string, gg, k_comment);
  if (std::get<0>(pair) == false){
    throw tinygemm_error(std::get<1>(pair));
  }
  
  tinygemm::TinygemmCachedSolution cached_soln(kernel_cache.at(k_dev).at(k_con).at(k_geo).at(k_comment));
  
  /* generating source files from cache */
  hyperparams::Graph graph(gg, devinfo, cached_soln.hyperstring, false);
  hyperparams::HyperParams hp(graph);
  bool bundle_verbose_get_default = true;
  auto bundle = tinygemm::kerngen::get_bundle(hp,gg, mowri, bundle_verbose_get_default);
 
  return { gg, cached_soln.stats, bundle.v_tgks, hp.get_string(), devinfo, constraints_string};

}
  
  
void benchgemm(
  cl_command_queue command_queue,
  const std::string & hyperstring,
  unsigned n_runs,
  const tinygemm::TinyGemmGeometry & gg,
  const tinygemm::TinyGemmOffsets & toff, 
  cl_mem a_gpu,
  cl_mem b_gpu, 
  cl_mem c_gpu,
  cl_mem workspace_gpu,  
  outputwriting::OutputWriter & mowri,
  bool c_is_const){
  
  
  bool full_constraints_expected = true;
  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);
  if (c_is_const == true){
    
    cl_mem c_cop = get_copy(command_queue, c_gpu, gg, toff, "copy of c in benchgemm");
    openclutil::SafeClMem c_copied("copy of c in find");
    c_copied.clmem = c_cop;
    
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a_gpu, b_gpu, c_copied.clmem, workspace_gpu, hyperstring, full_constraints_expected, mowri);
    oger.benchgemm(n_runs);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a_gpu, b_gpu, c_gpu, workspace_gpu, hyperstring, full_constraints_expected, mowri);
    oger.benchgemm(n_runs);
  }
}


tinygemm::TinyGemmSolution
find(float allotted_time, cl_command_queue command_queue, cl_mem a, cl_mem b, cl_mem c, bool enforce_determinism, const tinygemm::TinyGemmGeometry & tgg){

  float min_time_without_cache = 60.00;
  
  SummaryStat sumstat (tinygemm::Median);
  unsigned allotted_descents = 1000; /* letting time be the termination condition */
  unsigned n_runs_per_kernel = 3; 
  FindParams find_params(allotted_time, allotted_descents, n_runs_per_kernel, sumstat);

  cl_mem workspace = nullptr;

  std::string constraints_string = "A_WOS0__B_WOS0"; /* no workspace */
  if (enforce_determinism == true){
    constraints_string += "__C_ICE1";
  }
  
  tinygemm::TinyGemmOffsets toff(0,0,0,0,0,0,0);
  outputwriting::OutputWriter mowri(true, false, "");
  bool c_is_const = true;
  
  std::string k_comment = "";
  auto pair = check_for_default(command_queue, constraints_string, tgg, k_comment);
    
  if (std::get<0>(pair) == false && allotted_time < min_time_without_cache){
    
    std::stringstream ss;
    ss << "\nin tinygemm find (version without workspace), with the following:\n";
    ss << "(1) allotted_time (" << allotted_time << ") is less than min_time_without_cache (" << min_time_without_cache << ")\n";
    ss << "(2) there is not cache entry : ";
    ss <<  std::get<1>(pair);
    ss << "\nEither set allotted_time to be greater than min_time_without_cache, or generate a cache entry, otherwise you will get a poor solution\n";
    
    throw tinygemm_error(ss.str());
  }
  
  return find(command_queue, find_params, a, b, c, workspace, constraints_string, tgg, toff, mowri, c_is_const);  
}

} //namespace
