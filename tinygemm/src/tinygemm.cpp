#include <thread>
#include <limits>
#include <chrono>
#include <sstream>
#include <vector> 
#include <algorithm>
#include <map>

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

/* TODO : float4 */

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
  
  /* TODO : this belongs somewhere else */
  const std::vector<std::string> possible_basic_types = {"wsa", "wsb", "betac", "main"};
  

private:
  outputwriting::OutputWriter mowri;
  /* vector of times over a set of runs on core loop */
  std::vector<float> v_t_total;

  std::map<std::string, TinyGemmKernel > tk_kernels_map;
  std::vector <TinyGemmKernel *> tk_kernels_active;  
  std::vector<std::vector <unsigned > > v_wait_indices;

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
  gg(gg_),
  toff(toff_),
  gpum(a_gpu_, b_gpu_, c_gpu_, workspace_gpu_),
  mowri(verbose_, outputfilename.compare("") != 0, outputfilename_)
  {

    for (auto & x : possible_basic_types){
      tk_kernels_map[x] = TinyGemmKernel(command_queue, x);
    }
    
    run_checks();
  }
  
private:
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
    
    tk_kernels_map.at(type.basic).set_kernel_args(arg_sizes_values);
  }


  bool refresh_needed(const std::string & type, const hyperparams::HyperParams & new_hp, const derivedparams::DerivedParams & new_dp){
    
    /* TODO : check (here) hyper parameters to see if needed anew */
            
    if (type.compare("betac") == 0){
       if (tk_kernels_map.at("betac").is_set() == false && new_dp.main_does_beta_c_inc == 0){
         return true;
       }
       else{
         return false;
       }
    }
    
    else if (type.compare("main") == 0){
      return true;
    }
    
    else if (type.compare("wsa") == 0){
      if (tk_kernels_map.at("wsa").is_set() == false && new_hp.at(nsHP::matA).vs[nsHP::WOS] != 0){
         return true;
       }
       else{
         return false;
       }
    }

    else if (type.compare("wsb") == 0){
      if (tk_kernels_map.at("wsb").is_set() == false && new_hp.at(nsHP::matB).vs[nsHP::WOS] != 0){
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
    if (refresh_needed(type.basic, hp, dp) == true){
      tk_kernels_map.at(type.basic).update(ks, mowri);
      set_kern_args(type);
    }
  }


  void setup_tinykernels(const hyperparams::HyperParams & hp, const kerngen::Bundle & bundle ){
    
    v_wait_indices = bundle.v_wait_indices;
    tk_kernels_active.resize(0);
    
    for (unsigned ksi = 0; ksi < bundle.v_tgks.size(); ++ksi){
      std::string basic = bundle.v_tgks[ksi].type.basic;
      refresh_kernel(bundle.v_tgks[ksi], hp, bundle.dp);
      tk_kernels_active.push_back(&tk_kernels_map[basic]);
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
  
  void print_run_times(cl_int status){
    
    if (status == CL_SUCCESS){
      mowri << "total time : " <<  v_t_total.back() << "\t (";
      for (unsigned k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind){
        mowri << " k" << k_ind << ": " << tk_kernels_active[k_ind]->v_times.back() << " ";
      }
      mowri << ") " << "\tGflops/s : " << 2.0 * gg.m * gg.n * gg.k / (v_t_total.back() * 1e6) << Endl;
    }

    else{
      mowri << "elapsed time : " <<  " (max float) \n";
    }
  }


  void reset_v_times(){
    v_t_total.resize(0);
    for (auto & ptr_tk_kernel : tk_kernels_active){
      ptr_tk_kernel->reset_times();
    }
  }
  
  
  void core_gemm_loop(size_t n_runs){
    
    reset_v_times();
    
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
      
      /* TODO : find out when exactly to use cl Flush  */
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
      print_run_times(status);
    }
  }

  void deriveability_test(const hyperparams::HyperParams & hp, const std::string & hash){
    auto deriveability = derivedparams::get_deriveability(hp, gg);      
    if (std::get<0>(deriveability) == false){
      throw tinygemm_error(hash + ": the hyper parameters in benchgemm are not consistent, specifically, from get_deriveability \n" + std::get<1>(deriveability));
    }
  }

public:
  void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs){

    address_check_valid();

    if (n_runs == 0){
      throw tinygemm_error("n_runs to benchgemm should be a positive integer");
    }

    for ( unsigned i = 0; i < hps.size(); ++i) {

      mowri << "\nSource kernel " << "(" << i + 1 << "/" << hps.size() << ") "  << hps[i].get_string() << Endl;      
      
      deriveability_test(hps[i], "in benchgemm");
      
      auto bundle = tinygemm::kerngen::get_bundle(hps[i],gg); 
      auto atr = architests::architecture_specific_tests(command_queue, hps[i], bundle.dp);
      if (std::get<0>(atr) == false){
        throw tinygemm_error(std::get<1>(atr));
      }

      setup_tinykernels(hps[i], bundle); 
      
      mowri << "(benchgemm) geometry  \t:" << gg.get_string()  << "\nEntering the core gemm loops" << Endl;
      core_gemm_loop(n_runs);
    }
  }
  
  
  tinygemm::TinyGemmSolution
  get_default(std::string constraint_string){
    hyperparams::HyperParams hp = hyperparams::get_hp_start(FindStartType::Default, constraint_string, gg);
    deriveability_test(hp, "in get_default");    
    auto bundle = tinygemm::kerngen::get_bundle(hp,gg); 
    tinygemm::TinyGemmSolutionStatistics tgss(std::numeric_limits<float>::max(), 0, 0);    
    return { hp, gg, bundle.dp, tgss, bundle.v_tgks };
  }


  hyperparams::HyperParams get_hyper_param_start(std::string constraint_string, FindStartType fst){

    hyperparams::HyperParams hyper_param_start;
    bool found_a_deriveable_hp = false;
    unsigned deriveable_search_iteration = 0;
    std::stringstream deriveablesearch_ss;
    
    /* the number of attempts at finding a deriveable HyperParams given the constraint string */
    const unsigned n_trials = fst == FindStartType::Random ? 1000 : 1;


    ////set-up the search graphs based on gg. //TODO : pass constraint_string in here, so that no elimination of strings later. 
    //auto graphs = hyperparams::get_graphs(gg);

    
    while (found_a_deriveable_hp == false && deriveable_search_iteration < n_trials){
      
      hyper_param_start = hyperparams::get_hp_start(fst, constraint_string, gg);
      auto deriveability = derivedparams::get_deriveability(hyper_param_start, gg);
      if (std::get<0>(deriveability) == false){
        deriveablesearch_ss << hyper_param_start.get_string() << " is not deriveable, because " << std::get<1>(deriveability) << "\n";            
      }
      else{
        found_a_deriveable_hp = true;
      }
      ++deriveable_search_iteration;
    }
    
    if (found_a_deriveable_hp == false){
      deriveablesearch_ss << "\n\nStruggling to find a deriveable set of hyper parameters which satisfy the geometry and constraints. THe number of attempts made is " << n_trials << "\n throwing an error\n";
      //should rather return no solution ? or one which we know to be deriveable ? Yes. throwing an error should not be stochastic!
      throw tinygemm_error(deriveablesearch_ss.str());
    }
    return hyper_param_start;
  }
  
  tinygemm::TinyGemmSolution find(float allotted_time, std::string constraint_string, FindStartType fst, unsigned n_runs_per_kernel){
    
    mowri << "(find) geometry : " << gg.get_string()  << Endl;

    if (allotted_time <= 0){
      mowri << "allotted_time (" << allotted_time << ") is insufficient for benchmarking, returning default TinyGemmSolution based on gg and constraint_string without bencing/searching" << Endl;
      return get_default(constraint_string);
    }

    address_check_valid_and_reliable();

    /* we will count how many kernels are successfully generated AND compiled AND benchmarked */
    unsigned global_counter = 0;

    
    //TODO : get_params_from_full_hyperstring would be clearer that ", false". 
    auto constraint_params = hyperparams::get_params_from_string(constraint_string, false);

    
    hyperparams::HyperParams hyper_param_start = get_hyper_param_start(constraint_string, fst);

    /* we track the best TinyGemmSolution found during the search  */    
    std::vector<tinygemm::TinyGemmSolution> path_of_best_solns;
    /* In here, we will store all previously considered HyperParams strings, used to check and ensure that we do not consider a HyperParam more than once */
    std::vector<std::string> hyper_front_history;
    /* while generating, compiling and benchmarking kernels, we will keep track of the fastest found thus far */
    float best_time = std::numeric_limits<float>::max();
    hyperparams::HyperParams best_hp = hyper_param_start;
    std::vector<hyperparams::HyperParams> hyper_front = { hyper_param_start };

    bool improvement_found_on_front = true;    
    mowri << "allotted time : " << allotted_time << Endl;

    float elapsed_seconds;
    auto start = std::chrono::high_resolution_clock::now();

    auto update_elapsed_seconds  =  [&elapsed_seconds, &start]()  {      
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> fp_ms = end - start;
      elapsed_seconds = fp_ms.count();          
    };
    update_elapsed_seconds();
    
    while (improvement_found_on_front == true){
      improvement_found_on_front = false;
      unsigned hfi = 0;
      while (hfi < hyper_front.size() && improvement_found_on_front == false && elapsed_seconds < allotted_time){
        hyperparams::HyperParams hp = hyper_front[hfi];
        std::string hp_string = hp.get_string();
        hyper_front_history.push_back(hp_string);
        /* extra precaution, should be able to remove this */
        deriveability_test(hp, "in find loop");        
        auto bundle = tinygemm::kerngen::get_bundle(hp,gg);
        /* the OpenCL string was succesfully generated, we can now attempt to compile and benchmark it */
        ++global_counter;
        mowri << "\nglobal gen-com-bench : " << global_counter  <<  ".\n" << hp.get_string() << Endl;
        setup_tinykernels(hp, bundle);  
        core_gemm_loop(n_runs_per_kernel);
        std::sort(v_t_total.begin(), v_t_total.end());
        /* Taking the fastest or median? */ 
        float median_time = v_t_total[0];//[v_t_total.size()/2]; 
        if (std::abs(v_t_total.back() - median_time) / median_time > 0.2) {
          mowri << "tinygemm_warning: large variance in times. " <<  Endl;
        }
        
        update_elapsed_seconds();                        
        mowri << "median time  : " << median_time << "\t m-Gflops/s : " << 2.0 * gg.m * gg.n * gg.k / (median_time * 1e6);
        mowri << "  \ttotal elapsed seconds : " << elapsed_seconds << Endl;

        /* A new best kernel found */
        if (median_time < 1.000*best_time){
          improvement_found_on_front = true;
          best_time = median_time;
          float median_gflops = (2. * gg.m * gg.n * gg.k) / (median_time * 10e5);
          best_hp = hp;
          mowri << "********************************************" << Endl;
          mowri << "\tNEW BEST MEDIAN TIME FOUND" << Endl;
          mowri << "\tmedian gflops  : " << median_gflops << Endl;
          mowri << "\tbreaking from hyper front " << Endl;
          mowri << "********************************************" << Endl;
          update_elapsed_seconds();
          tinygemm::TinyGemmSolutionStatistics tgss(median_time, median_gflops, elapsed_seconds);               
          path_of_best_solns.emplace_back( hp, gg, bundle.dp, tgss, bundle.v_tgks  );
        }
        ++hfi;
        update_elapsed_seconds();
      }
      
  
      if (improvement_found_on_front == true && allotted_time > elapsed_seconds){
        mowri << "creating new current hyper front \t(";// << Endl;
        /* getting all `one-away's */        
        auto one_aways = best_hp.get_one_aways();
        char front_insertion_type;
        /* refreshing hyper front */
        hyper_front.resize(0);

        for (auto & hp : one_aways){
           
          auto hp_string = hp.get_string();

          if (std::count(one_aways.begin(), one_aways.end(), hp_string) > 1){
            throw tinygemm_error("duplicates in one_aways not allowed, or filter out here ");
          }        
          
          /* filtering out if it has already been considered */
          else if (std::find(hyper_front_history.begin(), hyper_front_history.end(), hp_string) != hyper_front_history.end()){
            front_insertion_type =  's'; 
          }

          /* filtering out if it does not satisfy the constraints */
          else if (hp.satisfies_where_source_defined(constraint_params) == false){
            front_insertion_type = 'c';
          }

          /* filtering out non-deriveables */
          else if (std::get<0>(derivedparams::get_deriveability(hp, gg)) == false){
            front_insertion_type = 'd';
          }
          
          /* looks ok, adding it to the hyper-front */
          else{
            front_insertion_type = '+';
            hyper_front.push_back(hp_string);
          }
          mowri << front_insertion_type;
        }
        
        mowri << ")\n" << "new hyper front size : " << hyper_front.size() << Endl;
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
    
    if (path_of_best_solns.size() == 0){
      throw tinygemm_error("\nThere were no solutions found. This suggests that the initial kernel did not work (could not derive hyper parameters, required too much memory, or did not compile. Maybe there is some preceding warning printed which sheds light on this? Probably with a modification to the FindStartType or the constraint_string, this should be resolved. For example, the unroll UNR can be reduced if the problem is memory. TODO(jn) catch certain problems in architests ");
    }
  
    mowri << "\nstart kernel : " << hyper_param_start.get_string() << Endl;
    mowri << "best kernel  : " << best_hp.get_string() << Endl;

    mowri << "the kernels along the path the final solution:  " << Endl; 
    mowri << "hyper parameter string                                          \t time when found\t median gflop/s" << Endl;

    for (auto & x : path_of_best_solns){
      mowri <<  x.get_hyper_param_string() << "\t " << x.statistics.solution_discovery_time << "\t\t " << x.statistics.median_benchmark_gflops  << Endl;
    }
    
    mowri <<  "\n" << path_of_best_solns.back().get_hyper_param_string() << "\t " << elapsed_seconds << "\t\t " << path_of_best_solns.back().statistics.median_benchmark_gflops  << Endl;
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
  size_t c_memsize = gg.derived.float_size_bytes*n_c;
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
const std::string constraint_string,
FindStartType fst,
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
bool verbose, 
std::string logfile, 
bool c_is_const){
  
  
  /* The number of times each kernel is run in find. 
   * consider adding this parameter to user API. */
  unsigned n_runs_per_kernel = 3;

  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);  

  if (c_is_const == true){
  
    openclutil::SafeClMem c_copied = get_copy(command_queue, c, gg, toff, "copy of c in find");
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a, b, c_copied.clmem, workspace, logfile, verbose); 
    return oger.find(allotted_time, constraint_string, fst, n_runs_per_kernel);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a, b, c, workspace, logfile, verbose); 
    return oger.find(allotted_time, constraint_string, fst, n_runs_per_kernel);
  }
}

tinygemm::TinyGemmSolution
get_default(
const std::string constraint_string,
const tinygemm::TinyGemmGeometry & gg,
bool verbose, 
std::string logfile){
  
  OpenCLGemmEncapsulator oger({}, gg, {0,0,0,0,0,0,0}, {}, {}, {}, {}, logfile, verbose); 
  return oger.get_default(constraint_string);
 
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
  
  
  std::cout << "in tinygemm's benchgemm" << std::endl;
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
















      
      //if (improvement_found_on_front == false && front_search_horizon == 1 && elapsed_seconds < allotted_time){
        //++front_search_horizon;

        ///* TODO : if you WANT to go onto front 2, set the following to true. This should be finalised TODO TODO TODO  */        
        //const bool jump_to_front_horizon_size_2 = true;
        
        //if (jump_to_front_horizon_size_2 == true){
          //improvement_found_on_front = true;
          //mowri << "\n\n----------SWITCHING TO FRONT HORIZON SIZE 2-------------\n\n" << Endl;
        //}
      //}
      
      //if (improvement_found_on_front == true && front_search_horizon == 2){        
        ///* getting all `two-aways' */
        ////hyper_front = best_hp.get_two_aways(gg);
      //}
      
      //if (improvement_found_on_front == false && front_search_horizon == 2){
        ///* this is going to cause the end of the search */
      //}
      
      //if (front_search_horizon != 1 && front_search_horizon != 2){
        //throw std::logic_error("front_search_horizon is neither 1 nor 2. This is currently not possible, Broken algorithm, come fix.");        
      //}
