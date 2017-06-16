#include <MIOpenGEMM/iterexperiments.hpp>

int main(){

  bool write_to_dir = false;
  std::string basedir ("");
  
  if (write_to_dir == true){
    
    /* Mayank : change this to some directory on your system where the results can be written */ 
    basedir = "/home/james/gemmpaper/data/apiless5/";
  
    if (basedir.back() != '/'){
      basedir += "/";
    }
  }


  float allotted_time = 2000.00; 
  unsigned n_runs_per_kernel = 1;
  MIOpenGEMM::SummaryStat sumstat(MIOpenGEMM::Max);
  bool verbose = false;
  bool verbose_outer = true;
  std::string fn_outer("");
  
  
 
  unsigned n_iterations_smallgrowing = 0;      // at 25 seconds each
  unsigned n_iterations_square = 1;            // at 6000 seconds each
  unsigned n_iterations_smalldeep = 0;         // at 250 seconds each
  unsigned n_iterations_largedeep = 10;          // at 1100 seconds each
  unsigned n_iterations_problem_geometries = 0;  // 1000;

  
  std::cout << "\nSMALLGROWING EXPERIMENTS : HOW DOES PERFORMANCE SCALE AS K INCREASES  ?  " << std::endl;
  auto geometries = MIOpenGEMM::get_small_growing_geometries();

  std::vector<std::string> v_constraints = {  
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE8_NAW10",
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE4_NAW10",
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE2_NAW10",
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE1_NAW10"};
  std::string subdir(basedir + "smallgrowing/");

  if (n_iterations_smallgrowing > 0){
    MIOpenGEMM::FindParams find_params(allotted_time, n_iterations_smallgrowing, n_runs_per_kernel, sumstat);
    MIOpenGEMM::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);
  }



  
  unsigned small_threshold = 1000*1000*200;
  
  std::cout << "\nSMALLDEEP EXPERIMENTS : HOW DO WE DO ON THE SMALL DEEPBENCH PROBLEMS (WITH AND WITHOUT ICE ALLOWED) ? " << std::endl;
  geometries = MIOpenGEMM::get_small_deepbench_geometries(small_threshold);
  v_constraints = {"", "C_ICE1"};
  if (basedir != "")
    subdir = basedir + "smalldeep/";
  
  if (n_iterations_smalldeep > 0){
    MIOpenGEMM::FindParams find_params(allotted_time, n_iterations_smalldeep, n_runs_per_kernel, sumstat);
    MIOpenGEMM::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);
  }


  std::cout << "\nLARGEDEEP EXPERIMENTS : HOW DO WE DO ON THE LARGE DEEPBENCH PROBLEMS (WITHOUT ICE ALLOWED) ?" << std::endl;
  geometries = MIOpenGEMM::get_large_deepbench_geometries(small_threshold);
  v_constraints = {""};
  if (basedir != "")
    subdir = basedir + "largedeep/";

  if (n_iterations_largedeep > 0){
    MIOpenGEMM::FindParams find_params(allotted_time, n_iterations_largedeep, n_runs_per_kernel, sumstat);
    MIOpenGEMM::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);
  }

  



  std::cout << "\nSQUARE EXPERIMENTS : HOW DO WE DO ON THE STANDARD SQUARE PROBLEMS ? (~600s per iteration) " << std::endl;
  geometries = MIOpenGEMM::get_square_geometries();
  v_constraints = {""};
  if (basedir != "")
    subdir = basedir + "square/";
  if (n_iterations_square > 0){
    MIOpenGEMM::FindParams find_params(allotted_time, n_iterations_square, n_runs_per_kernel, sumstat);
    MIOpenGEMM::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);
  }


  std::cout << "\nPROBLEM EXPERIMENTS : FOR GEOMETRIES WHICH ARE GIVING US A HARD TIME :-{|>" << std::endl;
  geometries = MIOpenGEMM::get_problem_geometries();
  v_constraints = {""};
  if (basedir != "")
    subdir = basedir + "problemgeoms/";
  if (n_iterations_problem_geometries > 0){
    MIOpenGEMM::FindParams find_params(allotted_time, n_iterations_problem_geometries, n_runs_per_kernel, sumstat);
    MIOpenGEMM::run_find_experiments(geometries, v_constraints, find_params, verbose, basedir, verbose_outer, fn_outer);
  }
  
}
 
