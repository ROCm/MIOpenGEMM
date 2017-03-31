#include "iterexperiments.hpp"

int main(){


  /* Mayank : change this to some directory on your system where the results can be written */ 
  std::string basedir = "/home/james/gemmpaper/data/apiless3/";

  if (basedir.back() != '/'){
    basedir += "/";
  }

  unsigned n_iterations_smallgrowing = 0;       // at 25 seconds each
  unsigned n_iterations_square = 0;        // at 6000 seconds each
  unsigned n_iterations_smalldeep = 0;       // at 250 seconds each
  unsigned n_iterations_largedeep = 10;       // at 1100 seconds each
  unsigned n_iterations_problem_geometries = 0;//1000;
  float allotted_time = 10;
  
  std::cout << "\nFIRST EXPERIMENTS : HOW DOES PERFORMANCE SCALE AS K INCREASES  ?  " << std::endl;
  auto geometries = get_small_growing_geometries();
  bool verbose = false;  
  std::vector<std::string> v_constraints = {  
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE8",
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE4",
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE2",
  "A_MIC8_PAD1_PLU1_LIW1_MIW1_WOS0__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0__C_UNR8_GAL1_PUN0_NAW16_UFO0_MAC5_ICE1"};
  std::string subdir(basedir + "smallgrowing/");
  run_find_experiments(geometries, allotted_time, verbose, v_constraints, n_iterations_smallgrowing, subdir);




  std::cout << "\nSECOND EXPERIMENTS : HOW DO WE DO ON THE STANDARD SQUARE PROBLEMS ? (~600s per iteration) " << std::endl;
  geometries = get_square_geometries();
  v_constraints = {""};
  subdir = basedir + "square/";
  run_find_experiments(geometries, allotted_time, verbose, v_constraints, n_iterations_square, subdir);

  
  unsigned small_threshold = 1000*1000*200;
  
  std::cout << "\nTHIRD EXPERIMENTS : HOW DO WE DO ON THE SMALL DEEPBENCH PROBLEMS (WITH AND WITHOUT ICE ALLOWED) ? " << std::endl;
  geometries = get_small_deepbench_geometries(small_threshold);
  v_constraints = {"", "C_ICE1"};
  subdir = basedir + "smalldeep/";
  run_find_experiments(geometries, allotted_time, verbose, v_constraints, n_iterations_smalldeep, subdir);


  std::cout << "\nFOURTH EXPERIMENTS : HOW DO WE DO ON THE LARGE DEEPBENCH PROBLEMS (WITHOUT ICE ALLOWED) ?" << std::endl;
  geometries = get_large_deepbench_geometries(small_threshold);
  v_constraints = {""};
  subdir = basedir + "largedeep/";
  run_find_experiments(geometries, allotted_time, verbose, v_constraints, n_iterations_largedeep, subdir);
  

  std::cout << "\nFIFTH EXPERIMENTS : FOR GEOMETRIES WHICH ARE GIVING US A HARD TIME :-{|>" << std::endl;
  geometries = get_problem_geometries(small_threshold);
  v_constraints = {""};
  subdir = basedir + "problemgeoms/";
  run_find_experiments(geometries, allotted_time, verbose, v_constraints, n_iterations_problem_geometries, subdir);  
  
}
 
