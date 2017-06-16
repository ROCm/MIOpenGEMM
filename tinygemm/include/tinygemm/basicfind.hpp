#ifndef BASICFIND_HPP
#define BASICFIND_HPP



namespace MIOpenGEMM{
  
  
TinyGemmSolution basicfind(const TinyGemmGeometry & geometry, const TinyGemmOffsets & toff, const FindParams & find_params, bool verbose, std::string logfile, std::string constraints_string, unsigned n_postfind_runs, bool do_cpu_test);


}

#endif

