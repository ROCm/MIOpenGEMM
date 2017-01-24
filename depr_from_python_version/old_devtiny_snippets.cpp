
template <typename TFloat>
void benchgemm(

bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, TFloat alpha, const TFloat * a, unsigned lda, unsigned a_offset, const TFloat * b, unsigned ldb, unsigned b_offset, TFloat beta, TFloat * c, unsigned ldc, unsigned c_offset, 

std::vector<std::string> cpu_algs, const std::vector<hyperparams::HyperParams> hps, 

bool capture_output, std::string & output,
const TFloat * c_true_for_test, unsigned do_test, 

unsigned n_runs, 

std::string outputfilename, 
bool findfirst, 
float allotted_time, 
bool enforce_deterministic){
    
  if (findfirst == true && hps.size() != 0){
    throw tinygemm_error( "findfirst is true, and so gpu_kernel_strings should be an empty list \n");
  }
  
  if (findfirst == true && cpu_algs.size() != 0){
    throw tinygemm_error("findfirst is true, and so cpu_algs should be an empty list \n ");
  }
  
  if (allotted_time >= 0 and findfirst == false){
    throw tinygemm_error("allotted_time is positive, so findfirst was expected to be true. We are being pedantic here, but please set allotted_time to a negative value. Just checking that you're aware that allotted_time is specific to the find algorith");
  }
  
  tinygemm::TinyGemmGeometry gg(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset);  

  std::stringstream buffer;
  auto cout_buff = std::cout.rdbuf();
  if (capture_output == true){
    std::cout.rdbuf(buffer.rdbuf());
  }
  std::ofstream nowhere;
    
  Gemini <TFloat> gem (gg, a, b, c, true, alpha, beta, outputfilename);
  if (findfirst == true){

    tinygemm::TinyGemmSolution tgs = gem.find(allotted_time, enforce_deterministic); 
    gem.benchgemm_cpu(cpu_algs);
    gem.benchgemm_gpu({tgs.hp}, n_runs, do_test, c_true_for_test);
  }
  
  else{
    gem.benchgemm_cpu(cpu_algs);
    gem.benchgemm_gpu(hps, n_runs, do_test, c_true_for_test);
  }
  
  if (capture_output == true){
    output = buffer.str();
    std::cout.rdbuf(cout_buff);
  }
  
  else{
    output = "capture_output was false, so nothing here";
  }
}

template void benchgemm(bool isColMajor, bool tA, bool tB, bool tC,  unsigned m, unsigned n, unsigned k, float alpha, const float * a, unsigned lda, unsigned a_offset, const float * b, unsigned ldb, unsigned b_offset, float beta, float * c, unsigned ldc, unsigned c_offset, std::vector<std::string> cpu_algs, std::vector<hyperparams::HyperParams > gpu_kernel_strings, bool capture_output, std::string & output, const float * c_true_for_test, unsigned do_test, unsigned n_runs, std::string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic);

template void benchgemm(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned k, double alpha, const double * a, unsigned lda, unsigned a_offset, const double * b, unsigned ldb, unsigned b_offset, double beta, double * c, unsigned ldc, unsigned c_offset, std::vector<std::string> cpu_algs, std::vector<hyperparams::HyperParams > gpu_kernel_strings, bool capture_output, std::string & output, const double * c_true_for_test, unsigned do_test, unsigned n_runs, std::string outputfilename, bool findfirst, float allotted_time, bool enforce_deterministic);
