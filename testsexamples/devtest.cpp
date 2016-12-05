#include <iostream>
#include <vector>

#include <tinygemm/tinygemm.hpp>
#include <tinygemm/devtinygemm.hpp>
#include <tinygemm/makekernelsource.hpp>
#include <tinygemm/defaultoutpath.hpp>




#include <tinygemm/stringutilbase.hpp>

/* Note that currently (13/11/2016) most testing is done through dev/python scripts */

/* testing the writing to text file of a gemm kernel */
int make_kernel(){
  std::map<std::string, unsigned> all_int_parms;
  all_int_parms["is_col_major"] = 0;
  all_int_parms["a_transposed"] = 0;
  all_int_parms["b_transposed"] = 0;
  all_int_parms["c_transposed"] = 0;
  all_int_parms["micro_tile_width"] = 2;  
  all_int_parms["micro_tile_height"] = 2;
  all_int_parms["macro_tile_width"] = 32;
  all_int_parms["macro_tile_height"] = 32; 
  all_int_parms["unroll"] = 16;
  all_int_parms["pad"] = 1;    
  all_int_parms["group_allocation"] = 1;
  all_int_parms["work_item_load_a_pll_to_unroll"] = 0;
  all_int_parms["work_item_load_b_pll_to_unroll"] = 1;
  all_int_parms["unroll_pragma"] = 0;
  all_int_parms["load_to_lds_interwoven"] = 0;
  all_int_parms["c_micro_tiles_interwoven"] = 1;
  all_int_parms["use_edge_trick"] = 1;
  all_int_parms["n_work_items_per_c_elm"] = 1;
  all_int_parms["n_target_active_workgroups"] = 64;
  std::string dir_name = "~/atest/directory/ofsorts";
  std::string t_float = "float";
  tinygemm::mkkern::make_kernel_via_python(dir_name, t_float, all_int_parms, "", true);
  return 0;
}


class DevGemmTester{
  
  public:

    /* ** The parameters common for testing gemmbench and find ********* */
    bool isColMajor;
    bool tA;
    bool tB;
    bool tC;
    size_t m;    
    size_t n;
    size_t k;
    float alpha;
    float beta;
    size_t lda;    
    size_t ldb;
    size_t ldc;
    
    size_t a_offset;    
    size_t b_offset;
    size_t c_offset;
    
    
    std::vector<float> v_c;
    std::vector<float> v_a;
    std::vector<float> v_b;
    std::string outputfilename;
    bool capture_output;
    std::string output;
    bool do_test;
    size_t n_runs;
    /* ****************************************************************** */
    
    DevGemmTester(){
      isColMajor = true;
      if (isColMajor == false){
        throw std::runtime_error("in this test, isColMajor should be true");
      }
      
      /* free to change these parameter values */
      tA = true;
      tB = false;
      tC = false;
      m = 3322;    
      n = 4532;
      k = 3131;
      alpha = 1.1;
      beta = 1.1;
      outputfilename = "";    
      capture_output = false;
      output = "";
      n_runs = 1;
      /* ********************************** */
      lda = tA ? k + 1 : m + 6;    
      ldb = tB ? n + 2 : k + 5;
      ldc = tC ? n + 3 : m + 4;
      size_t n_a = lda * (tA ? m : k);
      size_t n_b = ldb * (tB ? k : n);
      size_t n_c = ldc * (tC ? m : n); 
      
      /* TODO get the true post-gemm C so that we can do a test 
       * (or just keep the main testing from dev/python)*/
      do_test = false;
      
      /* fill matrices with random floats. 
       * Note : if they're integers, the kernel runs faster! */
      v_a.resize(n_a); 
      for (size_t i = 0; i < n_a; ++i){
        v_a[i] = float(rand() % 1000) / 1000. - 0.4;
      }


      
      v_b.resize(n_b);
      for (size_t i = 0; i < n_b; ++i){
        v_b[i] = float(rand() % 1000) / 1000. - 0.5;
      }


      v_c.resize(n_c);
      for (size_t i = 0; i < n_c; ++i){
        v_c[i] = float(rand() % 1000) / 500 - 1.;
      }
      
    }
    
    int red_benchmark(std::vector<std::vector<std::string> > & gpu_kernel_filenames, bool findfirst, float allotted_time, bool enforce_deterministic = false){
      std::cout << "about to call tinygemm::dev::gemm<float>" << std::endl;
      /* We pass cpu pointers to tinygemm::dev::benchmark, which does all the necessary opencl gpu boilerplating */
      tinygemm::dev::benchgemm<float>(isColMajor, tA, tB, tC, m, n, k, alpha, v_a.data(), lda, a_offset, v_b.data(), ldb, b_offset, beta, v_c.data(), ldc, c_offset, {}, gpu_kernel_filenames, capture_output, output, nullptr, do_test, n_runs, outputfilename, findfirst, allotted_time, enforce_deterministic);
      return 0;
    }
      
  
    int call_benchgemm(){
      std::string benchgemmtestpath = "some_valid_kernel_path_cl";
      std::vector<std::vector<std::string> > gpu_kernel_filenames {{ benchgemmtestpath }};
      std::cout << "Hello from tinygemm test!\n";  
      red_benchmark(gpu_kernel_filenames, false, -1.0);
      return 0;
    }
    
    int call_find(){
      std::vector<std::vector<std::string> > gpu_kernel_filenames {};
      red_benchmark(gpu_kernel_filenames, true, 10.0, false); //10 seconds, don't force to be determinisitc
      return 0;
    } 
};



int main (){
    
  DevGemmTester dgt;
  
  bool test_benchmark = false;
  if (test_benchmark == true){
    dgt.call_benchgemm();
  }
  
  bool test_make_kernel = false;  
  if (test_make_kernel == true){
    make_kernel();
  }

  bool test_find = true;  
  if (test_find == true){
    dgt.call_find();  
  }
  
  return 0;
}

