#include <stdexcept>
#include <string> 

#include "consistencychecks.hpp"

namespace tinygemm{
namespace consistencychecks{


  void check_ldx_mnk_consistent(const tinygemm::TinyGemmGeometry  & gg){
      check_ldx_mnk_consistent(gg.isColMajor, gg.tA, gg.tB, gg.tC, gg.lda, gg.ldb, gg.ldc, gg.m, gg.n, gg.k);
  }

  void check_ldx_mnk_consistent(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k){
  
    std::string errm("An error (or several) emerged while checking the geometry of the gemm problem\n ");
    
    bool isproblem = false;
    
  
    //isColMajor, lda, ldb, ldc, tA, tB, m, n, k, (tC?) 
    if (isColMajor == true){
      if (tC == false && ldc  < m){        
        errm += "column major, tC = false, ldc < m\n";
        isproblem = true;
      }
      
      if (tC == true && ldc  < n){
        errm += "column major, tC = true, ldc < n\n";
        isproblem = true;
      }
      
      if (tA == false && lda < m){
        errm += "column major, tA = false, lda < m\n";
        isproblem = true;
      }
      
      if (tA == true && lda < k){
        errm += "column major, tA = true, lda < k\n";
        isproblem = true;
      }
    
      if (tB == false && ldb < k){
        errm += "column major, tB = false, ldb < k\n";
        isproblem = true;
      }

      if (tB == true && ldb < n){
        errm += "column major, tB = true, ldb < n\n";
        isproblem = true;
      }
    }
    
    else{

      if (tC == false && ldc < n){
        errm += "row major, tC = false, ldc < n\n";
        isproblem = true;
      }

      if (tC == true && ldc < m){
        errm += "row major, tC = true, ldc < m\n";
        isproblem = true;
      } 
      
      if (tA == false && lda < k){
        errm += "row major, tA = false, lda < k\n";
        isproblem = true;
      }
      
      if (tA == true && lda < m){
        errm += "row major, tA = true, lda < m\n";
        isproblem = true;
      }
    
      if (tB == false && ldb < n){
        errm += "row major, tB = false, ldb < n\n";
        isproblem = true;
      }

      if (tB == true && ldb < k){
        errm += "row major, tB = true, ldb < k\n";
        isproblem = true;
      }      
    }

    errm += "This error is being reported from tinygemm file consistencychecks.cpp\n";
    if (isproblem == true){
      throw std::runtime_error(errm);
    }
  }

}
} //namespace
