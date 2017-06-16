#ifndef ACCURACYTESTS_HPP
#define ACCURACYTESTS_HPP


#include <algorithm>
#include <sstream>

#include <tinygemm/error.hpp>
#include <tinygemm/outputwriter.hpp>

namespace tinygemm {

namespace accuracytests {



template <typename TFloat>
void elementwise_compare(const TFloat * c_before, double beta, const TFloat * c_cpu, const TFloat * c_gpu, unsigned nels, tinygemm::outputwriting::OutputWriter & mowri);



/* An older and more informative version of testing, used in python dev code. Consider using its good parts elsewhere */
template <typename TFloat>
void accuracy_test(bool isColMajor, bool tC, unsigned m, unsigned n, unsigned ldc, const TFloat * c_true, const TFloat * c_computed, unsigned c_offset, outputwriting::OutputWriter & mowri, double l1_rel_err_tol){
   
  mowri << "Performing accuracy test. " << Flush;
  
  
  if (c_computed == nullptr){
    throw tinygemm_error("in accuracy_test, but pointer c_computed is a nullptr. c_computed needs to passed in here, all that happens here is a comparison between an externally provided c_true and c_computed");
  }
  

  if (c_true == nullptr){
    throw tinygemm_error("in accuracy_test, but pointer c_true is a nullptr. c_true needs to passed in here, all that happens here is a comparison between an externally provided c_true and c_computed");
  }
    
  
  c_computed += c_offset;
  
  /* The different cases of tC, isColMajor are handled in this block */
  if (isColMajor == false){
    std::swap(n,m);
  }
  
  size_t row_stride_c_true = isColMajor ? n : 1;
  size_t col_stride_c_true = isColMajor ? 1 : m; 
  
  size_t row_stride_c = tC ? ldc : 1; 
  size_t col_stride_c = tC ? 1 : ldc; 
  /* *********************************************************** */
    
  /* sum_{i,j} \|c_true[i,j] - c_computed[i,j]\|_1 */ 
  double sum_l1_diff = 0;
  
  /* sum_{i,j} \|c_true[i,j]\|_1 */ 
  double sum_l1_c_true = 0;
  
  /* worker bees */
  double c_true_ij;
  double c_ij;
  
  for (size_t row = 0; row < m; ++row){
    for (size_t col = 0; col < n; ++col){
      c_true_ij = c_true[row*row_stride_c_true + col*col_stride_c_true];
      c_ij = c_computed[row*row_stride_c + col*col_stride_c];
      
      sum_l1_diff += std::abs(c_true_ij - c_ij);
      sum_l1_c_true += std::abs(c_true_ij);
    }
  }
  
  if (sum_l1_c_true < l1_rel_err_tol){
    throw tinygemm_error("Not sure what to do : Is the true c basically a matrix of zeros? Apparently the sum of the absolute values is less than 1e-6, in accuracy_test");
  }
  
  double rel_err = sum_l1_diff / sum_l1_c_true;
  
  if (rel_err > l1_rel_err_tol){
    std::string errm("|| c_true - c_computed ||_1 =  ");
    errm += std::to_string(sum_l1_diff);
    errm +=  " and || c_true - c_computed ||_1 / || c_true ||_1 = ";
    errm += std::to_string(rel_err);
    errm += ". This is larger than l1_rel_err_tol (";
    errm += l1_rel_err_tol;
    errm += "). ";
    mowri << errm << Flush;
    
    mowri << "Here are the `top left' values of (true) and [computed] : " << Endl;
    for (size_t row = 0; row < 3; ++row){
      for (size_t col = 0; col < 2; ++col){
        c_true_ij = c_true[row*row_stride_c_true + col*col_stride_c_true];
        c_ij = c_computed[row*row_stride_c + col*col_stride_c];
        mowri << "(" << c_true_ij  << ") " << "[" << c_ij << "] \t";
      }
      mowri << Endl;
    }
    
    mowri << "\nHere is an larger part of the `top left', showing where ||c_true - c_ij|| > 1e-4" << Endl;    
    for (size_t row = 0; row < std::min<unsigned>(m, 12); ++row){
      for (size_t col = 0; col < std::min<unsigned>(n, 19); ++col){
        c_true_ij = c_true[row*row_stride_c_true + col*col_stride_c_true];
        c_ij = c_computed[row*row_stride_c + col*col_stride_c];
        mowri << int(std::fabs(c_true_ij - c_ij) > 1e-4) <<  " ";
      }
      mowri << Endl;
    }
    mowri << "The above two submatrices are correct up to transposes in c." << Endl;
    
    throw tinygemm_error("Accuracy test failed");
  }
  mowri << "Test passed [rel_err = " <<  rel_err << "]." << Endl;
}

}} //namespaces

#endif
