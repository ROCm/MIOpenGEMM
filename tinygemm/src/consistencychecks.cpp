//#include <tinygemm/tinygemmerror.hpp>
//#include <sstream> 
//#include <string> 

//#include <tinygemm/consistencychecks.hpp>

//namespace tinygemm{
//namespace consistencychecks{


  //void check_ldx_mnk_consistent(const tinygemm::TinyGemmGeometry  & gg){
      //check_ldx_mnk_consistent(gg.isColMajor, gg.tX[nsHP::matA], gg.tX[nsHP::matB], gg.tX[nsHP::matC], gg.ldX[nsHP::matA], gg.ldX[nsHP::matB], gg.ldX[nsHP::matC], gg.m, gg.n, gg.k);
  //}

  //void check_ldx_mnk_consistent(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k){
  
    //std::stringstream errm_ss;
    //errm_ss << "An error (or several) emerged while checking the geometry of the gemm problem\n";
    
    //bool isproblem = false;
    
  
    //if (isColMajor == true){
      //if (tC == false && ldc  < m){        
        //errm_ss << "column major, tC = false, ldc < m\n";
        //isproblem = true;
      //}
      
      //if (tC == true && ldc  < n){
        //errm_ss << "column major, tC = true, ldc < n\n";
        //isproblem = true;
      //}
      
      //if (tA == false && lda < m){
        //errm_ss << "column major, tA = false, lda < m\n";
        //isproblem = true;
      //}
      
      //if (tA == true && lda < k){
        //errm_ss << "column major, tA = true, lda < k\n";
        //isproblem = true;
      //}
    
      //if (tB == false && ldb < k){
        //errm_ss << "column major, tB = false, ldb < k\n";
        //isproblem = true;
      //}

      //if (tB == true && ldb < n){
        //errm_ss << "column major, tB = true, ldb < n\n";
        //isproblem = true;
      //}
    //}
    
    //else{

      //if (tC == false && ldc < n){
        //errm_ss << "row major, tC = false, ldc < n\n";
        //isproblem = true;
      //}

      //if (tC == true && ldc < m){
        //errm_ss << "row major, tC = true, ldc < m\n";
        //isproblem = true;
      //} 
      
      //if (tA == false && lda < k){
        //errm_ss << "row major, tA = false, lda < k\n";
        //isproblem = true;
      //}
      
      //if (tA == true && lda < m){
        //errm_ss << "row major, tA = true, lda < m\n";
        //isproblem = true;
      //}
    
      //if (tB == false && ldb < n){
        //errm_ss << "row major, tB = false, ldb < n\n";
        //isproblem = true;
      //}

      //if (tB == true && ldb < k){
        //errm_ss << "row major, tB = true, ldb < k\n";
        //isproblem = true;
      //}      
    //}

    //errm_ss << "This error is being reported from tinygemm file consistencychecks.cpp\n";
    //if (isproblem == true){
      //throw tinygemm_error(errm_ss.str());
    //}
  //}

//}
//} //namespace
