#include <MIOpenGEMM/geometryutil.hpp>
#include <tuple>

namespace MIOpenGEMM{
  
std::vector<Geometry> get_from_m_n_k_tA_tB(const std::vector<std::tuple<unsigned, unsigned, unsigned, bool, bool>> &  basicgeos, unsigned workspace_size ){
  
  std::vector<Geometry> geometries;
  bool isColMajor = true;
  bool tA;
  bool tB;
  bool tC = false;  
  unsigned lda;
  unsigned ldb;
  unsigned ldc;
  unsigned m;
  unsigned n;
  unsigned k;
  
  bool ldx_offset = false;
  
  for (auto & problem : basicgeos){
    std::tie(m, n, k, tA, tB) = problem;
    lda = (tA == isColMajor ? k : m) + (ldx_offset == 1 ? 5 : 0);
    ldb = (tB == isColMajor ? n : k) + (ldx_offset == 1 ? 7 : 0);
    ldc = (tC == isColMajor ? n : m) + (ldx_offset == 1 ? 13 : 0);
    geometries.emplace_back(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, 'f');  
  }
  return geometries;
}

std::vector<Geometry> get_from_m_n_k_ldaABC_tA_tB(const std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>> &  basicgeos, unsigned workspace_size){
  
  std::vector<Geometry> geometries;
  bool isColMajor = true;
  bool tA;
  bool tB;
  bool tC = false;  
  unsigned lda;
  unsigned ldb;
  unsigned ldc;
  unsigned m;
  unsigned n;
  unsigned k;

  for (auto & problem : basicgeos){
    std::tie(m, n, k, lda, ldb, ldc, tA, tB) = problem;
    geometries.emplace_back(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, 'f');  

  } 
  return geometries;  
}

}
