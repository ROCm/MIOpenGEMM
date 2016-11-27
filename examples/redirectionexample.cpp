#include <string>

#include "redirection.hpp"


struct ExampleGeometry{
  bool isColMajor, tA, tB, tC;
  unsigned m, n, lda, ldb;
  std::string a, b; 
  ExampleGeometry(bool isColMajor, bool tA, bool tB, bool tC, unsigned m, unsigned n, unsigned lda, unsigned ldb, std::string a, std::string b):isColMajor(isColMajor), tA(tA), tB(tB), tC(tC), m(m), n(n), lda(lda), ldb(ldb), a(a), b(b) {}
  ExampleGeometry(const ExampleGeometry & eg) = default;
  
};

int main(){
  ExampleGeometry g_in(true, false, false, false, 100, 10, 101, 11, "a", "b");
  ExampleGeometry g_out(g_in);
  tinygemm::redirection::redirect(g_out.isColMajor, g_out.tA, g_out.tB, g_out.tC, g_out.m, g_out.n, g_out.lda, g_out.ldb, g_out.a, g_out.b);
}
