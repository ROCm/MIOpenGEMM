#include <string>
#include <iostream>

#include <MIOpenGEMM/redirection.hpp>


struct ExampleGeometry{
  bool isColMajor, tA, tB, tC;
  unsigned m, n;
  std::string a, b; 
  ExampleGeometry(bool isColMajor_, bool tA_, bool tB_, bool tC_, unsigned m_, unsigned n_, std::string a_, std::string b_):isColMajor(isColMajor_), tA(tA_), tB(tB_), tC(tC_), m(m_), n(n_), a(a_), b(b_) {}
  ExampleGeometry(const ExampleGeometry & eg) = default;
  
};

int main(){
  
  ExampleGeometry g_in(true, true, true, false, 10, 50, "a", "b");
  ExampleGeometry g_out(g_in);
  MIOpenGEMM::redirection::redirect(g_out.isColMajor, g_out.tA, g_out.tB, g_out.tC, g_out.m, g_out.n, g_out.a, g_out.b);
  
  std::cout << "\n" << 
  "colMaj : " << g_in.isColMajor << " --> " << g_out.isColMajor << "\n" << 
  "tA     : " << g_in.tA << " --> " << g_out.tA << "\n" << 
  "tB     : " << g_in.tB << " --> " << g_out.tB << "\n" << 
  "tC     : " << g_in.tC << " --> " << g_out.tC << "\n" <<
  "m      : " << g_in.m << " --> " << g_out.m << "\n" <<  
  "n      : " << g_in.n << " --> " << g_out.n << "\n" <<    
  "a      : " << g_in.a << " --> " << g_out.a << "\n" <<    
  "b      : " << g_in.b << " --> " << g_out.b << "\n" << 
  std::endl;
  
}
