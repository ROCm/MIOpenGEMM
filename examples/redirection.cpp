/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <string>
#include <miopengemm/redirection.hpp>

struct ExampleGeometry
{
  bool        isColMajor, tA, tB, tC;
  size_t      m, n;
  std::string a, b;
  ExampleGeometry(bool        isColMajor_,
                  bool        tA_,
                  bool        tB_,
                  bool        tC_,
                  size_t      m_,
                  size_t      n_,
                  std::string a_,
                  std::string b_)
    : isColMajor(isColMajor_), tA(tA_), tB(tB_), tC(tC_), m(m_), n(n_), a(a_), b(b_)
  {
  }
  ExampleGeometry(const ExampleGeometry& eg) = default;
};

// Convert to column major, tC = false
int main()
{

  ExampleGeometry g_in(true, true, true, false, 10, 50, "a", "b");
  ExampleGeometry g_out(g_in);
  MIOpenGEMM::redirection::redirect(
    g_out.isColMajor, g_out.tA, g_out.tB, g_out.tC, g_out.m, g_out.n, g_out.a, g_out.b);

  std::cout << "=========================\n";
  std::cout << "Canonical form conversion\n";
  std::cout << "=========================\n"
            << "colMaj : " << g_in.isColMajor << " --> " << g_out.isColMajor << "\n"
            << "tA     : " << g_in.tA << " --> " << g_out.tA << "\n"
            << "tB     : " << g_in.tB << " --> " << g_out.tB << "\n"
            << "tC     : " << g_in.tC << " --> " << g_out.tC << "\n"
            << "m      : " << g_in.m << " --> " << g_out.m << "\n"
            << "n      : " << g_in.n << " --> " << g_out.n << "\n"
            << "a      : " << g_in.a << " --> " << g_out.a << "\n"
            << "b      : " << g_in.b << " --> " << g_out.b << "\n";
  std::cout << "=========================\n";
}
