/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <chrono>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/slowcpugemm.hpp>

/* TODO : add option to use OpenBLAS instead of this slow gemm code */

namespace MIOpenGEMM
{
namespace slowcpugemm
{

template <typename TFloat>
class NNInner
{
  public:
  inline TFloat operator()(const TFloat* a,
                           const TFloat* b,
                           size_t      x,
                           size_t      y,
                           size_t      lda,
                           size_t      ldb,
                           size_t      k)
  {
    TFloat inner = 0;
    for (size_t z = 0; z < k; ++z)
    {
      inner += a[x + z * lda] * b[y * ldb + z];
    }
    return inner;
  }
};

template <typename TFloat>
class NTInner
{
  public:
  inline TFloat operator()(const TFloat* a,
                           const TFloat* b,
                           size_t      x,
                           size_t      y,
                           size_t      lda,
                           size_t      ldb,
                           size_t      k)
  {
    TFloat inner = 0;
    for (size_t z = 0; z < k; ++z)
    {
      inner += a[x + z * lda] * b[y + z * ldb];
    }
    return inner;
  }
};

template <typename TFloat>
class TNInner
{
  public:
  inline TFloat operator()(const TFloat* a,
                           const TFloat* b,
                           size_t      x,
                           size_t      y,
                           size_t      lda,
                           size_t      ldb,
                           size_t      k)
  {
    TFloat inner = 0;
    for (size_t z = 0; z < k; ++z)
    {
      inner += a[x * lda + z] * b[y * ldb + z];
    }
    return inner;
  }
};

template <typename TFloat, class FInner>
void gemm_3fors_generic_cpu(const Geometry& gg,
                            const Offsets&  toff,
                            const TFloat*   a,
                            const TFloat*   b,
                            TFloat*         c,
                            TFloat          alpha,
                            TFloat          beta)
{
  // at this point, must be column contiguous (ala fortran)
  // this is a generic slow matrix multiplier for NN, TN, NT.
  // NN, TN, NT will have different FInner template parameters
  // TT should have have been redirected to NN at this point

  a += toff.offsets[Mem::E::A];
  b += toff.offsets[Mem::E::B];
  c += toff.offsets[Mem::E::C];

  FInner finner;

  // For rows of C
  for (size_t x = 0; x < gg.m; ++x)
  {
    // For columns of C
    for (size_t y = 0; y < gg.n; ++y)
    {
      // Set the index of the element in C we're setting,
      size_t target_index;
      if (gg.tX[Mat::E::C] == false)
      {
        target_index = x + y * gg.ldX[Mat::E::C];
      }
      else
      {
        target_index = y + x * gg.ldX[Mat::E::C];
      }
      // and set it
      c[target_index] *= beta;
      c[target_index] += alpha * finner(a, b, x, y, gg.ldX[Mat::E::A], gg.ldX[Mat::E::B], gg.k);
    }
  }
}

template <typename TFloat>
void gemm_3fors_cpu(const Geometry& gg,
                    const Offsets&  toff,
                    const TFloat*   a,
                    const TFloat*   b,
                    TFloat*         c,
                    TFloat          alpha,
                    TFloat          beta)
{

  if (gg.tX[Mat::E::A] == true && gg.tX[Mat::E::B] == true)
  {
    throw miog_error("tA and tB should have been redirected before calling gemm_3fors_cpu");
  }

  else if (gg.isColMajor == false)
  {
    throw miog_error("isColMajor should be true before calling gemm_3fors_cpu");
  }

  else if (gg.tX[Mat::E::A] == false && gg.tX[Mat::E::B] == false)
  {
    gemm_3fors_generic_cpu<TFloat, NNInner<TFloat>>(gg, toff, a, b, c, alpha, beta);
  }

  else
  {
    if (gg.m > gg.n)
    {
      throw std::logic_error("m > n should have been redirected before calling gemm_3fors_cpu");
    }

    if (gg.tX[Mat::E::A] == false && gg.tX[Mat::E::B] == true)
    {
      gemm_3fors_generic_cpu<TFloat, NTInner<TFloat>>(gg, toff, a, b, c, alpha, beta);
    }

    else if (gg.tX[Mat::E::A] == true && gg.tX[Mat::E::B] == false)
    {
      gemm_3fors_generic_cpu<TFloat, TNInner<TFloat>>(gg, toff, a, b, c, alpha, beta);
    }

    else
    {
      throw miog_error("this will never happen");
    }
  }
}

void check_cpu_algs(std::vector<std::string> cpu_algs)
{

  std::vector<std::string> known_algs = {"3fors"};

  for (auto& alg : cpu_algs)
  {
    bool isknown = false;
    for (auto& kalg : known_algs)
    {
      if (alg.compare(kalg) == 0)
      {
        isknown = true;
        break;
      }
    }

    if (isknown == false)
    {
      std::string errm = "unrecognised cpu algorithm, ";
      errm += alg;
      errm += '\n';
      throw miog_error(errm);
    }
  }
}

template <typename TFloat>
void gemms_cpu(Geometry                     gg,
               Offsets                      toff,
               const TFloat*                a,
               const TFloat*                b,
               TFloat*                      c,
               TFloat                       alpha,
               TFloat                       beta,
               std::vector<std::string>     algs,
               outputwriting::OutputWriter& mowri)
{
  check_cpu_algs(algs);
  bool tA = gg.tX[Mat::E::A];
  bool tB = gg.tX[Mat::E::B];
  bool tC = gg.tX[Mat::E::C];
  redirection::redirect(gg.isColMajor,
                        tA,
                        tB,
                        tC,
                        gg.m,
                        gg.n,
                        gg.ldX[Mat::E::A],
                        gg.ldX[Mat::E::B],
                        toff.offsets[Mem::E::A],
                        toff.offsets[Mem::E::B],
                        a,
                        b);
  gg.tX[Mat::E::A] = tA;
  gg.tX[Mat::E::B] = tB;
  gg.tX[Mat::E::C] = tC;

  redirection::confirm_redirection(gg.isColMajor, gg.tX[Mat::E::A], gg.tX[Mat::E::B], gg.m, gg.n);
  gg.check_ldx_consistent();

  for (auto& alg : algs)
  {
    mowri << "launching cpu algorithm : " << alg << Endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (alg.compare("3fors") == 0)
    {
      gemm_3fors_cpu(gg, toff, a, b, c, alpha, beta);
    }

    auto t1           = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    mowri << "elapsed time : " << elapsed_time * 1e-6 << " [s] " << Endl;
  }
}

template void gemms_cpu(Geometry                     gg,
                        Offsets                      toff,
                        const float*                 a,
                        const float*                 b,
                        float*                       c,
                        float                        alpha,
                        float                        beta,
                        std::vector<std::string>     algs,
                        outputwriting::OutputWriter& mowri);

template void gemms_cpu(Geometry                     gg,
                        Offsets                      toff,
                        const double*                a,
                        const double*                b,
                        double*                      c,
                        double                       alpha,
                        double                       beta,
                        std::vector<std::string>     algs,
                        outputwriting::OutputWriter& mowri);
}
}
