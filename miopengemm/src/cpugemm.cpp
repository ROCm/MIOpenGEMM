/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <miopengemm/cpugemm.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/redirection.hpp>

#ifdef MIOPENGEMM_USE_OPENBLAS
#include <cblas.h>
#endif

namespace MIOpenGEMM
{
namespace cpugemm
{

#ifdef MIOPENGEMM_USE_OPENBLAS
namespace openblas
{

using blasint = size_t;

template <typename TFloat>
void gemm_openblas_base(const CBLAS_ORDER,
                        const CBLAS_TRANSPOSE,
                        const CBLAS_TRANSPOSE,
                        const blasint,
                        const blasint,
                        const blasint,
                        const TFloat,
                        const TFloat*,
                        const blasint,
                        const TFloat*,
                        const blasint,
                        const TFloat,
                        TFloat*,
                        const blasint)
{
  throw miog_error("unrecognised float type in openblas gemm");
}

template <>
void gemm_openblas_base(const CBLAS_ORDER     Order,
                        const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB,
                        const blasint         M,
                        const blasint         N,
                        const blasint         K,
                        const float           alpha,
                        const float*          A,
                        const blasint         lda,
                        const float*          B,
                        const blasint         ldb,
                        const float           beta,
                        float*                C,
                        const blasint         ldc)
{
  cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void gemm_openblas_base(const CBLAS_ORDER     Order,
                        const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB,
                        const blasint         M,
                        const blasint         N,
                        const blasint         K,
                        const double          alpha,
                        const double*         A,
                        const blasint         lda,
                        const double*         B,
                        const blasint         ldb,
                        const double          beta,
                        double*               C,
                        const blasint         ldc)
{
  cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename TFloat>
void gemm_openblas(const Geometry& gg,
                   const Offsets&  toff,
                   const TFloat*   a,
                   const TFloat*   b,
                   TFloat*         c,
                   TFloat          alpha,
                   TFloat          beta)
{

  gemm_openblas_base<TFloat>(gg.isColMajor ? CblasColMajor : CblasRowMajor,
                             gg.tX[Mat::E::A] ? CblasTrans : CblasNoTrans,
                             gg.tX[Mat::E::B] ? CblasTrans : CblasNoTrans,
                             gg.m,
                             gg.n,
                             gg.k,
                             alpha,
                             a + toff.offsets[Mem::E::A],
                             gg.ldX[Mat::E::A],
                             b + toff.offsets[Mem::E::B],
                             gg.ldX[Mat::E::B],
                             beta,
                             c + toff.offsets[Mem::E::C],
                             gg.ldX[Mat::E::C]);
}
}

#endif

namespace custom
{
template <typename TFloat>
class NNInner
{
  public:
  inline TFloat
  operator()(const TFloat* a, const TFloat* b, size_t x, size_t y, size_t lda, size_t ldb, size_t k)
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
class TTInner
{
  public:
  inline TFloat
  operator()(const TFloat* a, const TFloat* b, size_t x, size_t y, size_t lda, size_t ldb, size_t k)
  {
    TFloat inner = 0;
    for (size_t z = 0; z < k; ++z)
    {
      inner += a[x * lda + z] * b[y + z * ldb];
    }
    return inner;
  }
};

template <typename TFloat>
class NTInner
{
  public:
  inline TFloat
  operator()(const TFloat* a, const TFloat* b, size_t x, size_t y, size_t lda, size_t ldb, size_t k)
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
  inline TFloat
  operator()(const TFloat* a, const TFloat* b, size_t x, size_t y, size_t lda, size_t ldb, size_t k)
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
void gemm_3fors_generic(const Geometry& gg,
                        const Offsets&  toff,
                        const TFloat*   a,
                        const TFloat*   b,
                        TFloat*         c,
                        TFloat          alpha,
                        TFloat          beta)
{
  // at this point, must be column contiguous (ala fortran)
  // this is a generic slow matrix multiplier for NN, TN, NT, TT.

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
      if (beta > 0 || beta < 0)
      {
        c[target_index] *= beta;
      }
      else
      {
        c[target_index] = 0;
      }

      c[target_index] += alpha * finner(a, b, x, y, gg.ldX[Mat::E::A], gg.ldX[Mat::E::B], gg.k);
    }
  }
}

template <typename TFloat>
void gemm_3fors(const Geometry& gg,
                const Offsets&  toff,
                const TFloat*   a,
                const TFloat*   b,
                TFloat*         c,
                TFloat          alpha,
                TFloat          beta)
{

  if (gg.tX[Mat::E::C] == true)
  {
    throw miog_error("tC should be false before calling gemm_3fors");
  }

  else if (gg.isColMajor == false)
  {
    throw miog_error("isColMajor should be true before calling gemm_3fors");
  }

  else if (gg.tX[Mat::E::A] == false && gg.tX[Mat::E::B] == false)
  {
    gemm_3fors_generic<TFloat, NNInner<TFloat>>(gg, toff, a, b, c, alpha, beta);
  }

  else if (gg.tX[Mat::E::A] == false && gg.tX[Mat::E::B] == true)
  {
    gemm_3fors_generic<TFloat, NTInner<TFloat>>(gg, toff, a, b, c, alpha, beta);
  }

  else if (gg.tX[Mat::E::A] == true && gg.tX[Mat::E::B] == false)
  {
    gemm_3fors_generic<TFloat, TNInner<TFloat>>(gg, toff, a, b, c, alpha, beta);
  }

  else if (gg.tX[Mat::E::A] == true && gg.tX[Mat::E::B] == true)
  {
    gemm_3fors_generic<TFloat, TTInner<TFloat>>(gg, toff, a, b, c, alpha, beta);
  }

  else
  {
    throw miog_error("this will never happen");
  }
}
}

template <typename TFloat>
void gemm(Geometry        gg,
          Offsets         toff,
          const TFloat*   a,
          const TFloat*   b,
          TFloat*         c,
          TFloat          alpha,
          TFloat          beta,
          owrite::Writer& mowri)
{

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

  redirection::confirm_redirection(gg.isColMajor, gg.tX[Mat::E::C]);
  gg.check_ldx_consistent();
  auto t0 = std::chrono::high_resolution_clock::now();

// dispatch depending on x
#ifdef MIOPENGEMM_USE_OPENBLAS
  mowri << "launching OpenBLAS CPU GEMM algorithm. " << Endl;
  openblas::gemm_openblas<TFloat>(gg, toff, a, b, c, alpha, beta);
#else
  mowri << "launching slow 3-fors CPU GEMM algorithm. " << Endl;
  custom::gemm_3fors<TFloat>(gg, toff, a, b, c, alpha, beta);
#endif  // end of no openblas case

  auto t1           = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  mowri << "elapsed time : " << elapsed_time * 1e-6 << " [s] " << Endl;
}

template void gemm(Geometry        gg,
                   Offsets         toff,
                   const float*    a,
                   const float*    b,
                   float*          c,
                   float           alpha,
                   float           beta,
                   owrite::Writer& mowri);

template void gemm(Geometry        gg,
                   Offsets         toff,
                   const double*   a,
                   const double*   b,
                   double*         c,
                   double          alpha,
                   double          beta,
                   owrite::Writer& mowri);
}
}
