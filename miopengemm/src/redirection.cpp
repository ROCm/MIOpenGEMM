/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <iostream>
#include <miopengemm/error.hpp>
#include <miopengemm/platform.hpp>
#include <miopengemm/redirection.hpp>

namespace MIOpenGEMM
{
namespace redirection
{

// transform so that is Column Major and tC = is false.
template <typename T>
void redirect_base(bool& isColMajor, bool& tA, bool& tB, bool& tC, size_t& m, size_t& n, T& a, T& b)
{
  if (isColMajor == false)
  {
    // perform minimal changes to get into col major
    std::swap(tA, tB);
    std::swap(a, b);
    std::swap(m, n);
    isColMajor = true;
    // it might still be tC == 1, redirect again
    redirect_base<T>(isColMajor, tA, tB, tC, m, n, a, b);
  }

  else if (tC == true)
  {
    tC          = false;
    auto old_tA = tA;
    tA          = tB == true ? false : true;
    tB          = old_tA == true ? false : true;
    std::swap(a, b);
    std::swap(m, n);
  }
}

// transform so that is Column Major, is not TT, and m > n if not NN.
template <typename T>
void redirect_base_mlessn(
  bool& isColMajor, bool& tA, bool& tB, bool& tC, size_t& m, size_t& n, T& a, T& b)
{
  if (isColMajor == false)
  {
    // perform minimal changes to get into row major
    std::swap(tA, tB);
    std::swap(a, b);
    std::swap(m, n);
    isColMajor = true;
    // it might still be TT or m < n && tA + tB == 1, redirect again
    redirect_base_mlessn<T>(isColMajor, tA, tB, tC, m, n, a, b);
  }

  else if (tA == true && tB == true)
  {
    tC = tC == true ? false : true;
    tA = false;
    tB = false;
    std::swap(a, b);
    std::swap(m, n);
  }

  else if (m > n && ((tA == true && tB == false) || (tA == false && tB == true)))
  {
    tC = tC == true ? false : true;
    std::swap(a, b);
    std::swap(m, n);
  }
}

template <typename TFloat>
class MatrixBundle
{
  public:
  const TFloat* x;
  size_t        ldx;
  size_t        x_offset;
  MatrixBundle(const TFloat* x_, size_t ldx_, size_t x_offset_)
    : x(x_), ldx(ldx_), x_offset(x_offset_)
  {
  }
};

template <typename TFloat>
void redirect(bool&          isColMajor,
              bool&          tA,
              bool&          tB,
              bool&          tC,
              size_t&        m,
              size_t&        n,
              size_t&        lda,
              size_t&        ldb,
              size_t&        a_offset,
              size_t&        b_offset,
              const TFloat*& a,
              const TFloat*& b)
{

  MatrixBundle<TFloat> a_bundle(a, lda, a_offset);
  MatrixBundle<TFloat> b_bundle(b, ldb, b_offset);
  redirect_base<MatrixBundle<TFloat>>(isColMajor, tA, tB, tC, m, n, a_bundle, b_bundle);

  a        = a_bundle.x;
  lda      = a_bundle.ldx;
  a_offset = a_bundle.x_offset;

  b        = b_bundle.x;
  ldb      = b_bundle.ldx;
  b_offset = b_bundle.x_offset;
}

template void redirect(bool&          isColMajor,
                       bool&          tA,
                       bool&          tB,
                       bool&          tC,
                       size_t&        m,
                       size_t&        n,
                       size_t&        lda,
                       size_t&        ldb,
                       size_t&        a_offset,
                       size_t&        b_offset,
                       const double*& a,
                       const double*& b);

template void redirect(bool&         isColMajor,
                       bool&         tA,
                       bool&         tB,
                       bool&         tC,
                       size_t&       m,
                       size_t&       n,
                       size_t&       lda,
                       size_t&       ldb,
                       size_t&       a_offset,
                       size_t&       b_offset,
                       const float*& a,
                       const float*& b);

void redirect(bool&        isColMajor,
              bool&        tA,
              bool&        tB,
              bool&        tC,
              size_t&      m,
              size_t&      n,
              std::string& a,
              std::string& b)
{
  redirect_base<std::string>(isColMajor, tA, tB, tC, m, n, a, b);
}

void confirm_redirection_mlessn(bool isColMajor, bool tA, bool tB, size_t m, size_t n)
{
  std::string errmbase = "redirection_mlessn failed or not performed";
  if (isColMajor == false)
  {
    throw miog_error("isColMajor == false : " + errmbase);
  }

  else
  {
    if (tA == true && tB == true)
    {
      throw miog_error("both matrices transposed : " + errmbase);
    }

    else if ((tA == true && tB == false) || (tA == false && tB == true))
    {
      if (m > n)
      {
        throw miog_error("tA + tB = 1 with m > n : " + errmbase);
      }
    }
  }
}

void confirm_redirection(bool isColMajor, bool tC)
{
  if (isColMajor == false)
  {
    throw miog_error("isColMajor == false : redirection failed or not performed");
  }

  else if (tC == true)
  {
    throw miog_error("tC == true : redirection failed or not performed ");
  }
}

class SimpleBundle
{
  public:
  size_t ldx;
  Mat::E emat;
  SimpleBundle(size_t ldx_, Mat::E e_) : ldx(ldx_), emat(e_) {}
};

Geometry get_canonical(const Geometry& gg, bool& swap_ab)
{
  bool         isColMajor = gg.isColMajor;
  bool         tA         = gg.tX[Mat::E::A];
  bool         tB         = gg.tX[Mat::E::B];
  bool         tC         = gg.tX[Mat::E::C];
  size_t       m          = gg.m;
  size_t       n          = gg.n;
  SimpleBundle sba(gg.ldX[Mat::E::A], Mat::E::A);
  SimpleBundle sbb(gg.ldX[Mat::E::B], Mat::E::B);
  redirect_base(isColMajor, tA, tB, tC, m, n, sba, sbb);
  swap_ab = (sba.emat == Mat::E::B);
  return {isColMajor,
          tA,
          tB,
          tC,
          sba.ldx,
          sbb.ldx,
          gg.ldX[Mat::E::C],
          m,
          n,
          gg.k,
          gg.wSpaceSize,
          gg.floattype};
}

Geometry get_canonical(const Geometry& gg)
{
  bool swap_ab{};
  return get_canonical(gg, swap_ab);
}

bool get_is_not_canonical(const Geometry& gg)
{
  bool swap_ab;
  auto gg2 = get_canonical(gg, swap_ab);
  return swap_ab;
}
}
}
