/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_PROBLEMGEOMETRY_HPP
#define GUARD_MIOPENGEMM_PROBLEMGEOMETRY_HPP

#include <string>
#include <vector>
#include <miopengemm/enums.hpp>

/* TODO : namespace should be lower-case. */
namespace MIOpenGEMM
{

// offsets and tails
class Offsets
{
  public:
  std::array<size_t, Mem::E::N> offsets;
  std::array<size_t, Mem::E::N> tails;

  Offsets(size_t oa, size_t ob, size_t oc, size_t ow, size_t ta, size_t tb, size_t tc, size_t tw);
};

Offsets get_padding_offsets();
Offsets get_zero_offsets();

class GeometryDerived
{
  public:
  size_t float_size_bits;
  size_t float_size_bytes;
  void reset(char floattype);
};

class Geometry
{

  private:
  void initialise(bool   isColMajor_,
                  bool   tA_,
                  bool   tB_,
                  bool   tC_,
                  size_t lda_,
                  size_t ldb_,
                  size_t ldc_,
                  size_t m_,
                  size_t n_,
                  size_t k_,
                  size_t wSpaceSize_,
                  char   floattype_);

  public:
  bool isColMajor;

  // indexed by eMat  (for A, B and C)
  std::vector<bool> tX;

  // indexed by eMat (for A, B and C)
  std::vector<size_t> ldX;

  size_t m;
  size_t n;
  size_t k;

  // the usable amount of workspace
  size_t wSpaceSize;

  // 'f' : 32-bit single precision or 'd' : 64-bit double precision
  char floattype;


  // log k ;  log m - log n ;  log m + log n  
  std::array<double, 3> metric_co;

  GeometryDerived derived;

  Geometry(bool   isColMajor,
           bool   tA,
           bool   tB,
           bool   tC,
           size_t lda,
           size_t ldb,
           size_t ldc,
           size_t m,
           size_t n,
           size_t k,
           size_t wSpaceSize,
           char   floattype);

  // assumes isColMajor is true, tC is false, lda, ldb, ldc are minimal.
  Geometry(size_t m, size_t n, size_t k, bool tA, bool tB, size_t wSpaceSize, char floattype);

  Geometry() = default;

  Geometry(const Geometry&) = default;

  Geometry(std::string geometry_string);

  Geometry& operator=(const Geometry&) = default;

  bool operator==(const Geometry&) const;

  size_t get_padless_dim(Mat::E M, bool isCoal) const;

  size_t get_coal(Mat::E M) const;

  size_t get_uncoal(Mat::E M) const;

  size_t get_non_k_dim(Mat::E M) const;

  bool coal_is_pll_k(Mat::E M) const;

  std::string get_string() const;

  std::string get_networkconfig_string() const;

  std::string get_tabbed_string() const;

  void check_ldx_consistent() const;

  size_t get_padded_area(Mat::E M) const;

  // extime is execution time in seconds
  double get_gflops(double extime) const;

  double get_distance(const Geometry& g2) const;

  bool same_transposes(const Geometry& g2) const;
};

template <typename TFloat>
Geometry get_geometry_from_padding(bool   isColMajor,
                                   bool   tA,
                                   bool   tB,
                                   bool   tC,
                                   size_t m,
                                   size_t n,
                                   size_t k,
                                   size_t wSpaceSize,
                                   size_t pad_a,
                                   size_t pad_b,
                                   size_t pad_c)
{
  char floattype;
  switch (sizeof(TFloat))
  {
  case 4: floattype = 'f'; break;
  case 8: floattype = 'd'; break;
  default: throw miog_error("unrecognised float size in get_geometry_from_padding");
  }
  size_t lda = (tA == isColMajor ? k : m) + pad_a;
  size_t ldb = (tB == isColMajor ? n : k) + pad_b;
  size_t ldc = (tC == isColMajor ? n : m) + pad_c;
  return Geometry(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, wSpaceSize, floattype);
}

template <typename TFloat>
Geometry get_padded_geometry(
  bool isColMajor, bool tA, bool tB, bool tC, size_t m, size_t n, size_t k, size_t wSpaceSize)
{
  return get_geometry_from_padding<TFloat>(isColMajor, tA, tB, tC, m, n, k, wSpaceSize, 9, 10, 12);
}

// lda, ldb, ldc are minimal.
template <typename TFloat>
Geometry get_tight_geometry(
  bool isColMajor, bool tA, bool tB, bool tC, size_t m, size_t n, size_t k, size_t wSpaceSize)
{
  return get_geometry_from_padding<TFloat>(isColMajor, tA, tB, tC, m, n, k, wSpaceSize, 0, 0, 0);
}

template <typename TFloat>
Geometry get_squareNN_geometry(size_t m)
{
  return get_geometry_from_padding<TFloat>(true, false, false, false, m,m,m , 0, 0, 0, 0);
}


size_t get_mat_size(const Geometry& gg, const Offsets& toff, Mat::E emat);
size_t get_mat_memsize(const Geometry& gg, const Offsets& toff, Mat::E emat);
size_t get_total_workspace(const Geometry& gg, const Offsets& toff);
}

#endif
