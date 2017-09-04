/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_PROBLEMGEOMETRY_HPP
#define GUARD_MIOPENGEMM_PROBLEMGEOMETRY_HPP

#include <string>
#include <vector>
#include <miopengemm/enums.hpp>

// TODO : namespace should be lower-case
namespace MIOpenGEMM
{

/*! @brief
 *  encapsulate buffering of memories, before and after  */
class Offsets
{
  public:
  /*! buffering before: number of values before the first value which may be accessed */
  std::array<size_t, Mem::E::N> offsets;
  /*! buffering after: number of values after the last value which may be accessed.
   * This tail buffering is only used for debugging purposes */
  std::array<size_t, Mem::E::N> tails;

  /*! @brief
   * constructor, first the 4 offsets (before memory) and then the 4 after (tail) */
  Offsets(size_t oa, size_t ob, size_t oc, size_t ow, size_t ta, size_t tb, size_t tc, size_t tw);
};

/*! @brief
 * factory function to retun Offsets of all non-zeros */
Offsets get_padding_offsets();

/*! @brief
 * factory function to retun Offsets of all zeros */
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

  private:
  // log k ;  log m - log n ;  log m + log n, 0.2*(log ldx's)
  std::array<double, 6> metric_co;
  std::array<bool, 5>   wSpaceSufficient;

  public:
  bool isColMajor;

  /*! transpose cases, index by Mat::E::A, Mat::E::B, Mat::E::C.
   *  the option of allowing transpose C is not in the standard GEMM API and
   *  is not necessary, but can simplify GEMM geometry remapping */
  std::vector<bool> tX;

  /*! leading dimensions, index by Mat::E::A, Mat::E::B, Mat::E::C. */
  std::vector<size_t> ldX;

  size_t m;
  size_t n;
  size_t k;

  /*! usable amount of workspace, in number of values (i.e. not in bytes). */
  size_t wSpaceSize;

  // TODO : investigate optional half-precision 'h'
  // TODO : rename from floattype to numerictype, and consider integer matrix multiplication
  /*! float type of values, currently either 'f' (32-bit single precision)
   *  or 'd' (64-bit double precision). */
  char floattype;

  public:
  GeometryDerived derived;

  template <typename T>
  Geometry(bool   isColMajor_,
           bool   tA_,
           bool   tB_,
           bool   tC_,
           T      lda_,
           T      ldb_,
           T      ldc_,
           T      m_,
           T      n_,
           T      k_,
           size_t wSpaceSize_,
           char   floattype_)
  {
    initialise(isColMajor_, tA_, tB_, tC_, lda_, ldb_, ldc_, m_, n_, k_, wSpaceSize_, floattype_);
  }

  /*! @brief
   * Constructor which assumes isColMajor is true, tC is false, lda, ldb, ldc are minimal */
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

  /*! @brief
   * extime is execution time in seconds */
  double get_gflops(double extime) const;

  double get_distance(const Geometry& g2) const;

  bool same_transposes(const Geometry& g2) const;
};

template <typename TFloat>
char get_floattype_char()
{
  throw miog_error("unrecognised float type");
}

template <>
char get_floattype_char<float>();

template <>
char get_floattype_char<double>();

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

  char floattype = get_floattype_char<TFloat>();

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
  return get_geometry_from_padding<TFloat>(true, false, false, false, m, m, m, 0, 0, 0, 0);
}

size_t get_mat_size(const Geometry& gg, const Offsets& toff, Mat::E emat);
size_t get_mat_memsize(const Geometry& gg, const Offsets& toff, Mat::E emat);
size_t get_total_workspace(const Geometry& gg, const Offsets& toff);
}

#endif
