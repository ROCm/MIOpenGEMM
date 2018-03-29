/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{

template <>
char get_floattype_char<float>()
{
  return 'f';
}

template <>
char get_floattype_char<double>()
{
  return 'd';
}

Geometry::Geometry(
  size_t m_, size_t n_, size_t k_, bool tA_, bool tB_, size_t wSpaceSize_, char floattype_)
  : Geometry(
      true, tA_, tB_, false, tA_ ? k_ : m_, tB_ ? n_ : k_, m_, m_, n_, k_, wSpaceSize_, floattype_)
{
}

size_t get_mat_size(const Geometry& gg, const Offsets& toff, Mat::E emat)
{
  auto emem = Mem::mat_to_mem(emat);
  return (gg.get_padded_area(emat) + toff.offsets[emem] + toff.tails[emem]);
}

size_t get_mat_memsize(const Geometry& gg, const Offsets& toff, Mat::E emat)
{
  return gg.derived.float_size_bytes * get_mat_size(gg, toff, emat);
}

Offsets::Offsets(
  size_t oa_, size_t ob_, size_t oc_, size_t ow_, size_t ta_, size_t tb_, size_t tc_, size_t tw_)
{

  offsets[Mem::E::A] = oa_;
  offsets[Mem::E::B] = ob_;
  offsets[Mem::E::C] = oc_;
  offsets[Mem::E::W] = ow_;

  tails[Mem::E::A] = ta_;
  tails[Mem::E::B] = tb_;
  tails[Mem::E::C] = tc_;
  tails[Mem::E::W] = tw_;
}

Offsets get_padding_offsets() { return Offsets(11, 17, 13, 23, 67, 15, 29, 17); }

Offsets get_zero_offsets() { return Offsets(0, 0, 0, 0, 0, 0, 0, 0); }

Geometry get_tight_geometry() { return {false, false, false, false, 1, 1, 1, 1, 1, 1, 1, 'f'}; }

char get_floattype(size_t nbits)
{

  char ft = 'x';
  if (nbits == 8 * sizeof(float))
  {
    ft = 'f';
  }
  else if (nbits == 8 * sizeof(double))
  {
    ft = 'd';
  }
  else
  {
    throw miog_error("what is the floattype with number of bints : " + std::to_string(nbits) +
                     std::string(" ? in get_floattype of geometry"));
  }
  return ft;
}

void GeometryDerived::reset(char floattype)
{
  if (floattype == 'f')
  {
    float_size_bytes = sizeof(float);
  }
  else if (floattype == 'd')
  {
    float_size_bytes = sizeof(double);
  }
  else
  {
    throw miog_error("what is this floattype : " + std::to_string(floattype) +
                     std::string(" ? in reset of geometry"));
  }
  float_size_bits = 8 * float_size_bytes;
}

// return one of the dimensions of matrix a,b,c.
// isCoal : the coalesced dimesion?
// For example, for A which is m x k,
// if tA = false, isColMajor = false,
// isCoal = true, then k is returned as k is
// the coalesced dim.
// (false == false) == true  evaluates to true,
// so gate is true, so m is returned
size_t Geometry::get_padless_dim(Mat::E M, bool isCoal) const
{

  bool gate = (tX.at(M) == isColMajor) == isCoal;

  if (M == Mat::E::A)
  {
    return gate ? k : m;
  }

  else if (M == Mat::E::B)
  {
    return gate ? n : k;
  }

  else if (M == Mat::E::C)
  {
    return gate ? n : m;
  }

  else
  {
    throw miog_error("unrecognised M passed to get_coal in "
                     "get_padless_dim of geometry");
  }
}

size_t Geometry::get_non_k_dim(Mat::E M) const
{

  if (M == Mat::E::A)
  {
    return m;
  }

  else if (M == Mat::E::B)
  {
    return n;
  }

  else
  {
    throw miog_error("invalid char passed to get_non_k_dim in get_non_k_dim of "
                     "geometry, it should "
                     "be either a or b");
  }
}

void Geometry::check_ldx_consistent() const
{

  bool error = false;
  for (auto x : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    if (ldX[x] < get_coal(x))
    {
      error = true;
    }
  }

  if (error == true)
  {

    std::stringstream errm_ss;

    errm_ss << "Checking that lda, ldb, ldc are consistent with m,n,k. "
            << "In particulary, checking that ldx is at least as large as "
            << "coalesced dimension, "
            << "coal_x (for x in {a,b,c}) given by:  "
            << "coal_a = (tA == isColMajor ? k : m),  "
            << "coal_b = (tB == isColMajor ? n : k),  "
            << "coal_c = (tC == isColMajor ? n : m).  "
            << "\n\n"
            << "ldx = coal_x + pad_x, and so for consisteny it must be true "
            << "that ldx >= coal_x (can't have negative pad_x).  "
            << "As an example, if tA = false and isColMajor = false, then "
            << "coal_a = k.  "
            << "A full table of the lower bounds of ldx for x in {a,b,c} can "
            << "be found at, "
            << "https://software.intel.com/en-us/"
            << "mkl-developer-reference-c-cblas-gemm.  "
            << "\n\n"
            << "The particular geometry received by in geometry "
            << "check_ldx_consistent is  " << get_string() << ", and the problems detected are:  ";

    for (auto x : {Mat::E::A, Mat::E::B, Mat::E::C})
    {
      if (ldX[x] < get_coal(x))
      {
        errm_ss << "ld" << Mat::M().name[x] << " (" << ldX[x] << ") <  coal_" << Mat::M().name[x]
                << " (" << get_coal(x) << ").  ";
      }
    }

    throw miog_error(errm_ss.str());
  }
}

size_t Geometry::get_uncoal(Mat::E M) const { return get_padless_dim(M, false); }

// this is lda, ldb, ldc if they are minimal.
size_t Geometry::get_coal(Mat::E M) const { return get_padless_dim(M, true); }

bool Geometry::coal_is_pll_k(Mat::E M) const
{
  // proof : false, false, true should give 1
  return (static_cast<size_t>(isColMajor) + static_cast<size_t>(tX.at(M)) +
          static_cast<size_t>(M == Mat::E::A)) %
         2;
}

void Geometry::initialise(bool   isColMajor_,
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
                          char   floattype_)
{

  isColMajor = isColMajor_;
  m          = m_;
  n          = n_;
  k          = k_;
  wSpaceSize = wSpaceSize_;
  floattype  = floattype_;

  tX.resize(Mat::E::N);
  tX[Mat::E::A] = tA_;
  tX[Mat::E::B] = tB_;
  tX[Mat::E::C] = tC_;

  ldX.resize(Mat::E::N);
  ldX[Mat::E::A] = lda_;
  ldX[Mat::E::B] = ldb_;
  ldX[Mat::E::C] = ldc_;

  if (floattype != 'd' and floattype != 'f')
  {
    throw miog_error("floattype should be one of 'f' and 'd' (in Geometry constructor)");
  }

  check_ldx_consistent();

  derived.reset(floattype);

  metric_co[0] = std::log2(static_cast<double>(k));
  metric_co[1] = std::log2(static_cast<double>(m)) - std::log2(static_cast<double>(n));
  metric_co[2] = std::log2(static_cast<double>(m)) + std::log2(static_cast<double>(n));

  metric_co[3] = 0.2 * std::log2(static_cast<double>(ldX[Mat::E::A]));
  metric_co[4] = 0.2 * std::log2(static_cast<double>(ldX[Mat::E::B]));
  metric_co[5] = 0.2 * std::log2(static_cast<double>(ldX[Mat::E::C]));

  // memory required for copying (an estimate)
  std::array<size_t, Mat::E::N> forPadCopy;
  for (auto emat : {Mat::E::A, Mat::E::B})
  {
    forPadCopy[emat] = get_uncoal(emat) * (get_coal(emat) + 16);
  }

  wSpaceSufficient[0] = forPadCopy[Mat::E::A] < wSpaceSize;
  wSpaceSufficient[1] = forPadCopy[Mat::E::B] < wSpaceSize;
  wSpaceSufficient[2] = 1 * (forPadCopy[Mat::E::A] + forPadCopy[Mat::E::B]) < wSpaceSize;
  wSpaceSufficient[3] = 2 * (forPadCopy[Mat::E::A] + forPadCopy[Mat::E::B]) < wSpaceSize;
  wSpaceSufficient[4] = 4 * (forPadCopy[Mat::E::A] + forPadCopy[Mat::E::B]) < wSpaceSize;
}

std::map<std::string, size_t> get_key_val_map(std::string geometry_string)
{
  auto frags = stringutil::split(geometry_string, "_");
  std::map<std::string, size_t> key_val_map;
  for (auto& frag : frags)
  {
    auto key_val     = stringutil::splitnumeric(frag);
    auto key         = std::get<0>(key_val);
    auto val         = std::get<1>(key_val);
    key_val_map[key] = val;
  }
  return key_val_map;
}

size_t safeat(std::map<std::string, size_t>& map, std::string key)
{
  if (map.count(key) == 0)
  {
    std::stringstream errm;
    errm << "Unrecognised key `";
    errm << key << "' in safeat of geometry";
    throw miog_error(errm.str());
  }
  return map.at(key);
}

Geometry::Geometry(std::string geometry_string)
{

  auto key_val_map = get_key_val_map(geometry_string);

  Geometry goldstandard_geometry(
    false, false, false, false, 100, 100, 100, 100, 100, 100, 100, 'f');
  std::string goldstandard_geometry_string = goldstandard_geometry.get_string();
  auto        goldstandard_map             = get_key_val_map(goldstandard_geometry_string);

  std::stringstream errm_ss;
  bool              good_string{true};
  for (auto& x : key_val_map)
  {
    if (goldstandard_map.count(x.first) == 0)
    {
      errm_ss << "The key in the geometry string `" << x.first << "' is not valid.  ";
      good_string = false;
    }
  }

  for (auto& x : goldstandard_map)
  {
    if (key_val_map.count(x.first) == 0)
    {
      errm_ss << "The geometry string should contain key `" << x.first << "', but does not.  ";
      good_string = false;
    }
  }

  if (good_string == false)
  {
    throw miog_error(errm_ss.str());
  }

  initialise(safeat(key_val_map, "colMaj"),
             safeat(key_val_map, "tA"),
             safeat(key_val_map, "tB"),
             safeat(key_val_map, "tC"),
             safeat(key_val_map, "lda"),
             safeat(key_val_map, "ldb"),
             safeat(key_val_map, "ldc"),
             safeat(key_val_map, "m"),
             safeat(key_val_map, "n"),
             safeat(key_val_map, "k"),
             safeat(key_val_map, "ws"),
             get_floattype(safeat(key_val_map, "f")));
}

std::string Geometry::get_string() const { return get_networkconfig_string(); }

std::string Geometry::get_networkconfig_string() const
{
  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC" << tX[Mat::E::C] << "_tA" << tX[Mat::E::A] << "_tB" << tX[Mat::E::B]
                        << "_colMaj" << isColMajor << "_m" << m << "_n" << n << "_k" << k << "_lda"
                        << ldX[Mat::E::A] << "_ldb" << ldX[Mat::E::B] << "_ldc" << ldX[Mat::E::C]
                        << "_ws" << wSpaceSize << "_f" << derived.float_size_bits;
  return geometry_stringstream.str();
}

std::string Geometry::get_tabbed_string() const
{

  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC=" << tX[Mat::E::C] << " tA=" << tX[Mat::E::A]
                        << " tB=" << tX[Mat::E::B] << " colMaj=" << isColMajor
                        << " m=" << stringutil::get_char_padded(m, 6)
                        << " n=" << stringutil::get_char_padded(n, 6)
                        << " k=" << stringutil::get_char_padded(k, 6)
                        << " lda=" << stringutil::get_char_padded(ldX[Mat::E::A], 6)
                        << " ldb=" << stringutil::get_char_padded(ldX[Mat::E::B], 6)
                        << " ldc=" << stringutil::get_char_padded(ldX[Mat::E::C], 6)
                        << " ws=" << wSpaceSize << " f=" << derived.float_size_bits;

  return geometry_stringstream.str();
}

size_t Geometry::get_padded_area(Mat::E M) const { return get_uncoal(M) * ldX[M]; }

// Safer would be compare via get_string(), assuming get_string() is comprehensive.
bool Geometry::operator==(const Geometry& rhs) const
{
  return (isColMajor == rhs.isColMajor && tX == rhs.tX && ldX == rhs.ldX && m == rhs.m &&
          n == rhs.n && k == rhs.k && wSpaceSize == rhs.wSpaceSize && floattype == rhs.floattype);
}

double Geometry::get_gflops(double extime) const { return (2. * m * n * k) / (1e9 * extime); }

bool Geometry::same_transposes(const Geometry& g2) const
{
  return ((tX == g2.tX) && (isColMajor == g2.isColMajor));
}

double Geometry::get_distance(const Geometry& g2) const
{

  double distance = 0;
  if (same_transposes(g2) == false)
  {
    distance = std::numeric_limits<double>::max();
  }

  else
  {

    for (unsigned i = 0; i < 6; ++i)
    {
      distance += std::abs(metric_co[i] - g2.metric_co[i]);
    }
    for (size_t x : {2, 4, 8})
    {
      for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
      {
        distance += 0.2 * ((ldX[emat] % x == 0) != (g2.ldX[emat] % x == 0));
      }
    }

    for (size_t x : {256, 512, 1024})
    {
      for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
      {
        distance += 0.2 * (std::min<size_t>(ldX[emat] % x, x - ldX[emat] % x) % 4 !=
                           std::min<size_t>(g2.ldX[emat] % x, x - g2.ldX[emat] % x) % 4);
      }
    }

    for (size_t i = 0; i < wSpaceSufficient.size(); ++i)
    {
      distance += 0.2 * (wSpaceSufficient[i] != g2.wSpaceSufficient[i]);
    }
  }

  distance += 1e-5 * (std::log(wSpaceSize + 1.1) - std::log(g2.wSpaceSize + 1.1));

  return distance;
}

size_t get_total_workspace(const Geometry& gg, const Offsets& toff)
{
  return gg.wSpaceSize + toff.offsets[Mem::E::W] + toff.tails[Mem::E::W];
}
}
