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

size_t get_mat_memsize(const Geometry& gg, const Offsets& toff, Mem::E emem)
{
  Mat::E emat = Mat::mem_to_mat(emem);
  return gg.derived.float_size_bytes *
         (gg.get_padded_area(emat) + toff.offsets[emem] + toff.tails[emem]);
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

Offsets get_padding_offsets() { return Offsets(11, 17, 13, 22, 61, 15, 18, 7); }

Offsets get_zero_offsets() { return Offsets(0, 0, 0, 0, 0, 0, 0, 0); }

Geometry get_null_geometry() { return {false, false, false, false, 1, 1, 1, 1, 1, 1, 1, 'f'}; }

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
    throw miog_error("what is this floattype : " + floattype +
                     std::string(" ? in reset of geometry"));
  }
  float_size_bits = 8 * float_size_bytes;
}

// return one of the dimensions of matrix a,b,c.
// this has nothing to do with lda, ldb, ldc.
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

    errm_ss << "Checking that lda, ldb, ldc are consistent with m,n,k. ";
    errm_ss << "In particulary, checking that ldx is at least as large as "
               "coalesced dimension, "
               "coal_x (for x in {a,b,c}) given by:  ";
    errm_ss << "coal_a = (tA == isColMajor ? k : m),  ";
    errm_ss << "coal_b = (tB == isColMajor ? n : k),  ";
    errm_ss << "coal_c = (tC == isColMajor ? k : m).  ";
    errm_ss << "\n\n";

    errm_ss << "ldx = coal_x + pad_x, and so for consisteny it must be true "
               "that ldx >= coal_x "
               "(can't have negative pad_x).  ";
    errm_ss << "As an example, if tA = false and isColMajor = false, then "
               "coal_a = k.  ";
    errm_ss << "A full table of the lower bounds of ldx for x in {a,b,c} can "
               "be found at, "
               "https://software.intel.com/en-us/"
               "mkl-developer-reference-c-cblas-gemm.  ";
    errm_ss << "\n\n";

    errm_ss << "The particular geometry received by in geometry "
               "check_ldx_consistent is  ";
    errm_ss << get_string();
    errm_ss << ", and the problems detected are:  ";

    for (auto x : {Mat::E::A, Mat::E::B, Mat::E::C})
    {
      if (ldX[x] < get_coal(x))
      {
        errm_ss << "ld" << Mat::M.name[x] << " (" << ldX[x] << ") <  coal_" << Mat::M.name[x]
                << " (" << get_coal(x) << ").  ";
      }
    }

    throw miog_error(errm_ss.str());
  }
}

size_t Geometry::get_uncoal(Mat::E M) const { return get_padless_dim(M, false); }

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
                          size_t workspace_size_,
                          char   floattype_)
{

  isColMajor     = isColMajor_;
  m              = m_;
  n              = n_;
  k              = k_;
  workspace_size = workspace_size_;
  floattype      = floattype_;

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
}

Geometry::Geometry(bool   isColMajor_,
                   bool   tA_,
                   bool   tB_,
                   bool   tC_,
                   size_t lda_,
                   size_t ldb_,
                   size_t ldc_,
                   size_t m_,
                   size_t n_,
                   size_t k_,
                   size_t workspace_size_,
                   char   floattype_)
{
  initialise(isColMajor_, tA_, tB_, tC_, lda_, ldb_, ldc_, m_, n_, k_, workspace_size_, floattype_);
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
                        << "_ws" << workspace_size << "_f" << derived.float_size_bits;
  return geometry_stringstream.str();
}

std::string Geometry::get_tabbed_string() const
{

  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC=" << tX[Mat::E::C] << " tA=" << tX[Mat::E::A]
                        << " tB=" << tX[Mat::E::B] << " colMaj=" << isColMajor
                        << " m=" << stringutil::get_char_padded(m, 5)
                        << " n=" << stringutil::get_char_padded(n, 5)
                        << " k=" << stringutil::get_char_padded(k, 5)
                        << " lda=" << stringutil::get_char_padded(ldX[Mat::E::A], 5)
                        << " ldb=" << stringutil::get_char_padded(ldX[Mat::E::B], 5)
                        << " ldc=" << stringutil::get_char_padded(ldX[Mat::E::C], 5)
                        << " ws=" << workspace_size << " f=" << derived.float_size_bits;

  return geometry_stringstream.str();
}

size_t Geometry::get_padded_area(Mat::E M) const { return get_uncoal(M) * ldX[M]; }
}
