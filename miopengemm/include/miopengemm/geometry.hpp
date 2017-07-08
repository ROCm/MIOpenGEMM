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
  // offsets of a,b,c and workspace
  size_t oa;
  size_t ob;
  size_t oc;
  size_t oworkspace;

  // tails of a,b and c. why no workspace?
  size_t tail_off_a;
  size_t tail_off_b;
  size_t tail_off_c;

  Offsets(size_t oa,
          size_t ob,
          size_t oc,
          size_t oworkspace,
          size_t tail_off_a,
          size_t tail_off_b,
          size_t tail_off_c);

  const size_t& operator[](char c) const;
};

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
  void initialise(bool     isColMajor_,
                  bool     tA_,
                  bool     tB_,
                  bool     tC_,
                  size_t lda_,
                  size_t ldb_,
                  size_t ldc_,
                  size_t m_,
                  size_t n_,
                  size_t k_,
                  size_t workspace_size_,
                  char     floattype_);

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
  size_t workspace_size;

  // 'f' : 32-bit single precision or 'd' : 64-bit double precision
  char floattype;

  GeometryDerived derived;

  /* TODO : decide on style :  workspace_size vs workspaceSize. */
  Geometry(bool     isColMajor,
           bool     tA,
           bool     tB,
           bool     tC,
           size_t lda,
           size_t ldb,
           size_t ldc,
           size_t m,
           size_t n,
           size_t k,
           size_t workspace_size,
           char     floattype);

  Geometry() = default;

  Geometry(const Geometry&) = default;

  Geometry(std::string geometry_string);

  Geometry& operator=(const Geometry&) = default;

  size_t get_padless_dim(Mat::E M, bool isCoal) const;

  size_t get_coal(Mat::E M) const;

  size_t get_uncoal(Mat::E M) const;

  size_t get_non_k_dim(Mat::E M) const;

  bool coal_is_pll_k(Mat::E M) const;

  std::string get_string() const;

  std::string get_networkconfig_string() const;
  
  std::string get_tabbed_string() const;

  void check_ldx_consistent() const;
};
}

#endif
