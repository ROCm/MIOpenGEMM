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
  
  Offsets(size_t oa,
          size_t ob,
          size_t oc,
          size_t ow,
          size_t ta,
          size_t tb,
          size_t tc,
          size_t tw);

  //const size_t& operator[](char c) const;
};

Offsets get_padding_offsets();

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
  

  size_t get_padded_area(Mat::E M) const;
  
};


// TODO : move to cpp 
template<typename TFloat>
MIOpenGEMM::Geometry get_padded_geometry(bool isColMajor, bool tA, bool tB, bool tC, size_t m, size_t n, size_t k, size_t workspace_size){
  char floattype = sizeof(TFloat) == 4 ? 'f' : 'd';
  size_t lda = (tA == isColMajor ? k : m) + 9;
  size_t ldb = (tB == isColMajor ? n : k) + 10;
  size_t ldc = (tC == isColMajor ? n : m) + 12;
  return  MIOpenGEMM::Geometry(
    isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
}
  
}

#endif
