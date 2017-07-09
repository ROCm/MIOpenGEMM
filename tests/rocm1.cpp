/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <iostream>
#include <sstream>
#include <vector>
#include <miopengemm/bundle.hpp>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/stringutilbase.hpp>
#include <miopengemm/setabcw.hpp>


MIOpenGEMM::Offsets get_offsets()
{
  size_t a_offset         = 0;
  size_t b_offset         = 0;
  size_t c_offset         = 0;
  size_t workspace_offset = 4;
  size_t tail_off_a       = 0;
  size_t tail_off_b       = 0;
  size_t tail_off_c       = 0;
  return {a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c};
}




int main()
{

  /* crashes if all ints are unsigned in OpenCL kernel. Changing certain to ushort stops crash */
  std::string hyperstring("A_MIC6_PAD1_PLU1_LIW0_MIW0_WOS0__B_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__C_UNR8_GAL3_PUN1_ICE1_NAW16_UFO0_MAC4_SKW9");
  MIOpenGEMM::Geometry gg("tC0_tA0_tB0_colMaj1_m1024_n8_k10000_lda1024_ldb10000_ldc1024_ws1_f32");
                           

  // gives incorrect results. Unresolved. 
  //hyperstring = "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9";
  //"tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws1_f32"

  
  std::string                             fout("");
  MIOpenGEMM::outputwriting::OutputWriter mowri(true, fout != "", fout);

  typedef float tfloat;

  MIOpenGEMM::Offsets toff = get_offsets();
  mowri << "generating cpu data ... " << MIOpenGEMM::Flush;
  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;
  MIOpenGEMM::setabcw::set_abc<tfloat>(v_a, v_b, v_c, gg, toff);
  mowri << "done." << MIOpenGEMM::Endl;


  size_t n_runs_benchgemm = 5;
  MIOpenGEMM::dev::benchgemm(
    {hyperstring}, n_runs_benchgemm, gg, toff, v_a.data(), v_b.data(), v_c.data(), mowri);
    
  return 0;
}
