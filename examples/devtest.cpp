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

/* (13/11/2016) most testing is done through dev/python scripts */
/* (30/01/2017) this is the preferred place for doing testing */
/* (17/06/2017) much testing is now done in an indepent project, using OpenBLAS */



std::string get_hyperstring(std::string hyperstring = "")
{
  //if (hyperstring.compare("") == 0)
  //{
    hyperstring = "A_MIC6_PAD1_PLU1_LIW0_MIW0_WOS0__B_MIC5_PAD1_PLU1_LIW0_MIW1_WOS0__C_UNR8_GAL3_PUN1_ICE1_NAW16_UFO0_MAC4_SKW9";
    //hyperstring = "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9";
  //}
  return hyperstring;
}

template <typename TFloat>
MIOpenGEMM::Geometry get_geometry()
{
  //return {"tC0_tA0_tB0_colMaj1_m1024_n8_k10000_lda1024_ldb10000_ldc1024_ws1_f32"};
  return {"tC0_tA1_tB0_colMaj1_m512_n48000_k2816_lda2816_ldb2816_ldc512_ws1_f32"};
}

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

template <typename TFloat>
void print_kernel()
{
  std::string kernel_string;
  std::string hyperstring = get_hyperstring();
  auto        gg          = get_geometry<TFloat>();

  MIOpenGEMM::openclutil::OpenCLDeviceInfo devinfo;
  devinfo.wg_atom_size = 32;
  MIOpenGEMM::hyperparams::Graph          graph(gg, devinfo, hyperstring, true);
  MIOpenGEMM::hyperparams::HyperParams    hp(graph);
  bool                                    mowri_verbose      = true;
  bool                                    verbose_get_bundle = true;
  std::string                             mowri_out("");
  MIOpenGEMM::outputwriting::OutputWriter mowri(mowri_verbose, mowri_out != "", mowri_out);
  auto bundle = MIOpenGEMM::kerngen::get_bundle(hp, gg, mowri, verbose_get_bundle);

  for (auto& x : bundle.v_tgks)
  {
    // set this to the directory to write kernels to
    auto dirname = "/home/james/test/" + gg.get_string() + "/" + get_hyperstring() + "/";
    // WARNING : mkdir only works on linux/mac
    std::string syscall = "mkdir -p " + dirname;
    std::system(syscall.c_str());
    auto fname = dirname + x.type.full + ".cl";
    std::cout << "writing " << fname << " ... " << std::flush;
    std::ofstream floper(fname, std::ios::out);
    floper << x.kernstr;
    floper.close();
    std::cout << "done." << std::endl;
  }
}

int main()
{

  std::string                             fout("");
  MIOpenGEMM::outputwriting::OutputWriter mowri(true, fout != "", fout);

  // what test(s) are you running ?
  bool test_print     = false;
  bool test_benchgemm = false;
  bool test_find      = true;
  bool test_accuracy  = false;
  bool test_default   = false;

  typedef float tfloat;

  MIOpenGEMM::Geometry gg   = get_geometry<tfloat>();
  MIOpenGEMM::Offsets  toff = get_offsets();
  mowri << "generating cpu data ... " << MIOpenGEMM::Flush;
  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;
  MIOpenGEMM::setabcw::set_abc<tfloat>(v_a, v_b, v_c, gg, toff);
  mowri << "done." << MIOpenGEMM::Endl;
  const tfloat* c_true_bla = nullptr;
  if (test_print)
  {
    print_kernel<tfloat>();
  }

  if (test_accuracy || test_benchgemm)
  {
    std::string hyperstring = get_hyperstring();
    if (test_accuracy)
    {
      MIOpenGEMM::dev::accuracy_test(
        hyperstring, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, mowri);
    }

    if (test_benchgemm)
    {
      size_t max_n_runs_benchgemm = 5;
      double max_time_per_kernel = 1e12;
      MIOpenGEMM::dev::benchgemm(
        {hyperstring}, max_n_runs_benchgemm, max_time_per_kernel, gg, toff, v_a.data(), v_b.data(), v_c.data(), mowri);
    }
  }

  if (test_find || test_default)
  {
    // constraint string
    std::string constraints_string(
      "");//A_MIC8_PAD1_PLU0_LIW0_MIW0_WOS0__B_MIC8_PAD1_PLU0_LIW0_MIW0_WOS0__C_UNR64");
    if (test_find)
    {
      float                   allotted_find_time     = 20.00;
      size_t                allotted_find_descents = 1000;
      size_t                max_n_runs_per_kernel      = 2;
      double max_time_per_kernel = 1e12;
      MIOpenGEMM::SummStat::E sumstat(MIOpenGEMM::SummStat::E::MAX);
      MIOpenGEMM::FindParams  find_params(
        allotted_find_time, allotted_find_descents, max_n_runs_per_kernel, max_time_per_kernel, sumstat);
      auto soln = MIOpenGEMM::dev::find(
        find_params, v_a.data(), v_b.data(), v_c.data(), constraints_string, gg, toff, mowri);
      std::cout << "\n\n " << soln.get_cache_entry_string() << "\n\n";
    }

    if (test_default)
    {
      std::string                                   k_comment("");
      MIOpenGEMM::openclutil::CommandQueueInContext tgcq(mowri, "in get_default in devtest.cpp");
      auto                                          soln =
        MIOpenGEMM::get_default(tgcq.command_queue, constraints_string, gg, k_comment, mowri);
      std::cout << soln.hyper_param_string << std::endl;
      MIOpenGEMM::dev::accuracy_test(
        soln.hyper_param_string, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, mowri);
      std::cout << soln.hyper_param_string << std::endl;
    }
  }

  return 0;
}
