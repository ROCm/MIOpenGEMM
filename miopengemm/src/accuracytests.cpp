/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <cmath>
#include <vector>
#include <miopengemm/accuracytests.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{

namespace accuracytests
{

enum class Status
{
  UNCHECKED = 0,
  INCORRECT,
  CORRECT
};

/* TODO : document logic. are matrices positive or zero centeted?
 * Provide better diagnostics when not correct.
 * Output full matrices to files for visualisation. etc etc etc. */

template <typename T>
bool exactly_equal(T a, T b)
{
  return ((std::isnan(a) && std::isnan(b)) || (a >= b && a <= b));
}

template <typename TFloat>
void elementwise_compare(const Geometry& gg,
                         const Offsets&  toff,
                         const TFloat*   c_before,
                         const TFloat*   c_cpu,
                         const TFloat*   c_gpu,
                         const TFloat*   c_cpu_abs,
                         std::string     info_str,
                         owrite::Writer& mowri)
{
  double threshold      = 1e-6;
  size_t nels           = get_mat_size(gg, toff, Mat::E::C);
  size_t n_mat_els      = gg.get_padded_area(Mat::E::C);
  size_t n_errs_printed = 0;
  double max_abs_err    = 0;
  double max_rel_err    = 0;
  double max_test_err   = 0;
  size_t n_per_category = 25;  // number of errors to print for the 4 different regions.

  std::vector<Status> status(nels, Status::UNCHECKED);
  std::stringstream   errm;

  errm << info_str << '\n';

  auto get_message = [c_before, c_cpu, c_gpu, c_cpu_abs, &gg, &toff](size_t i) {
    std::stringstream ss;
    ss << "\nc_before : " << c_before[i] << "   c_cpu : " << c_cpu[i] << "   c_gpu : " << c_gpu[i]
       << "   c_cpu_abs : " << c_cpu_abs[i] << "\n\n";
    return ss.str();
  };

  size_t zone = 1;
  // First check the pre-padding zone,
  for (size_t i = 0; i < toff.offsets[Mem::E::C]; ++i)
  {
    status[i] = exactly_equal(c_cpu[i], c_gpu[i]) ? Status::CORRECT : Status::INCORRECT;
    if (status[i] == Status::INCORRECT && n_errs_printed < zone * n_per_category)
    {
      ++n_errs_printed;
      errm << "(in offset, " << i << '/' << toff.offsets[Mem::E::C] << ')' << get_message(i);
    }
  }
  ++zone;

  // Now check the post-padding zone,
  for (size_t i = toff.offsets[Mem::E::C] + n_mat_els; i < nels; ++i)
  {
    status[i] = exactly_equal(c_cpu[i], c_gpu[i]) ? Status::CORRECT : Status::INCORRECT;
    if (status[i] == Status::INCORRECT && n_errs_printed < zone * n_per_category)
    {
      ++n_errs_printed;
      errm << "(in tail, " << nels - i << '/' << toff.tails[Mem::E::C] << ')' << get_message(i);
    }
  }
  ++zone;

  // Now check the matrix proper zone,
  for (size_t i = 0; i < gg.get_uncoal(Mat::E::C); ++i)
  {
    for (size_t j = 0; j < gg.get_coal(Mat::E::C); ++j)
    {
      size_t coord = toff.offsets[Mem::E::C] + i * gg.ldX[Mat::E::C] + j;
      max_abs_err =
        std::max<double>(max_abs_err, static_cast<double>(std::abs(c_cpu[coord] - c_gpu[coord])));
      max_rel_err    = max_abs_err / (std::abs(static_cast<double>(c_cpu[coord])) + 1e-9);
      double relerr1 = static_cast<double>(std::abs(c_cpu[coord] - c_gpu[coord])) /
                       (std::max<double>(static_cast<double>(c_cpu_abs[coord]), 1e-9));

      max_test_err = std::max<double>(relerr1, max_test_err);

      status[coord] = relerr1 > threshold ? Status::INCORRECT : Status::CORRECT;
      if (status[coord] == Status::INCORRECT && n_errs_printed < zone * n_per_category)
      {
        ++n_errs_printed;
        errm << "(in matrix zone, "
             << "uncoal = " << i << "/" << gg.get_uncoal(Mat::E::C) << ", coal = " << j << "/"
             << gg.ldX[Mat::E::C] << ")\n"
             << "abs(cpu - gpu)/max(absgemm, 1e-9)=" << relerr1 << ">" << threshold << ". "
             << get_message(coord);
      }
    }
  }
  ++zone;

  // Finally, check the matrix ldx zone,
  for (size_t i = 0; i < gg.get_uncoal(Mat::E::C); ++i)
  {
    for (size_t j = gg.get_coal(Mat::E::C); j < gg.ldX[Mat::E::C]; ++j)
    {
      size_t coord = toff.offsets[Mem::E::C] + i * gg.ldX[Mat::E::C] + j;

      status[coord] =
        exactly_equal(c_cpu[coord], c_gpu[coord]) ? Status::CORRECT : Status::INCORRECT;

      if (status[coord] == Status::INCORRECT && n_errs_printed < zone * n_per_category)
      {
        ++n_errs_printed;
        errm << "(in ldX zone, "
             << "uncoal = " << i << "/" << gg.get_uncoal(Mat::E::C) << ", coal = " << j << "/"
             << gg.ldX[Mat::E::C] << ")" << get_message(coord);
      }
    }
  }
  ++zone;

  bool was_error = false;
  for (size_t i = 0; i < nels; ++i)
  {
    if (status[i] == Status::UNCHECKED)
    {
      std::stringstream err2;
      err2 << "logic error, index " << i << " was not checked.";
      throw miog_error(err2.str());
    }
    else if (status[i] == Status::INCORRECT)
    {
      was_error = true;
    }
  }

  if (was_error)
  {
    throw miog_error(errm.str());
  }

  mowri.bw[OutPart::E::ACC] << '[' << "max_abs_err=" << max_abs_err
                            << "   max_rel_err=" << max_rel_err
                            << "   max_test_err=" << max_test_err << ']' << Flush;
}

template void elementwise_compare(const Geometry& gg,
                                  const Offsets&  toff,
                                  const float*    c_before,
                                  const float*    c_cpu,
                                  const float*    c_gpu,
                                  const float*    c_cpu_abs,
                                  std::string,
                                  owrite::Writer& mowri);

template void elementwise_compare(const Geometry& gg,
                                  const Offsets&  toff,
                                  const double*   c_before,
                                  const double*   c_cpu,
                                  const double*   c_gpu,
                                  const double*   c_cpu_abs,
                                  std::string,
                                  owrite::Writer& mowri);
}
}
