/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>
#include <miopengemm/error.hpp>
#include <miopengemm/randomutil.hpp>
#include <miopengemm/setabcw.hpp>

namespace MIOpenGEMM
{
namespace setabcw
{

template <typename TFloat>
void fill_uni(std::vector<TFloat>& v, size_t r_small, size_t r_big)
{

  if (r_small > r_big)
  {
    std::stringstream ss;
    ss << "strange request : in fill_uni, with r_small > r_big";
    throw miog_error(ss.str());
  }

  if (r_small > v.size())
  {
    throw miog_error("strange request : in fill_uni, r_small > v.size()");
  }

  if (r_big > v.size())
  {
    throw miog_error("strange request : in fill_uni, r_big > v.size()");
  }

  // previously, had larger values in pad region.
  (void)r_small;

  unsigned int             n_threads = 4;
  std::vector<std::thread> threads;
  for (unsigned int t = 0; t < n_threads; ++t)
  {

    threads.emplace_back([n_threads, r_big, t, &v]() {
      // this makes the values deterministic, consider using random_device for initialisation
      unsigned int g_seed = t;
      for (auto i = (t * r_big) / n_threads; i < ((t + 1) * r_big) / n_threads; ++i)
      {
        // fast generation of low quality pseudo-random, see blog at
        // fast-random-number-generator-on-the-intel-pentiumr-4-processor
        g_seed                 = (214013 * g_seed + 2531011);
        unsigned int new_value = (g_seed >> 16) & 0x7FFF;
        v[i]                   = TFloat(-1) + (new_value % 2000) / TFloat(1000);
      }
    });
  }

  for (auto& t : threads)
  {
    t.join();
  }
}

template <typename TFloat>
void set_multigeom_abc(const MatData<TFloat>&       v_abc,
                       const std::vector<Geometry>& ggs,
                       const Offsets&               toff)
{
  if (v_abc.size() != Mat::E::N)
  {
    throw miog_error("vector should contain Mat::E::N (3) pointers in set_multigeom_abc");
  }

  std::vector<size_t> max_n(Mat::E::N, 0);
  for (auto& gg : ggs)
  {
    if (gg.derived.float_size_bytes != sizeof(TFloat))
    {
      throw miog_error("geometry is not of correct floattype in set_multigeom_abc");
    }
    for (auto emat_x : {Mat::E::A, Mat::E::B, Mat::E::C})
    {
      max_n[emat_x] = std::max<size_t>(max_n[emat_x], get_mat_size(gg, toff, emat_x));
    }
  }

  size_t n_elmnts_limit = static_cast<size_t>(16e9 / sizeof(TFloat));
  for (auto emat_x : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    if (max_n[emat_x] > n_elmnts_limit)
    {
      std::stringstream ss;
      ss << "currently, this code only generates random matrices with fewer than " << n_elmnts_limit
         << " elements. The request here is for " << Mat::M().name[emat_x] << " to have "
         << max_n[emat_x] << "elements. ";
      throw miog_error(ss.str());
    }
  }
  // fill matrices with random floats.
  // Sometimes it seems to be important
  // to fill them with random floats,
  // as if they're integers, the kernel
  // can sometimes  run faster.

  for (auto emat_x : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    v_abc[emat_x]->resize(max_n[emat_x]);
    auto emem_x = Mem::mat_to_mem(emat_x);
    fill_uni<TFloat>(*v_abc[emat_x], max_n[emat_x] - toff.tails[emem_x], max_n[emat_x]);
  }
}

template <typename TFloat>
void set_abc(const MatData<TFloat>& v_abc, const Geometry& gg, const Offsets& toff)
{
  set_multigeom_abc(v_abc, {gg}, toff);
}

template <typename TFloat>
void set_abcw(const MatData<TFloat>& v_abcw, const Geometry& gg, const Offsets& toff)
{

  if (v_abcw.size() != Mem::E::N)
  {
    throw miog_error("vector should contain Mat::E::N (4) pointers in set_abcw");
  }

  MatData<TFloat> v_abc = v_abcw;
  v_abc.pop_back();
  set_abc<TFloat>(v_abc, gg, toff);

  size_t total_workspace = get_total_workspace(gg, toff);

  v_abcw[Mem::E::W]->resize(total_workspace);
  fill_uni(*(v_abcw[Mem::E::W]), total_workspace, total_workspace);
}

template void set_abc(const MatData<double>& v_abc, const Geometry& gg, const Offsets& toff);

template void set_abc(const MatData<float>& v_abc, const Geometry& gg, const Offsets& toff);

template void
set_multigeom_abc(const MatData<double>& v_abc, const std::vector<Geometry>&, const Offsets& toff);

template void
set_multigeom_abc(const MatData<float>& v_abc, const std::vector<Geometry>&, const Offsets& toff);

template void set_abcw(const MatData<double>& v_abcw, const Geometry& gg, const Offsets& toff);

template void set_abcw(const MatData<float>& v_abcw, const Geometry& gg, const Offsets& toff);
}
}
