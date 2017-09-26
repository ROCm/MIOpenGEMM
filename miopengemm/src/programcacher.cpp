/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <mutex>
#include <miopengemm/bundle.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/hyperparams.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/programcacher.hpp>
#include <miopengemm/programs.hpp>
#include <miopengemm/timer.hpp>
#include <miopengemm/tinyzero.hpp>

namespace MIOpenGEMM
{

// void ProgramCacher::free(size_t ID)
//{
// std::lock_guard<std::mutex> lock(mutt);
// if (ID >= max_cache_size - 1) // program_cache.size())
//{
// std::stringstream errm;
// errm << "Attempt to free non-existing ID (max allowed ID is " << max_cache_size - 1 << ").";
// throw miog_error(errm.str());
//}
// program_cache[ID].programs = {};
//// not deleting hyper_params;

// std::string geom_key("");
// for (auto& n : IDs)
//{
// if (std::get<1>(n) == ID)
//{
// geom_key = std::get<0>(n);
// break;
//}
//}
// if (geom_key == "")
//{
// std::stringstream ss;
// ss << "ID in free does not correspond to any cached geometry.";
// ss << " Being pedantic and throwing an error (although could continue (..))";
// throw miog_error(ss.str());
//}
// else
//{
// IDs.erase(geom_key);
//}

// available_IDs.push_back(ID);
//}

int ProgramCacher::get_ID_from_geom(const Geometry&   gg,
                                    BetaType          betatype,
                                    cl_command_queue* ptr_queue)
{
  return get_ID(gg.isColMajor,
                gg.tX[Mat::E::A],
                gg.tX[Mat::E::B],
                gg.tX[Mat::E::C],
                gg.m,
                gg.n,
                gg.k,
                gg.ldX[Mat::E::A],
                gg.ldX[Mat::E::B],
                gg.ldX[Mat::E::C],
                gg.wSpaceSize,
                betatype,
                gg.floattype,
                ptr_queue);
}

int ProgramCacher::get_ID(bool              isColMajor,
                          bool              tA,
                          bool              tB,
                          bool              tC,
                          size_t            m,
                          size_t            n,
                          size_t            k,
                          size_t            lda,
                          size_t            ldb,
                          size_t            ldc,
                          size_t            w_size,
                          BetaType          beta_type,
                          char              floattype,
                          cl_command_queue* ptr_queue)
{

  std::unique_lock<std::mutex> lock(mutt);

  int               ID = -1;
  std::stringstream ss;

  // get device id from ptr_queue.
  cl_device_id device_id;
  clGetCommandQueueInfo(*ptr_queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr);

  // Getting device name.
  size_t      info_size(0);
  std::string info_st(400, ' ');
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, info_st.size(), &info_st[0], &info_size);
  std::string device_name = info_st.substr(0, info_size - 1);

  ss << isColMajor << tA << tB << tC << '.' << m << '.' << n << '.' << k << '.' << lda << '.' << ldb
     << '.' << ldc << '.' << w_size << '.' << beta_type << '.' << floattype << '.' << device_name;

  auto key = ss.str();

  if (IDs.count(key) != 0)
  {
    ID = IDs[key];
  }

  else
  {

    owrite::Writer silent_mowri(Ver::E::SILENT, "");
    cl_context     context;
    oclutil::cl_set_command_queue_info(
      *ptr_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr, "GEMM", true);

    size_t      rank = 0;
    Constraints constraints("");
    Geometry    gg(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, w_size, floattype);

    oclutil::DevInfo devinfo(*ptr_queue);
    auto             soln =
      get_default_soln(devinfo, gg, constraints, silent_mowri, IfNoCache::E::GENERIC, rank);

    std::vector<KernBlob> v_blobs;

    for (auto& x : soln.v_tgks)
    {
      if (beta_type == BetaType::IsOne && x.e_ktype == KType::E::BETAC)
      {
        // don't run the beta kernel.
      }
      else
      {
        v_blobs.push_back(x);
      }
    }

    ID = current_ID;

    ++current_ID;
    if (current_ID >= max_cache_size)
    {
      std::stringstream errm;
      errm << "Number of programs exceeded limit of max_cache_size = " << max_cache_size << '.';
      throw miog_error(errm.str());
    }

    program_cache[ID] = Programs(device_id, context, silent_mowri);
    hyper_params[ID]  = soln.hypas;

    IDs[key] = ID;

    lock.unlock();
    program_cache[ID].update(v_blobs);
  }

  return ID;
}

ProgramCacher& get_cacher()
{
  static ProgramCacher cacher;
  return cacher;
}
}
