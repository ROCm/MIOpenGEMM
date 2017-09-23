/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

#include <mutex>
#include <miopengemm/bundle.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/geometry.hpp>
#include <miopengemm/miogemm.hpp>
#include <miopengemm/programs.hpp>
#include <miopengemm/timer.hpp>
#include <miopengemm/tinyzero.hpp>

namespace MIOpenGEMM
{

enum BetaType
{
  IsOne,
  IsOther
};

template <typename T>
BetaType get_beta_type(T beta)
{
  return (beta >= T(1) && beta <= T(1)) ? BetaType::IsOne : BetaType::IsOther;
  //(std::abs<T>(beta - T(1)) < std::numeric_limits<T>::epsilon
}

class ProgramCacher
{

  public:
  std::vector<Programs> program_cache;
  std::unordered_map<std::string, int> IDs;
  std::mutex mutt;

  void free(size_t ID)
  {
    std::lock_guard<std::mutex> lock(mutt);
    if (ID >= program_cache.size())
    {
      std::stringstream errm;
      errm << "Attempt to free non-existing ID (max ID is " << ID << ").";
      throw miog_error(errm.str());
    }
    program_cache[ID].programs = {};

    std::string geom_key("");
    for (auto& n : IDs)
    {
      if (std::get<1>(n) == ID)
      {
        geom_key = std::get<0>(n);
        break;
      }
    }
    if (geom_key == "")
    {
      std::stringstream ss;
      ss << "ID in free does not correspond to any cached geometry.";
      ss << " Being pedantic and throwing an error (although could continue (..))";
      throw miog_error(ss.str());
    }
    else
    {
      IDs.erase(geom_key);
    }
  }

  int get_ID(bool              isColMajor,
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

    ss << isColMajor << tA << tB << tC << '.' << m << '.' << n << '.' << k << '.' << lda << '.'
       << ldb << '.' << ldc << '.' << w_size << '.' << beta_type << '.' << floattype << '.'
       << device_name;

    auto                         key = ss.str();
    std::unique_lock<std::mutex> lock(mutt);

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
      program_cache.emplace_back(device_id, context, silent_mowri);
      ID       = program_cache.size() - 1;
      IDs[key] = ID;
      lock.unlock();
      program_cache.back().update(v_blobs);
    }

    return ID;
  }
};

inline ProgramCacher& get_cacher()
{
  static ProgramCacher cacher;
  return cacher;
}

// TODO : alpha = 0 optimisation. beta = 0 optimisation.
template <typename T>
GemmStatus xgemm(bool              isColMajor,
                 bool              tA,
                 bool              tB,
                 size_t            m,
                 size_t            n,
                 size_t            k,
                 T                 alpha,
                 cl_mem            a,
                 size_t            a_offset,
                 size_t            lda,
                 cl_mem            b,
                 size_t            b_offset,
                 size_t            ldb,
                 T                 beta,
                 cl_mem            c,
                 size_t            c_offset,
                 size_t            ldc,
                 cl_mem            w,
                 size_t            w_offset,
                 size_t            w_size,
                 cl_command_queue* ptr_queue,
                 cl_uint           num_events_in_wait_list,
                 const cl_event*   event_wait_list,
                 cl_event*         ptr_event_user,
                 int               ID)
{

  if (ID < 0)
  {

    BetaType beta_type = get_beta_type(beta);

    ID = get_cacher().get_ID(isColMajor,
                             tA,
                             tB,
                             false,  // tC not passed to xgemm.
                             m,
                             n,
                             k,
                             lda,
                             ldb,
                             ldc,
                             w_size,
                             beta_type,
                             get_floattype_char<T>(),
                             ptr_queue);
  }

  const Programs& programs = get_cacher().program_cache[ID];

  std::array<cl_mem, Mem::E::N> gpu_mems;
  std::array<size_t, Mem::E::N> offsets;

  gpu_mems[Mem::E::A] = a;
  gpu_mems[Mem::E::B] = b;
  gpu_mems[Mem::E::C] = c;
  gpu_mems[Mem::E::W] = w;

  offsets[Mem::E::A] = a_offset;
  offsets[Mem::E::B] = b_offset;
  offsets[Mem::E::C] = c_offset;
  offsets[Mem::E::W] = w_offset;

  AllKernArgs all_kern_args(0);
  for (auto& index : programs.act_inds)
  {
    auto& program = programs.programs[index];
    all_kern_args.emplace_back(
      kerngen::get_arg_sizes_values(program.kblob, gpu_mems, offsets, sizeof(T), &alpha, &beta));
  }

  KernelTimes* ktimes     = nullptr;
  bool         debug_mode = false;
  programs.run(*ptr_queue,
               all_kern_args,
               num_events_in_wait_list,
               event_wait_list,
               ktimes,  // update_times,
               ptr_event_user,
               debug_mode);

  return {true, ID};
}

template GemmStatus xgemm<float>(bool,
                                 bool,
                                 bool,
                                 size_t,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_command_queue*,
                                 cl_uint,
                                 const cl_event*,
                                 cl_event*,
                                 int ID);

template GemmStatus xgemm<double>(bool,
                                  bool,
                                  bool,
                                  size_t,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_command_queue*,
                                  cl_uint,
                                  const cl_event*,
                                  cl_event*,
                                  int ID);

// TODO : beta = 1 optimisation. alpha = 0 optimisation. beta = 0 optimisation.
template <typename T>
GemmStatus gemm0(bool              isColMajor,
                 bool              tA,
                 bool              tB,
                 size_t            m,
                 size_t            n,
                 size_t            k,
                 T                 alpha,
                 cl_mem            a,
                 size_t            a_offset,
                 size_t            lda,
                 cl_mem            b,
                 size_t            b_offset,
                 size_t            ldb,
                 T                 beta,
                 cl_mem            c,
                 size_t            c_offset,
                 size_t            ldc,
                 cl_command_queue* ptr_queue,
                 cl_uint           num_events_in_wait_list,
                 const cl_event*   event_wait_list,
                 cl_event*         ptr_event_user)
{
  return xgemm<T>(isColMajor,
                  tA,
                  tB,
                  m,
                  n,
                  k,
                  alpha,
                  a,
                  a_offset,
                  lda,
                  b,
                  b_offset,
                  ldb,
                  beta,
                  c,
                  c_offset,
                  ldc,
                  nullptr,
                  0,
                  0,
                  ptr_queue,
                  num_events_in_wait_list,
                  event_wait_list,
                  ptr_event_user,
                  -1);
}

template GemmStatus gemm0<float>(bool,
                                 bool,
                                 bool,
                                 size_t,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 float,
                                 cl_mem,
                                 size_t,
                                 size_t,
                                 cl_command_queue*,
                                 cl_uint,
                                 const cl_event*,
                                 cl_event*);

template GemmStatus gemm0<double>(bool,
                                  bool,
                                  bool,
                                  size_t,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  double,
                                  cl_mem,
                                  size_t,
                                  size_t,
                                  cl_command_queue*,
                                  cl_uint,
                                  const cl_event*,
                                  cl_event*);
}
