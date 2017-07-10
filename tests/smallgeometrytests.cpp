/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/miogemm.hpp>

template <typename TFloat>
void geometrytest(bool isColMajor, bool tA, bool tB, bool tC, size_t m, size_t n, size_t k)
{

  size_t lda = (tA == isColMajor ? k : m) + 1;
  size_t ldb = (tB == isColMajor ? n : k) + 2;
  size_t ldc = (tC == isColMajor ? n : m) + 4;

  size_t a_offset = 5;
  size_t b_offset = 7;
  size_t c_offset = 11;

  size_t tail_off_a = 13;
  size_t tail_off_b = 17;
  size_t tail_off_c = 19;
  size_t tail_off_w = 34;  

  float                   allotted_time       = 5.001;
  size_t                allotted_iterations = 2;
  size_t                max_max_n_runs_per_kernel   = 1;
  double max_time_per_kernel = 1e9;
  MIOpenGEMM::SummStat::E sumstat             = MIOpenGEMM::SummStat::E::MEDIAN;

  // set verbose to true if you want output to terminal
  bool verbose = false;
  bool use_mowri_tracker = false;
  // set logfile if you want output forked to file
  std::string logfile("");

  std::string constraints_string = "A_WOS0__B_WOS0__C_ICE3";

  size_t n_postfind_runs = 1;
  bool     do_cpu_test     = true;

  size_t             workspace_size   = 3;
  size_t             workspace_offset = 4;
  char                 floattype        = sizeof(TFloat) == 4 ? 'f' : 'd';
  MIOpenGEMM::Geometry gg(
    isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype);
  MIOpenGEMM::Offsets offsets(
    a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c, tail_off_w);
  MIOpenGEMM::FindParams find_params(
    allotted_time, allotted_iterations, max_max_n_runs_per_kernel, max_time_per_kernel, sumstat);
  MIOpenGEMM::dev::basicfind(
    gg, offsets, find_params, verbose, logfile, constraints_string);
}


//Solution basicfind(const Geometry&   geometry,
                        //const Offsets&    toff,
                        //const FindParams& find_params,
                        //int verbose,
                        //std::string logfile,
                        //std::string constraints_string);



int main()
{
  size_t m     = 55;
  size_t k     = 118;
  size_t testi = 0;
  for (bool tC : {false, true})
  {
    for (bool isColMajor : {false, true})
    {
      for (bool tA : {false, true})
      {
        for (bool tB : {false, true})
        {
          for (size_t n : {m - 10, m + 10})
          {
            testi += 1;
            k += 1;
            std::cout << "\ntest " << testi << "/32";
            std::cout << "\nm=" << m << " n=" << n << " k=" << k << "\ntA=" << tA << " tB=" << tB
                      << " tC=" << tC << " isColMajor=" << isColMajor << std::endl;
            std::cout << "<float>  ";
            geometrytest<float>(isColMajor, tA, tB, tC, m, n, k);
            std::cout << "<double> ";
            geometrytest<double>(isColMajor, tA, tB, tC, m, n, k);
          }
        }
      }
    }
  }
  
  
  return 0;
}
