/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP
#define GUARD_MIOPENGEMM_DEGEMMAPIQQ_HPP

#include <stdlib.h>
#include <string>
#include <vector>
#include <miopengemm/geometry.hpp>
#include <miopengemm/solution.hpp>

namespace MIOpenGEMM
{
namespace dev
{

template <typename TFloat>
void benchgemm(const std::vector<std::string>& hyperstrings,
               size_t                        max_n_runs,
               double                         max_time,
               const Geometry&                 gg,
               const Offsets&                  toff,
               const TFloat*                   a,
               const TFloat*                   b,
               const TFloat*                   c,
               outputwriting::OutputWriter&    mowri);

template <typename TFloat>
void accuracy_test(const std::string&           hyperstring,
                   const Geometry&              gg,
                   const Offsets&               toff,
                   const TFloat*                a,
                   const TFloat*                b,
                   const TFloat*                c,
                   const TFloat*                c_true_for_test,
                   outputwriting::OutputWriter& mowri);

template <typename TFloat>
Solution find(const FindParams&            find_params,
              const TFloat*                a,
              const TFloat*                b,
              const TFloat*                c,
              std::string                  constraints_string,
              const Geometry&              gg,
              const Offsets&               toff,
              outputwriting::OutputWriter& mowri);


Solution basicfind(const Geometry&   geometry,
                        const Offsets&    toff,
                        const FindParams& find_params,
                        int verbose,
                        std::string logfile,
                        std::string constraints_string);


// TODO NOW : bring basicfind.hpp func here.

// first : kill the mowri tracker. 
//template <typename TFloat>
//Solution base_basicfind(const Geometry&   geometry,
                        //const Offsets&    toff,
                        //const FindParams& find_params,
                        //bool        verbose,
                        //std::string logfile,
                        //std::string constraints_string,
                        //size_t    n_postfind_runs, // remove. 
                        //bool        do_cpu_test, // remove.
                        //bool        use_mowri_tracker) //absorbed in OutputWriter
                        

}
}

#endif
