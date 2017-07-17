/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_ITEREXPERIMENS_HPP
#define GUARD_MIOPENGEMM_ITEREXPERIMENS_HPP

namespace MIOpenGEMM
{

int run_find_experiments(const std::vector<Geometry>& geometries,
                         const std::vector<Constraints>&    v_constraints,
                         const FindParams&            find_params,
                         bool                         verbose_inner,
                         std::string                  basedir_inner,
                         bool                         verbose_outer,
                         std::string                  fn_outer, 
                         const CLHint &     devhint);

std::vector<Geometry> get_deepbench_geometries(size_t wSpaceSize = 1);

std::vector<Geometry> get_old_deepbench_geometries(size_t wSpaceSize = 1);

std::vector<Geometry> get_new_deepbench_geometries(size_t wSpaceSize = 1);

std::vector<Geometry> get_small_deepbench_geometries(size_t small_threshold,
                                                     size_t wSpaceSize = 1);

std::vector<Geometry> get_large_deepbench_geometries(size_t large_threshold,
                                                     size_t wSpaceSize = 1);

std::vector<Geometry> get_problem_geometries();

std::vector<Geometry> get_backconvwrw_geometries(size_t wSpaceSize = 1);

std::vector<Geometry> get_small_growing_geometries(size_t wSpaceSize = 1);

std::vector<Geometry> get_square_geometries(size_t wSpaceSize = 1);
}

#endif
