#ifndef TG_ITEREXPERIMENS_HPP
#define TG_ITEREXPERIMENS_HPP

#include <MIOpenGEMM/miogemm.hpp>

namespace MIOpenGEMM{
  
int run_find_experiments(const std::vector<Geometry> & geometries, std::vector<std::string> & v_constraints, const FindParams & find_params, bool verbose_inner, std::string basedir_inner, bool verbose_outer, std::string fn_outer);

std::vector<Geometry> get_deepbench_geometries(unsigned workspace_size = 1);

std::vector<Geometry> get_small_deepbench_geometries(unsigned small_threshold, unsigned workspace_size = 1);

std::vector<Geometry> get_large_deepbench_geometries(unsigned large_threshold, unsigned workspace_size = 1);

std::vector<Geometry> get_problem_geometries();

std::vector<Geometry> get_backconvwrw_geometries(unsigned workspace_size = 1);

std::vector<Geometry> get_small_growing_geometries(unsigned workspace_size = 1);

std::vector<Geometry> get_square_geometries(unsigned workspace_size = 1);

}

#endif
