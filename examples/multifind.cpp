/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <string>
#include <miopengemm/geometries.hpp>
#include <miopengemm/tinytwo.hpp>

int main()
{

  srand(time(NULL));

  using namespace MIOpenGEMM;

  std::vector<Constraints> constraints_s{{""}};

  // std::vector<Geometry> geometries;

  // std::vector<Geometry> geometries0 = get_deepbench(0);
  // for (auto & gg : geometries0)
  //{
  // if (gg.m*gg.n*gg.k > 1000*1000*1000 && gg.m*gg.n*gg.k < 8000L*1000L*1000L){
  // geometries.push_back(gg);
  //}
  //}

  // auto m = geometries[i].m;
  // auto n = geometries[i].n;
  // auto k = geometries[i].k;
  // geometries[i].wSpaceSize = (m + 16)*(k + 16) + (n + 16)*(k + 16);
  //}

  std::vector<Geometry> geometries;

  auto&& kernel_cache = get_kernel_cache();
  auto   keys         = kernel_cache.get_keys();
  for (auto& key : keys)
  {
    // if (key.gg.wSpaceSize > 0){
    // auto gg = key.gg;
    // auto m = gg.m;
    // auto n = gg.n;
    // auto k = gg.k;
    // auto wSpaceSize = (m + 16)*(k + 16) + (n + 16)*(k + 16);
    // geometries.push_back({m,n,k, gg.tX[Mat::E::A], gg.tX[Mat::E::B], wSpaceSize, 'f'});
    //}

    if (kernel_cache.at(key).sus[Mat::E::C].vs[NonChi::MAC] == 1 &&
        key.gg.m * key.gg.n * key.gg.k > 100 * 100 * 100)
    {
      geometries.push_back(key.gg);
      std::cout << key.gg.get_string() << std::endl;
    }

    // if (key.gg.wSpaceSize == 0 && key.gg.m * key.gg.n * key.gg.k > 1) //250 * 250 * 250)
    //{
    // geometries.push_back(key.gg);
    //}
  }

  for (unsigned i = 0; i < geometries.size(); ++i)
  {
    std::swap(geometries[i], geometries[i + rand() % (geometries.size() - i)]);
  }

  std::cout << "n geometries : " << geometries.size() << std::endl;
  size_t                counter = 0;
  std::vector<Solution> solutions;
  for (auto& gg : geometries)
  {
    for (auto& cons : constraints_s)
    {
      ++counter;
      std::cout << '(' << counter << ')' << std::endl;
      std::string basedir = "/home/james/test49/";
      // WARNING : this call might only work on linux/mac
      std::string syscall =
        "./examples/multifindbase " + basedir + " " + gg.get_string() + " " + cons.get_string();
      // The reason we use a system call here is that if all done directly in loop,
      // compilation gets much slower. I don't know why.
      std::system(syscall.c_str());
    }
  }

  return 0;
}
