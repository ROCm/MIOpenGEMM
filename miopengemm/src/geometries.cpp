/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <algorithm>
#include <sstream>
#include <vector>

#include <miopengemm/geometries.hpp>
#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM
{

std::vector<Geometry> take_fives(size_t wSpaceSize)
{
  std::vector<Geometry> geometries;
  
  //std::vector<size_t> fives{1, 77, 363, 1002};
  std::vector<size_t> fives{3, 10001};
  
  for (size_t m : fives)
  {
    for (size_t n : fives)
    {
      for (size_t k : fives)
      {
        for (bool tA : {true, false}){
          for (bool tB : {true, false}){
            geometries.push_back({m,n,k,tA,tB, wSpaceSize, 'f'});
          }
        }
      }
    }
  }
  return geometries;
}

std::vector<Geometry> get_deepbench(size_t wSpaceSize)
{

  return {{512, 8, 500000, true, false, wSpaceSize, 'f'},
          {512, 8, 500000, false, false, wSpaceSize, 'f'},
          {512, 16, 512, false, true, wSpaceSize, 'f'},
          {512, 16, 512, false, false, wSpaceSize, 'f'},
          {512, 16, 500000, true, false, wSpaceSize, 'f'},
          {1024, 8, 500000, true, false, wSpaceSize, 'f'},
          {512, 16, 500000, false, false, wSpaceSize, 'f'},
          {1024, 8, 500000, false, false, wSpaceSize, 'f'},
          {512, 32, 512, false, true, wSpaceSize, 'f'},
          {512, 32, 512, false, false, wSpaceSize, 'f'},
          {1024, 16, 512, false, true, wSpaceSize, 'f'},
          {1024, 16, 512, false, false, wSpaceSize, 'f'},
          {1024, 16, 500000, true, false, wSpaceSize, 'f'},
          {1024, 16, 500000, false, false, wSpaceSize, 'f'},
          {1760, 16, 1760, true, false, wSpaceSize, 'f'},
          {1760, 16, 1760, false, false, wSpaceSize, 'f'},
          {1024, 32, 512, false, true, wSpaceSize, 'f'},
          {1024, 32, 512, false, false, wSpaceSize, 'f'},
          {2048, 16, 2048, true, false, wSpaceSize, 'f'},
          {2048, 16, 2048, false, false, wSpaceSize, 'f'},
          {2560, 16, 2560, true, false, wSpaceSize, 'f'},
          {2560, 16, 2560, false, false, wSpaceSize, 'f'},
          {3072, 16, 1024, true, false, wSpaceSize, 'f'},
          {3072, 16, 1024, false, false, wSpaceSize, 'f'},
          {1760, 32, 1760, true, false, wSpaceSize, 'f'},
          {1760, 32, 1760, false, false, wSpaceSize, 'f'},
          {4096, 16, 4096, true, false, wSpaceSize, 'f'},
          {2048, 32, 2048, true, false, wSpaceSize, 'f'},
          {4096, 16, 4096, false, false, wSpaceSize, 'f'},
          {2048, 32, 2048, false, false, wSpaceSize, 'f'},
          {4608, 16, 1536, true, false, wSpaceSize, 'f'},
          {4608, 16, 1536, false, false, wSpaceSize, 'f'},
          {2560, 32, 2560, true, false, wSpaceSize, 'f'},
          {2560, 32, 2560, false, false, wSpaceSize, 'f'},
          {6144, 16, 2048, true, false, wSpaceSize, 'f'},
          {6144, 16, 2048, false, false, wSpaceSize, 'f'},
          {3072, 32, 1024, true, false, wSpaceSize, 'f'},
          {3072, 32, 1024, false, false, wSpaceSize, 'f'},
          {1760, 64, 1760, true, false, wSpaceSize, 'f'},
          {1760, 64, 1760, false, false, wSpaceSize, 'f'},
          {7680, 16, 2560, true, false, wSpaceSize, 'f'},
          {7680, 16, 2560, false, false, wSpaceSize, 'f'},
          {4096, 32, 4096, true, false, wSpaceSize, 'f'},
          {2048, 64, 2048, true, false, wSpaceSize, 'f'},
          {4096, 32, 4096, false, false, wSpaceSize, 'f'},
          {2048, 64, 2048, false, false, wSpaceSize, 'f'},
          {8448, 16, 2816, true, false, wSpaceSize, 'f'},
          {8448, 16, 2816, false, false, wSpaceSize, 'f'},
          {4608, 32, 1536, true, false, wSpaceSize, 'f'},
          {4608, 32, 1536, false, false, wSpaceSize, 'f'},
          {2560, 64, 2560, true, false, wSpaceSize, 'f'},
          {2560, 64, 2560, false, false, wSpaceSize, 'f'},
          {6144, 32, 2048, true, false, wSpaceSize, 'f'},
          {6144, 32, 2048, false, false, wSpaceSize, 'f'},
          {3072, 64, 1024, true, false, wSpaceSize, 'f'},
          {3072, 64, 1024, false, false, wSpaceSize, 'f'},
          {1760, 128, 1760, true, false, wSpaceSize, 'f'},
          {1760, 128, 1760, false, false, wSpaceSize, 'f'},
          {7680, 32, 2560, true, false, wSpaceSize, 'f'},
          {7680, 32, 2560, false, false, wSpaceSize, 'f'},
          {4096, 64, 4096, true, false, wSpaceSize, 'f'},
          {2048, 128, 2048, true, false, wSpaceSize, 'f'},
          {4096, 64, 4096, false, false, wSpaceSize, 'f'},
          {2048, 128, 2048, false, false, wSpaceSize, 'f'},
          {8448, 32, 2816, true, false, wSpaceSize, 'f'},
          {8448, 32, 2816, false, false, wSpaceSize, 'f'},
          {35, 8457, 4096, true, false, wSpaceSize, 'f'},
          {35, 8457, 2560, true, false, wSpaceSize, 'f'},
          {35, 8457, 2048, true, false, wSpaceSize, 'f'},
          {35, 8457, 1760, true, false, wSpaceSize, 'f'},
          {35, 8457, 4096, false, false, wSpaceSize, 'f'},
          {35, 8457, 2560, false, false, wSpaceSize, 'f'},
          {35, 8457, 2048, false, false, wSpaceSize, 'f'},
          {35, 8457, 1760, false, false, wSpaceSize, 'f'},
          {2560, 128, 2560, true, false, wSpaceSize, 'f'},
          {2560, 128, 2560, false, false, wSpaceSize, 'f'},
          {3072, 128, 1024, true, false, wSpaceSize, 'f'},
          {3072, 128, 1024, false, false, wSpaceSize, 'f'},
          {7680, 64, 2560, true, false, wSpaceSize, 'f'},
          {7680, 64, 2560, false, false, wSpaceSize, 'f'},
          {4096, 128, 4096, true, false, wSpaceSize, 'f'},
          {4096, 128, 4096, false, false, wSpaceSize, 'f'},
          {1024, 700, 512, true, false, wSpaceSize, 'f'},
          {1024, 700, 512, false, false, wSpaceSize, 'f'},
          {7680, 128, 2560, true, false, wSpaceSize, 'f'},
          {7680, 128, 2560, false, false, wSpaceSize, 'f'},
          {512, 24000, 1530, true, false, wSpaceSize, 'f'},
          {512, 24000, 2560, true, false, wSpaceSize, 'f'},
          {512, 24000, 2048, true, false, wSpaceSize, 'f'},
          {512, 24000, 2816, true, false, wSpaceSize, 'f'},
          {512, 24000, 1530, false, false, wSpaceSize, 'f'},
          {512, 24000, 2560, false, false, wSpaceSize, 'f'},
          {512, 24000, 2048, false, false, wSpaceSize, 'f'},
          {512, 24000, 2816, false, false, wSpaceSize, 'f'},
          {1760, 7000, 1760, true, false, wSpaceSize, 'f'},
          {1760, 7000, 1760, false, false, wSpaceSize, 'f'},
          {1760, 7133, 1760, false, true, wSpaceSize, 'f'},
          {2048, 7000, 2048, true, false, wSpaceSize, 'f'},
          {2048, 7000, 2048, false, false, wSpaceSize, 'f'},
          {2048, 7133, 2048, false, true, wSpaceSize, 'f'},
          {2560, 7000, 2560, true, false, wSpaceSize, 'f'},
          {2560, 7000, 2560, false, false, wSpaceSize, 'f'},
          {2560, 7133, 2560, false, true, wSpaceSize, 'f'},
          {3072, 7435, 1024, false, true, wSpaceSize, 'f'},
          {512, 48000, 1530, true, false, wSpaceSize, 'f'},
          {512, 48000, 2560, true, false, wSpaceSize, 'f'},
          {512, 48000, 2048, true, false, wSpaceSize, 'f'},
          {512, 48000, 2816, true, false, wSpaceSize, 'f'},
          {512, 48000, 1530, false, false, wSpaceSize, 'f'},
          {512, 48000, 2560, false, false, wSpaceSize, 'f'},
          {512, 48000, 2048, false, false, wSpaceSize, 'f'},
          {512, 48000, 2816, false, false, wSpaceSize, 'f'},
          {1024, 24000, 1530, true, false, wSpaceSize, 'f'},
          {1024, 24000, 2560, true, false, wSpaceSize, 'f'},
          {1024, 24000, 2048, true, false, wSpaceSize, 'f'},
          {1024, 24000, 2816, true, false, wSpaceSize, 'f'},
          {1024, 24000, 1530, false, false, wSpaceSize, 'f'},
          {1024, 24000, 2560, false, false, wSpaceSize, 'f'},
          {1024, 24000, 2048, false, false, wSpaceSize, 'f'},
          {1024, 24000, 2816, false, false, wSpaceSize, 'f'},
          {4096, 7000, 4096, true, false, wSpaceSize, 'f'},
          {4096, 7000, 4096, false, false, wSpaceSize, 'f'},
          {4096, 7133, 4096, false, true, wSpaceSize, 'f'},
          {7680, 5481, 2560, false, true, wSpaceSize, 'f'},
          {5124, 9124, 4096, true, false, wSpaceSize, 'f'},
          {5124, 9124, 2560, true, false, wSpaceSize, 'f'},
          {5124, 9124, 2048, true, false, wSpaceSize, 'f'},
          {5124, 9124, 1760, true, false, wSpaceSize, 'f'},
          {5124, 9124, 4096, false, false, wSpaceSize, 'f'},
          {5124, 9124, 2560, false, false, wSpaceSize, 'f'},
          {5124, 9124, 2048, false, false, wSpaceSize, 'f'},
          {5124, 9124, 1760, false, false, wSpaceSize, 'f'},
          {1024, 48000, 1530, true, false, wSpaceSize, 'f'},
          {1024, 48000, 2560, true, false, wSpaceSize, 'f'},
          {1024, 48000, 2048, true, false, wSpaceSize, 'f'},
          {1024, 48000, 2816, true, false, wSpaceSize, 'f'},
          {1024, 48000, 1530, false, false, wSpaceSize, 'f'},
          {1024, 48000, 2560, false, false, wSpaceSize, 'f'},
          {1024, 48000, 2048, false, false, wSpaceSize, 'f'},
          {1024, 48000, 2816, false, false, wSpaceSize, 'f'},
          {3072, 24000, 1024, true, false, wSpaceSize, 'f'},
          {3072, 24000, 1024, false, false, wSpaceSize, 'f'},
          {4608, 24000, 1536, true, false, wSpaceSize, 'f'},
          {4608, 24000, 1536, false, false, wSpaceSize, 'f'},
          {3072, 48000, 1024, true, false, wSpaceSize, 'f'},
          {6144, 24000, 2048, true, false, wSpaceSize, 'f'},
          {3072, 48000, 1024, false, false, wSpaceSize, 'f'},
          {6144, 24000, 2048, false, false, wSpaceSize, 'f'},
          {7680, 24000, 2560, true, false, wSpaceSize, 'f'},
          {7680, 24000, 2560, false, false, wSpaceSize, 'f'},
          {8448, 24000, 2816, true, false, wSpaceSize, 'f'},
          {8448, 24000, 2816, false, false, wSpaceSize, 'f'},
          {4608, 48000, 1536, true, false, wSpaceSize, 'f'},
          {4608, 48000, 1536, false, false, wSpaceSize, 'f'},
          {6144, 48000, 2048, true, false, wSpaceSize, 'f'},
          {6144, 48000, 2048, false, false, wSpaceSize, 'f'},
          {7680, 48000, 2560, true, false, wSpaceSize, 'f'},
          {7680, 48000, 2560, false, false, wSpaceSize, 'f'},
          {8448, 48000, 2816, true, false, wSpaceSize, 'f'},
          {8448, 48000, 2816, false, false, wSpaceSize, 'f'}};
}

std::vector<Geometry> get_squares(size_t wSpaceSize)
{

  std::vector<Geometry> geometries;
  std::vector<size_t> dimensions;
  for (size_t k = 5; k < 14; ++k)
  {
    dimensions.push_back(std::pow(2, k));
    for (size_t off = 1; off < 5; ++off)
    {
      dimensions.push_back(std::pow(2, k) + off);
      dimensions.push_back(std::pow(2, k) - off);
    }
  }

  for (size_t k = 317; k < 7000; k += 300)
  {
    dimensions.push_back(k);
  }

  for (auto d : dimensions)
  {
    geometries.push_back({1, 0, 0, 0, d, d, d, d, d, d, wSpaceSize, 'f'});
    geometries.push_back({1, 0, 1, 0, d, d, d, d, d, d, wSpaceSize, 'f'});
    geometries.push_back({1, 1, 0, 0, d, d, d, d, d, d, wSpaceSize, 'f'});
    geometries.push_back({1, 1, 1, 0, d, d, d, d, d, d, wSpaceSize, 'f'});
  }
  return geometries;
}


}
