/*******************************************************************************
 * 
 * MIT License
 * 
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 *******************************************************************************/

#include <tuple>
#include <miopengemm/geometryutil.hpp>

namespace MIOpenGEMM
{

std::vector<Geometry> get_from_m_n_k_tA_tB(
  const std::vector<std::tuple<unsigned, unsigned, unsigned, bool, bool>>& basicgeos,
  unsigned workspace_size)
{

  std::vector<Geometry> geometries;
  bool                  isColMajor = true;
  bool                  tA;
  bool                  tB;
  bool                  tC = false;
  unsigned              lda;
  unsigned              ldb;
  unsigned              ldc;
  unsigned              m;
  unsigned              n;
  unsigned              k;

  bool ldx_offset = false;

  for (auto& problem : basicgeos)
  {
    std::tie(m, n, k, tA, tB) = problem;
    lda = (tA == isColMajor ? k : m) + (ldx_offset == 1 ? 5 : 0);
    ldb = (tB == isColMajor ? n : k) + (ldx_offset == 1 ? 7 : 0);
    ldc = (tC == isColMajor ? n : m) + (ldx_offset == 1 ? 13 : 0);
    geometries.emplace_back(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, 'f');
  }
  return geometries;
}

std::vector<Geometry> get_from_m_n_k_ldaABC_tA_tB(
  const std::vector<
    std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>>& basicgeos,
  unsigned workspace_size)
{

  std::vector<Geometry> geometries;
  bool                  isColMajor = true;
  bool                  tA;
  bool                  tB;
  bool                  tC = false;
  unsigned              lda;
  unsigned              ldb;
  unsigned              ldc;
  unsigned              m;
  unsigned              n;
  unsigned              k;

  for (auto& problem : basicgeos)
  {
    std::tie(m, n, k, lda, ldb, ldc, tA, tB) = problem;
    geometries.emplace_back(isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, 'f');
  }
  return geometries;
}
}
