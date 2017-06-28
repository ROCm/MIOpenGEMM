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

#ifndef GUARD_MIOPENGEMM_OUTPUTWRITER_H
#define GUARD_MIOPENGEMM_OUTPUTWRITER_H

#include <fstream>
#include <iostream>
#include <string>

namespace MIOpenGEMM
{
namespace outputwriting
{

class Flusher
{
  public:
  void increment(){};
};

class Endline
{
  public:
  void increment(){};
};

class OutputWriter
{

  public:
  bool          to_terminal;
  bool          to_file;
  std::ofstream file;
  std::string   filename;

  OutputWriter();
  ~OutputWriter();
  OutputWriter(bool to_terminal, bool to_file, std::string filename = "");
  void operator()(std::string);

  template <typename T>
  OutputWriter& operator<<(T t)
  {

    if (to_terminal)
    {
      std::cout << t;
    }

    if (to_file)
    {
      file << t;
    }
    return *this;
  }
};

template <>
OutputWriter& OutputWriter::operator<<(Flusher f);

template <>
OutputWriter& OutputWriter::operator<<(Endline e);
}

extern outputwriting::Flusher Flush;
extern outputwriting::Endline Endl;
}

#endif
