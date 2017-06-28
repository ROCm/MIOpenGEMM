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

#include <iostream>
#include <string>
#include <miopengemm/error.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{
namespace outputwriting
{

OutputWriter::OutputWriter(bool to_terminal_, bool to_file_, std::string filename_)
  : to_terminal(to_terminal_), to_file(to_file_), filename(filename_)
{

  if (to_file == true)
  {
    if (filename.compare("") == 0)
    {
      throw miog_error("empty filename passed to OutputWrite, with to_file "
                       "flag true. This is not possible.");
    }

    file.open(filename, std::ios::out);

    if (file.good() == false)
    {
      std::string errm = "bad filename in constructor of OutputWriter object. "
                         "The filename provided is `";
      errm += filename;
      errm += "'. The directory of the file must exist, OutputWriters do not "
              "create directories. "
              "Either create all directories in the path, or change the "
              "provided path.  ";
      throw miog_error(errm);
    }
  }
}

OutputWriter::~OutputWriter() { file.close(); }

template <>
OutputWriter& OutputWriter::operator<<(Flusher f)
{
  f.increment();

  if (to_terminal)
  {
    std::cout << std::flush;
  }

  if (to_file)
  {
    file.flush();
  }
  return *this;
}

template <>
OutputWriter& OutputWriter::operator<<(Endline e)
{
  e.increment();
  if (to_terminal)
  {
    std::cout << std::endl;
  }

  if (to_file)
  {
    file << "\n";
    file.flush();
  }
  return *this;
}
}

outputwriting::Endline Endl;
outputwriting::Flusher Flush;
}
