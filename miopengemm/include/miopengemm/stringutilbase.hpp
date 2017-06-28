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

#ifndef GUARD_MIOPENGEMM_STRINGUTILBASE_HPP
#define GUARD_MIOPENGEMM_STRINGUTILBASE_HPP

#include <string>
#include <vector>

namespace MIOpenGEMM
{
namespace stringutil
{

void indentify(std::string& source);

// split the string tosplit by delim.
// With x appearances of delim in tosplit,
// the returned vector will have length x + 1
// (even if appearances at the start, end, contiguous.
std::vector<std::string> split(const std::string& tosplit, const std::string& delim);

// split on whitespaces
std::vector<std::string> split(const std::string& tosplit);

std::string getdirfromfn(const std::string& fn);

// split something like QWE111 into QWE and 111.
std::tuple<std::string, unsigned> splitnumeric(std::string alphanum);

std::string get_padded(unsigned x, unsigned length = 4);
}
}

#endif
