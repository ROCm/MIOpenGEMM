/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <sstream>
#include <string>
#include <miopengemm/enums.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/outputwriter.hpp>

namespace MIOpenGEMM
{
namespace owrite
{

void Writer::initialise_file()
{
  if (filename.compare("") == 0)
  {
    std::stringstream errm;
    errm << "empty filename passed to Writer, with to_file as true. This is not allowed";
    throw miog_error(errm.str());
  }

  file.open(filename, std::ios::out);

  if (file.good() == false)
  {
    std::stringstream errm;
    errm << "bad filename in constructor of Writer object. "
         << "The filename provided is `" << filename << "'."
         << "The directory of the file must exist, Writers do not create directories. "
         << "Either create all directories in the path, or change the provided path.  ";
    throw miog_error(errm.str());
  }
}

Writer::Writer(Ver::E v_, std::string filename_) : v(v_), filename(filename_)
{
  if (Ver::get_fileRequired()[v])
  {
    initialise_file();
  }
  else if (filename != "")
  {
    throw miog_error("Non-empty filename, but no file writing in Writer. Performing pedantic bail");
  }
  else
  {
    // no filename, and no filename required, good to continue
  }

  std::ofstream* ptr_file;
  for (size_t op = 0; op < OutPart::E::N; ++op)
  {
    ptr_file = Ver::get_toFile()[v][op] ? &file : nullptr;
    bw[op]   = BasicWriter(Ver::get_toTerm()[v][op], ptr_file);
  }
}

Writer::~Writer() { file.close(); }

template <>
BasicWriter& BasicWriter::operator<<(Flusher f)
{
  (void)f;

  if (to_terminal)
  {
    std::cout << std::flush;
  }

  if (ptr_file != nullptr)
  {
    ptr_file->flush();
  }
  return *this;
}

template <>
BasicWriter& BasicWriter::operator<<(Endline e)
{
  (void)e;
  if (to_terminal)
  {
    std::cout << std::endl;
  }

  if (ptr_file != nullptr)
  {
    (*ptr_file) << '\n';
    ptr_file->flush();
  }
  return *this;
}
}

// owrite::Endline Endl and owrite::Endline Flush are initialised in enums.cpp
// so as to keep all global variables in a single translation unit.
}
