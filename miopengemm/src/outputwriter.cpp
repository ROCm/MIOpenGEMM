/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <iostream>
#include <string>
#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/outputwriter.hpp>
#include <miopengemm/enums.hpp>

namespace MIOpenGEMM
{
namespace outputwriting
{


void OutputWriter::set_v_bits(){

  main_to_file = false;
  main_to_terminal = false;
  tracker_to_file = false;
  tracker_to_terminal = false;

  if (v == Ver::E::SILENT){
  }

  else if (v == Ver::E::TERMINAL){
    main_to_terminal = true;
  }

  else if (v == Ver::E::SPLIT){
    main_to_file = true;
    main_to_terminal = true;
  }

  else if (v == Ver::E::TOFILE){
    main_to_file = true;
  }

  else if (v == Ver::E::TRACK){
    tracker_to_terminal = true;
  }
  

  else if (v == Ver::E::STRACK){
    main_to_file = true;
    tracker_to_terminal = true;
  }
  
  else{
    throw miog_error("unrecognised Ver enum in set_from_v");
  }   
}


void OutputWriter::initialise_file(){
  if (main_to_file || tracker_to_file)
  {
    if (filename.compare("") == 0)
    {
      std::stringstream errm;
      /* TODO : improve message getting string from enum verbse */
      errm << "empty filename passed to OutputWriter,"
       << "with to_file as true." 
       << "This is not allowed";
       throw miog_error(errm.str());
    }

    file.open(filename, std::ios::out);

    if (file.good() == false)
    {
      std::stringstream errm;
      errm  << "bad filename in constructor of OutputWriter object. "
            << "The filename provided is `";
      errm <<  filename;
      errm << "'. The directory of the file must exist, OutputWriters do not "
            <<  "create directories. "
             << "Either create all directories in the path, or change the "
             <<  "provided path.  ";
      throw miog_error(errm.str());
    }
  }
}


OutputWriter::OutputWriter(Ver::E v_, std::string filename_): v(v_), filename(filename_){

  set_v_bits();
  
  initialise_file();
  
  std::ofstream * main_ptr_file = main_to_file ? &file : nullptr;
  main = BasicWriter(main_to_terminal, main_ptr_file);

  std::ofstream * tracker_ptr_file = tracker_to_file ? &file : nullptr;
  tracker = BasicWriter(tracker_to_terminal, tracker_ptr_file);

  
}



OutputWriter::~OutputWriter() { file.close(); }

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
  (void)e;//.increment();
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

outputwriting::Endline Endl;
outputwriting::Flusher Flush;
}
