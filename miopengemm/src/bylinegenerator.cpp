/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <iostream>
#include <sstream>
#include <string>
#include <miopengemm/bylinegenerator.hpp>
#include <miopengemm/error.hpp>
#include <miopengemm/prepgenerator.hpp>

/* TODO : interwoven hyper-parameter */
/* TODO : work-group size hyper-parameter (32 for vega) */

namespace MIOpenGEMM
{
namespace bylinegen
{

void ByLineGenerator::setup_final()
{

  setup_additional();

  if (emat_x >= Mat::E::N)
  {
    std::stringstream ss;
    ss << "in ByLineGenerator::setup, invalid emat_x : " << emat_x;
    ss << "\nMCHAR is " << MCHAR;
    ss << "\nmchar is " << mchar;
    throw miog_error(ss.str());
  }

  n_full_work_items_per_line = gg.get_coal(emat_x) / get_work_per_thread();
  n_work_items_per_line =
    n_full_work_items_per_line + (gg.get_coal(emat_x) % get_work_per_thread() != 0);
  n_full_work_items            = n_full_work_items_per_line * gg.get_uncoal(emat_x);
  n_work_items                 = n_work_items_per_line * gg.get_uncoal(emat_x);
  start_in_coal_last_work_item = get_work_per_thread() * n_full_work_items_per_line;
  work_for_last_item_in_coal   = gg.get_coal(emat_x) % get_work_per_thread();
}

ByLineGenerator::ByLineGenerator(Mat::E               emat_x_,
                                 const HyPas&         hp_,
                                 const Geometry&      gg_,
                                 const DerivedParams& dp_)
  : prepgen::PrepGenerator(emat_x_, hp_, gg_, dp_)
{
}

void ByLineGenerator::append_description_string(std::stringstream& ss) { ss << description_string; }

void ByLineGenerator::append_how_definitions(std::stringstream& ss)
{
  ss <<
    R"(/* The number of values from C which each non-edge work-item will scale by beta */
#define WORK_PER_THREAD  )"
     << get_work_per_thread() << R"(
/* The number of work items per work group */
#define N_WORK_ITEMS_PER_GROUP )"
     << get_local_work_size() << "\n\n";
}

void ByLineGenerator::append_derived_definitions(std::stringstream& ss)
{

  ss << "/*      each (full) work item will process WORK_PER_THREAD elements "
        "in the coalesced "
        "direction, */ \n";
  ss << "/*      so the number of work items per coalesced line is DIM_COAL / "
        "WORK_PER_THREAD */ \n";
  ss << "#define N_FULL_WORK_ITEMS_PER_LINE " << n_full_work_items_per_line << "\n";
  ss << "/*      including the possible final tail thread, */\n";
  ss << "/*      there are N_FULL_WORK_ITEMS_PER_LINE + (DIM_COAL % "
        "WORK_PER_THREAD != 0) */ \n";
  ss << "#define N_WORK_ITEMS_PER_LINE " << n_work_items_per_line << "\n";
  ss << "/*      in total there are N_FULL_WORK_ITEMS_PER_LINE * DIM_UNCOAL "
        "full work items, */ \n";

  ss << "#define N_FULL_WORK_ITEMS " << n_full_work_items << "\n";
  ss << "/*      and a grand total of N_WORK_ITEMS_PER_LINE * DIM_UNCOAL work "
        "items. */ \n";
  ss << "#define N_WORK_ITEMS " << n_work_items << "\n";
  ss << "/*      tail work items start at WORK_PER_THREAD * "
        "N_FULL_WORK_ITEMS_PER_LINE in the "
        "coalesced direction,  */\n";
  ss << "#define START_IN_COAL_LAST_WORK_ITEM " << start_in_coal_last_work_item << "\n";
  ss << "/*      and process DIM_COAL % WORK_PER_THREAD elements of c */\n";

  ss << "#define WORK_FOR_LAST_ITEM_IN_COAL " << work_for_last_item_in_coal << "\n";
  ss << "/*      the target stride between lines, derived from hp and gg (see "
        "DerivedParams) */\n";

  append_derived_definitions_additional(ss);
}

size_t ByLineGenerator::get_n_work_groups()
{
  size_t number_of_workgroups =
    (n_work_items / get_local_work_size()) + ((n_work_items % get_local_work_size()) != 0);
  return number_of_workgroups;
}

void ByLineGenerator::append_setup_coordinates(std::stringstream& ss)
{

  ss << "\n\n\n/* setting up where this thread works */";
  ss << "TINT" << MCHAR << " group_id = get_group_id(0);\n";
  ss << "TSHORT local_id = (TSHORT)(get_local_id(0));\n";
  ss << "TINT" << MCHAR << " global_id = group_id*N_WORK_ITEMS_PER_GROUP + local_id;\n";
  ss << "TINT" << MCHAR << " start_uncoal = 0;\n";
  ss << "TINT" << MCHAR << " start_coal = 0;\n";
  ss << "bool is_in_full_zone = (global_id < N_FULL_WORK_ITEMS);\n";

  if (n_full_work_items != 0)
  {
    ss << R"(
if (is_in_full_zone){   
start_uncoal = global_id / N_FULL_WORK_ITEMS_PER_LINE;
start_coal = WORK_PER_THREAD * (global_id % N_FULL_WORK_ITEMS_PER_LINE);
}

else if (global_id < N_WORK_ITEMS){
start_uncoal = (global_id - N_FULL_WORK_ITEMS)% DIM_UNCOAL;
start_coal = START_IN_COAL_LAST_WORK_ITEM;
}

)";
  }

  else
  {
    ss << "start_uncoal = (global_id)% DIM_UNCOAL;\n";
    ss << "start_coal = 0;";
  }
}

void ByLineGenerator::append_positioning_x_string(std::stringstream& ss)
{

  ss << "\n\n/* moving the " << mchar << " pointer to the first element to process */\n";
  ss << mchar << " += " << mchar << "_offset;\n";
  ss << mchar << " += start_uncoal * LD" << MCHAR << ";\n";
  ss << mchar << " += start_coal;\n";
}

void ByLineGenerator::append_inner_work(std::stringstream& ss) { ss << inner_work_string; }

void ByLineGenerator::append_work_string(std::stringstream& ss)
{

  ss <<
    R"(
if (is_in_full_zone){
#pragma unroll WORK_PER_THREAD
for (TSHORT i = 0; i < WORK_PER_THREAD; ++i){  )";
  append_inner_work(ss);
  ss << "\n}\n}\n";

  ss << R"(
else if (global_id < N_WORK_ITEMS){
for (TSHORT i = 0; i < WORK_FOR_LAST_ITEM_IN_COAL; ++i){  )";
  append_inner_work(ss);
  ss << "\n}\n}\n";
}

void ByLineGenerator::append_positioning_w_string(std::stringstream& ss)
{

  ss << R"(

/* moving the y pointer to the first element to process */
w += GLOBAL_OFFSET_W;
w += w_offset;
w += start_uncoal * LDW;
w += start_coal;
)";
}

KernBlob ByLineGenerator::get_kernelstring()
{

  std::stringstream ss;

  ss << get_time_string();
  append_description_string(ss);

  ss << "\n\n" << get_what_string() << "\n";
  append_basic_what_definitions(ss);

  ss << get_how_string() << "\n";
  append_how_definitions(ss);

  ss << get_derived_string() << "\n";
  append_derived_definitions(ss);

  ss << "#define TINT" << MCHAR << " " << dp.tints[emat_x] << "\n";
  ss << "#define TSHORT" << ' ' << dp.tshort << '\n';

  ss << "\n\n"
     << "__attribute__((reqd_work_group_size(N_WORK_ITEMS_PER_GROUP,1,1)))"
     << "\n";
  ss << "__kernel void ";

  ss << kernelname;
  append_fargs(ss);

  ss << "{";

  append_setup_coordinates(ss);
  append_positioning_x_string(ss);

  if (emat_x == Mat::E::A || emat_x == Mat::E::B)
  {
    append_positioning_w_string(ss);
  }

  append_work_string(ss);

  ss << "\n}\n\n\n";

  return {get_ktype(),
          {u_a, u_b, u_c, u_w, u_alpha, u_beta},
          ss.str(),
          kernelname,
          get_global_work_size(),
          get_local_work_size()};
}
}
}
