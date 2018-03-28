/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <sstream>
#include <string>
#include <miopengemm/error.hpp>
#include <miopengemm/normalformgenerator.hpp>

namespace MIOpenGEMM
{
namespace nformgen
{

class NormalFormGenerator : public prepgen::PrepGenerator
{

  public:
  NormalFormGenerator(Mat::E               emat_x_,
                      const HyPas&         hp_,
                      const Geometry&      gg_,
                      const DerivedParams& dp_)
    : prepgen::PrepGenerator(emat_x_, hp_, gg_, dp_)
  {
  }

  virtual ~NormalFormGenerator() = default;

  virtual void set_type() override final { type = "nform" + std::string(1, mchar); }

  size_t get_local_work_size() override final { return dp.at(emat_x).cw2_local_work_size; }

  size_t get_n_work_groups() override final
  {
    return dp.cw2_n_macro_tiles_pll_unroll * dp.at(emat_x).n_groups;
  }

  void append_copy_string(std::stringstream& ss)
  {
    ss << "w[mu_pll_i*WRITE_STRIDE_PLL_K + mu_perp_i*WRITE_STRIDE_PERP_K] = " << mchar
       << "[mu_pll_i*READ_STRIDE_PLL_K + mu_perp_i*READ_STRIDE_PERP_K];";
  }

  KernBlob get_kernelstring() override final
  {
    std::stringstream ss;

    ss << "#define TFLOAT " << dp.t_float << '\n'
       << "#define TINT" << Mem::M().name[emat_x] << " " << dp.tints[emat_x] << '\n'
       << "#define N_WORK_ITEMS_PER_GROUP " << dp.at(emat_x).cw2_local_work_size << '\n'
       << "#define UNROLL " << hp.sus[Mat::E::C].vs[NonChi::E::UNR] << '\n'
       << "#define KV__ " << gg.k << '\n';

    append_unroll_block_geometry(emat_x, ss, false, false);
    ss << '\n';
    append_stride_definitions(emat_x, ss, 0, false, "READ_", false);
    ss << '\n';
    append_stride_definitions(emat_x, ss, 2, false, "WRITE_", false);

    ss << '\n'
       << "#define LOAD_PLL_TO_UNROLL " << dp.at(emat_x).cw2_load_pll_to_unroll << '\n'
       << "\n/* MICRO_TILE_PLL_UNROLL * MICRO_TILE_PERP_UNROLL = "
       << "N_ELEMENTS_TO_LOAD_PER_WORKITEM "
       << "*/\n"
       << "#define MICRO_TILE_PLL_UNROLL " << dp.at(emat_x).cw2_micro_tile_pll_unroll << " \n"
       << "#define MICRO_TILE_PERP_UNROLL " << dp.at(emat_x).cw2_micro_tile_perp_unroll << '\n'
       << "#define N_MICRO_TILES_PLL_UNROLL " << dp.at(emat_x).cw2_n_micro_tiles_pll_unroll << '\n'
       << "#define N_MICRO_TILES_PERP_UNROLL " << dp.at(emat_x).cw2_n_micro_tiles_perp_unroll
       << '\n'
       << "#define N_ELEMENTS_PERP_UNROLL " << dp.at(emat_x).cw2_n_elements_perp_unroll << '\n'
       << "#define N_ELEMENTS_PER_WORK_ITEM " << dp.at(emat_x).cw2_n_elements_to_load_per_workitem
       << '\n'
       << "\n#define N_MACRO_TILES_PLL_UNROLL " << dp.cw2_n_macro_tiles_pll_unroll << '\n'
       << "\n#define GLOBAL_WORKSPACE_OFFSET " << dp.at(emat_x).cw_global_offset << '\n'
       << "\n#define PRESHIFT_FINAL_TILE " << dp.at(emat_x).preshift_final_tile << '\n';

    size_t final_unroll_depth = gg.k % hp.sus[Mat::E::C].vs[NonChi::E::UNR];
    final_unroll_depth =
      (final_unroll_depth == 0 ? hp.sus[Mat::E::C].vs[NonChi::E::UNR] : final_unroll_depth);

    ss << "\n#define FINAL_UNROLL_DEPTH " << final_unroll_depth << "\n\n\n"
       << "__attribute__((reqd_work_group_size(N_WORK_ITEMS_PER_GROUP,1,1)))" << '\n'
       << "__kernel void " << kernelname;

    append_fargs(ss);

    ss << "{"
       << "\n/* setting up where this thread works */\n"
       << "TINT" << Mem::M().name[emat_x] << " group_id = get_group_id(0);\n"
       << "TINT" << Mem::M().name[emat_x] << " micro_id = (TINT" << Mem::M().name[emat_x]
       << ")(get_local_id(0));\n"
       << "\n"
       << "TINT" << Mem::M().name[emat_x]
       << " macro_id_pll_unroll = group_id % N_MACRO_TILES_PLL_UNROLL;\n"
       << "TINT" << Mem::M().name[emat_x]
       << " macro_id_perp_unroll = group_id / N_MACRO_TILES_PLL_UNROLL;\n"
       << "TINT" << Mem::M().name[emat_x]
       << " micro_id_pll_unroll = micro_id / N_MICRO_TILES_PERP_UNROLL;\n"
       << "TINT" << Mem::M().name[emat_x]
       << " micro_id_perp_unroll = micro_id % N_MICRO_TILES_PERP_UNROLL;\n";

    ss << mchar << " += macro_id_pll_unroll*READ_MACRO_STRIDE_PLL_K*UNROLL;\n"
       << mchar << " += "
       << "macro_id_perp_unroll*READ_MACRO_STRIDE_PERP_K*MACRO_"
       << "TILE_LENGTH;\n"
       << mchar << " += micro_id_pll_unroll*READ_STRIDE_PLL_K * "
       << "MICRO_TILE_PLL_UNROLL;\n"
       << mchar << " += micro_id_perp_unroll*READ_STRIDE_PERP_K * "
       << "MICRO_TILE_PERP_UNROLL;\n"
       << "\nif (macro_id_perp_unroll == N_GROUPS - 1){\n"
       << mchar << " -= READ_MACRO_STRIDE_PERP_K*(MACRO_TILE_LENGTH - "
       << "PRESHIFT_FINAL_TILE)"
       << ";\n}\n"
       << mchar << " += " << mchar << "_offset;\n\n"
       << "w += GLOBAL_WORKSPACE_OFFSET;\n"
       << "w += macro_id_pll_unroll  *WRITE_MACRO_STRIDE_PLL_K   *UNROLL;\n"
       << "w += macro_id_perp_unroll *WRITE_MACRO_STRIDE_PERP_K  "
       << "*MACRO_TILE_LENGTH;\n"
       << "w += micro_id_pll_unroll  *WRITE_STRIDE_PLL_K         "
       << "*MICRO_TILE_PLL_UNROLL;\n"
       << "w += micro_id_perp_unroll *WRITE_STRIDE_PERP_K        "
       << "*MICRO_TILE_PERP_UNROLL;\n"
       << "w += w_offset;\n";

    ss << R"(
if (macro_id_pll_unroll == N_MACRO_TILES_PLL_UNROLL - 1){
#pragma unroll
for (ushort mu_pll_i = 0; mu_pll_i < MICRO_TILE_PLL_UNROLL; ++mu_pll_i) {
for (ushort mu_perp_i = 0; mu_perp_i < MICRO_TILE_PERP_UNROLL; ++mu_perp_i) {
if (micro_id_pll_unroll * MICRO_TILE_PLL_UNROLL + mu_pll_i < FINAL_UNROLL_DEPTH) { 
)";
    append_copy_string(ss);
    ss << R"(
}
}
}
}


else{
#pragma unroll
for (ushort mu_pll_i = 0; mu_pll_i < MICRO_TILE_PLL_UNROLL; ++mu_pll_i) {
for (ushort mu_perp_i = 0; mu_perp_i < MICRO_TILE_PERP_UNROLL; ++mu_perp_i) { 
)";
    append_copy_string(ss);
    ss << R"(

}
}
}

)";

    ss << "\n}\n";

    return {get_ktype(),
            {u_a, u_b, u_c, u_w, u_alpha, u_beta},
            ss.str(),
            kernelname,
            get_global_work_size(),
            get_local_work_size()};
  }

  virtual void setup_final() override final {}

  virtual KType::E get_ktype() override final
  {
    switch (emat_x)
    {
    case Mat::E::A: return KType::E::WSA;
    case Mat::E::B: return KType::E::WSB;
    case Mat::E::C: throw miog_error("no option `C' in get_ktype in normalformgenerator");
    case Mat::E::N: throw miog_error("no option `C' in get_ktype in normalformgenerator");
    }
    throw miog_error("failed in get_ktype");
  }
};

KernBlob
get_nform_kernelstring(Mat::E emat_x, const HyPas& hp, const Geometry& gg, const DerivedParams& dp)
{
  NormalFormGenerator nfg(emat_x, hp, gg, dp);
  nfg.setup();
  return nfg.get_kernelstring();
}
}
}
