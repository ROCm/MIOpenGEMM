#include <string>
#include <sstream>

#include <MIOpenGEMM/normalformgenerator.hpp>
#include <MIOpenGEMM/error.hpp>

namespace MIOpenGEMM{
namespace nformgen{


class NormalFormGenerator : public prepgen::PrepGenerator{


public:
  NormalFormGenerator(const hyperparams::HyperParams & hp_,  const Geometry & gg_, const derivedparams::DerivedParams & dp_, std::string type_): prepgen::PrepGenerator(hp_, gg_, dp_, type_){}

  void setup() override final{
    

    if (type.compare("nforma") == 0){
      initialise_matrixtype('a');
    }
    
    else if (type.compare("nformb") == 0){
      initialise_matrixtype('b');
    }
  
    else{
      throw miog_error("Unrecognised type in normalformgenerator.cpp : " + type + ". should be either nforma or nformb \n");
    }
    
    set_usage_from_matrixchar();

  }
  
  size_t get_local_work_size() override final{
    return dp.at(emat_x).cw2_local_work_size;
  }

  size_t get_n_work_groups() override final{
    return dp.cw2_n_macro_tiles_pll_unroll * dp.at(emat_x).n_groups;
  }


 void append_copy_string(std::stringstream & ss){
   ss << "w[mu_pll_i*WRITE_STRIDE_PLL_K + mu_perp_i*WRITE_STRIDE_PERP_K] = " << matrixchar << "[mu_pll_i*READ_STRIDE_PLL_K + mu_perp_i*READ_STRIDE_PERP_K];";
 }



KernelString get_kernelstring(){
  std::stringstream ss;

  ss << "#define TFLOAT " << dp.t_float << "\n";
  ss << "#define " << "N_WORK_ITEMS_PER_GROUP " << dp.at(emat_x).cw2_local_work_size << "\n";
  ss << "#define UNROLL " << hp.at(nsHP::matC).vs[nsHP::UNR] << "\n";
  ss << "#define __K__ " << gg.k << "\n";
  
  append_unroll_block_geometry(matrixchar, ss, false, false);
  ss << "\n";
  append_stride_definitions(MATRIXCHAR, ss, 0, false, "READ_", false);
  ss << "\n";
  append_stride_definitions(MATRIXCHAR, ss, 2, false, "WRITE_", false);

  ss << "\n";
  
  ss << "#define LOAD_PLL_TO_UNROLL " << dp.at(emat_x).cw2_load_pll_to_unroll << "\n";

  ss << "\n/* MICRO_TILE_PLL_UNROLL * MICRO_TILE_PERP_UNROLL = N_ELEMENTS_TO_LOAD_PER_WORKITEM */\n";  
  ss << "#define MICRO_TILE_PLL_UNROLL " << dp.at(emat_x).cw2_micro_tile_pll_unroll << " \n";
  ss << "#define MICRO_TILE_PERP_UNROLL " << dp.at(emat_x).cw2_micro_tile_perp_unroll << "\n";
  
  ss << "#define N_MICRO_TILES_PLL_UNROLL " << dp.at(emat_x).cw2_n_micro_tiles_pll_unroll << "\n";
  ss << "#define N_MICRO_TILES_PERP_UNROLL " << dp.at(emat_x).cw2_n_micro_tiles_perp_unroll << "\n";

  ss << "#define N_ELEMENTS_PERP_UNROLL " << dp.at(emat_x).cw2_n_elements_perp_unroll << "\n";
  ss << "#define N_ELEMENTS_PER_WORK_ITEM " << dp.at(emat_x).cw2_n_elements_to_load_per_workitem << "\n";

  ss << "\n#define N_MACRO_TILES_PLL_UNROLL " << dp.cw2_n_macro_tiles_pll_unroll << "\n";
  
  ss << "\n#define GLOBAL_WORKSPACE_OFFSET " << dp.at(emat_x).cw_global_offset << "\n";
  
  ss << "\n#define PRESHIFT_FINAL_TILE " << dp.at(emat_x).preshift_final_tile << "\n";
  unsigned final_unroll_depth = gg.k%hp.at(nsHP::matC).vs[nsHP::UNR];
  final_unroll_depth = (final_unroll_depth == 0 ? hp.at(nsHP::matC).vs[nsHP::UNR] : final_unroll_depth);
  
  ss << "\n#define FINAL_UNROLL_DEPTH " << final_unroll_depth << "\n";
  
  ss << "\n\n" << "__attribute__((reqd_work_group_size(N_WORK_ITEMS_PER_GROUP,1,1)))" << "\n";
  ss << "__kernel void ";
  
  ss << kernelname;
  append_parameter_list_from_usage(ss);

  ss << "{";

  ss << "\n/* setting up where this thread works */\n";
  ss << "unsigned group_id = get_group_id(0);\n";
  ss << "unsigned micro_id = get_local_id(0);\n";

  ss << R"(

unsigned macro_id_pll_unroll = group_id % N_MACRO_TILES_PLL_UNROLL;
unsigned macro_id_perp_unroll = group_id / N_MACRO_TILES_PLL_UNROLL;

unsigned micro_id_pll_unroll = micro_id / N_MICRO_TILES_PERP_UNROLL;
unsigned micro_id_perp_unroll = micro_id % N_MICRO_TILES_PERP_UNROLL;

)";
  
  ss << matrixchar << " += macro_id_pll_unroll*READ_MACRO_STRIDE_PLL_K*UNROLL;\n";
  ss << matrixchar << " += macro_id_perp_unroll*READ_MACRO_STRIDE_PERP_K*MACRO_TILE_LENGTH;\n";
  ss << matrixchar << " += micro_id_pll_unroll*READ_STRIDE_PLL_K * MICRO_TILE_PLL_UNROLL;\n";
  ss << matrixchar << " += micro_id_perp_unroll*READ_STRIDE_PERP_K * MICRO_TILE_PERP_UNROLL;\n";

  ss << "\nif (macro_id_perp_unroll == N_GROUPS - 1){\n";
  
  ss << matrixchar << " -= READ_MACRO_STRIDE_PERP_K*(MACRO_TILE_LENGTH - PRESHIFT_FINAL_TILE)";
  ss << ";\n}\n";

  ss << matrixchar << " += " << matrixchar << "_offset;\n\n";
  
  ss << "w += GLOBAL_WORKSPACE_OFFSET;\n";
  ss << "w += macro_id_pll_unroll  *WRITE_MACRO_STRIDE_PLL_K   *UNROLL;\n";
  ss << "w += macro_id_perp_unroll *WRITE_MACRO_STRIDE_PERP_K  *MACRO_TILE_LENGTH;\n";
  ss << "w += micro_id_pll_unroll  *WRITE_STRIDE_PLL_K         *MICRO_TILE_PLL_UNROLL;\n";
  ss << "w += micro_id_perp_unroll *WRITE_STRIDE_PERP_K        *MICRO_TILE_PERP_UNROLL;\n";
  ss << "w += w_offset;\n";

  ss << R"(
if (macro_id_pll_unroll == N_MACRO_TILES_PLL_UNROLL - 1){
#pragma unroll
for (unsigned mu_pll_i = 0; mu_pll_i < MICRO_TILE_PLL_UNROLL; ++mu_pll_i) {
for (unsigned mu_perp_i = 0; mu_perp_i < MICRO_TILE_PERP_UNROLL; ++mu_perp_i) {
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
for (unsigned mu_pll_i = 0; mu_pll_i < MICRO_TILE_PLL_UNROLL; ++mu_pll_i) {
for (unsigned mu_perp_i = 0; mu_perp_i < MICRO_TILE_PERP_UNROLL; ++mu_perp_i) { 
)";
  append_copy_string(ss);
ss << R"(

}
}
}

)";

  ss << "\n}\n";
  
  return {{uses_a, uses_b, uses_c, uses_workspace, uses_alpha, uses_beta}, ss.str(), kernelname, get_global_work_size(), get_local_work_size()};
}


};




KernelString get_nforma_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp){
 NormalFormGenerator nfg(hp, gg, dp, "nforma");
 nfg.setup();
 return nfg.get_kernelstring();
}

KernelString get_nformb_kernelstring(const hyperparams::HyperParams & hp, const Geometry & gg, const derivedparams::DerivedParams & dp){
 NormalFormGenerator nfg(hp, gg, dp, "nformb");
 nfg.setup();
 return nfg.get_kernelstring();
}


}
}



