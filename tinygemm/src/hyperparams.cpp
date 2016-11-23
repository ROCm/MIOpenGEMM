#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

#include "stringutilbase.hpp"
#include "stringutilbase.hpp"
#include "hyperparams.hpp"
#include "tinygemmerror.hpp"
#include "stringutilbase.hpp"



namespace tinygemm{
namespace hyperparams{


/* take in hyper-parameter string, and some default hyper parameters, and combine them*/
HyperParams get_hyperparam(const HyperParams & default_hp, std::string hyperstring){
  auto frags = stringutil::split(hyperstring, "_");
  std::map<std::string, unsigned> stripped;
  std::string key;
  unsigned val;
  for (auto & x : frags){
    std::tie(key, val) = stringutil::splitnumeric(x);
    stripped[key] = val;
  }
  HyperParams hp(default_hp.params);
  hp.params["micro_tile_height"] = stripped.at("y");
  hp.params["micro_tile_width"] = stripped.at("x");
  hp.params["macro_tile_height"] = stripped.at("Y");
  hp.params["macro_tile_width"] = stripped.at("X");
  hp.params["group_allocation"] = stripped.at("GA");
  hp.params["work_item_load_a_pll_to_unroll"] = stripped.at("APLU");
  hp.params["work_item_load_b_pll_to_unroll"] = stripped.at("BPLU");
  hp.params["unroll_pragma"] = stripped.at("PU");
  hp.params["load_to_lds_interwoven"] = stripped.at("LIW");
  hp.params["c_micro_tiles_interwoven"] = stripped.at("MIW");
  hp.params["unroll"] = stripped.at("U");
  hp.params["n_work_items_per_c_elm"] = stripped.at("ICE");
  hp.params["pad"] = stripped.at("P");
  hp.params["unroll_for_offset"] = stripped.at("UFO");  
  return hp;
}





HyperParams get_default_small(bool enforce_deterministic){
  std::map<std::string, unsigned> params;
  params["micro_tile_width"] = 2;  
  params["micro_tile_height"] = 2;
  params["macro_tile_width"] = 16;
  params["macro_tile_height"] = 16; 
  params["unroll"] = 16;
  params["pad"] = 1;    
  params["group_allocation"] = 1;
  params["work_item_load_a_pll_to_unroll"] = 0;
  params["work_item_load_b_pll_to_unroll"] = 1;
  params["unroll_pragma"] = 1;
  params["load_to_lds_interwoven"] = 0;
  params["c_micro_tiles_interwoven"] = 1;
  params["n_work_items_per_c_elm"] = (enforce_deterministic == false) ? 3 : 1;
  params["n_target_active_workgroups"] = 64;
  params["unroll_for_offset"] = 0;
  return HyperParams(params);
}


/* Find the nearest geometry in the cache, and take its hyper params */
HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, bool enforce_deterministic){
  HyperParams default_hp = get_default_small(enforce_deterministic);

  /* The case  of  (gg.m < 16 || gg.n < 16) */  
  if (gg.m < 16 || gg.n < 16) {
    throw tinygemm_error("Currently, we do not support matrices which are skinnier (m or n) than 16. This can easily be fixed... please contanct me at jnewling@idiap.ch ");
  }
  
  tinygemm::TinyGemmGeometry nearestgeometry;
  HyperParams  best_hp;
  float start_min_distance = std::numeric_limits<float>::max();
  float min_distance = start_min_distance;
  
  tinygemm::TinyGemmGeometry geo;
  std::string hpstring;
  HyperParams hp;
  
  for (auto geohyp : HyperParams::kernel_cache){
    std::tie(geo, hpstring) = geohyp; 
    hp = get_hyperparam(default_hp, hpstring);
    float new_distance = gg.get_distance(geo);
    if (new_distance < min_distance){
      std::cout << "distance : " << new_distance << std::endl;
      nearestgeometry = geo;
      std::cout << nearestgeometry.get_string() << std::endl;
      best_hp = hp;
      min_distance = new_distance;
    }
  }
  
  if (enforce_deterministic == true){
    best_hp.params["n_work_imtes_per_c_elm"] = 1;
  }
  
  /* No near matches, this means that there are no tiles which are smaller in both dimensions */
  if (min_distance == start_min_distance){
    best_hp = default_hp;
  }
  
  return best_hp;
}




HyperParams::HyperParams(std::map<std::string, unsigned> params):params(params){
  do_checks();
}
  
  
void HyperParams::do_checks(){
  for (auto & x : all_hyper_param_names){
    if (params.count(x) == 0){
      std::string errm("The parameter `");
      errm += x;
      errm += "', which should appear as a hyper-parameter but appears not to, should be included\n";
      throw tinygemm_error(errm);
    }
  }
  
  for (auto & x : params){
    auto blip = std::find(all_hyper_param_names.cbegin(), all_hyper_param_names.cend(), x.first);
    if (blip == all_hyper_param_names.cend()) {
      std::string errm("The parameter `");
      errm += x.first;
      errm += "', which appears in the user-defined list of hyper-parameter, is not recognised\n";
      throw tinygemm_error(errm);        
    }
  }
}

unsigned HyperParams::get_workgroup_size(){
  return (params.at("macro_tile_height")*params.at("macro_tile_width")) / (params.at("micro_tile_height")*params.at("micro_tile_width"));
}

unsigned HyperParams::get_nwitems_h(){
  return params.at("macro_tile_height")/params.at("micro_tile_height");
}

unsigned HyperParams::get_nwitems_w(){
  return params.at("macro_tile_width")/params.at("micro_tile_width");
}

bool HyperParams::operator == (const HyperParams & hpr){
  for (auto & x : params){
    if (x.second != hpr.params.at(x.first)){
      return false;
    }
  }
  return true;
}
  
    
void add_hyperparam(const HyperParams & default_hp, std::string hyperstring, std::vector<HyperParams> & one_aways){
  one_aways.push_back(get_hyperparam(default_hp, hyperstring));
}
    
  
std::vector<HyperParams> HyperParams::get_one_aways(const tinygemm::TinyGemmGeometry & gg){
  
  if (gg.m < 16 || gg.n < 16){
    throw tinygemm_error("Currently, if matrix C has a dimension less that 16, it is not supported. If you are seeing this, please remind to fix it..  jnewling@amd.com / jnewling@idiap.ch ");
  }
 
  std::vector<HyperParams> one_aways;
  size_t n_h0 = get_nwitems_h();

  size_t n_w0 = get_nwitems_w();

  
  /* *****************  micro tile sizes ***************************** 
   * We form a superset of the micro-tile to micro-tile one-step edges
   * as the cartesian product, micro_tile_step ^ 2. 
   * to view the micro_tile_step edge "graph", take a look at one_away_generator.py 
   * */
  std::map <unsigned, std::vector<unsigned> > micro_tile_step;
  micro_tile_step[1] = {1,2};
  micro_tile_step[2] = {1,2,3,4};
  micro_tile_step[3] = {2,3,4};
  micro_tile_step[4] = {2,3,4,5,6};
  micro_tile_step[5] = {4,5,6,8};
  micro_tile_step[6] = {4,5,6,8};
  micro_tile_step[8] = {6,8};

  /* we now form the one-step micro tile edges by pruning the product */
  std::map< std::array<unsigned, 2>, std::vector< std::array<unsigned, 2> > > micro_tile_edges;
  std::vector<unsigned> CC {1,2,3,4,5,6,8};
  for (auto & x: CC){
    for (auto & y : CC){
      for (auto & nx : micro_tile_step[x]){
        for (auto & ny : micro_tile_step[y]){
          /* eliminate type 1 skinny micro-tiles */
          bool not_too_skinny = (std::abs(int(nx) - int(ny)) <= 4);
          
          float delta_ratio = (float(x)/float(y)) / (float(nx)/float(ny));
          
          /* eliminate too dramatic changes is skinniness */
          bool skininess_change_good = (delta_ratio < 2.01 && delta_ratio > 0.499);
          float delta_volume = (float(x)*float(y)) / (float(nx)*float(ny));
          
          /* eliminate too dramatic changes in volume unless going to an `even hub' */
          bool volumn_change_good = ((nx%2 == 0 and ny%2 == 0) || (delta_volume <= 2.01 and delta_volume > 0.499));
          /* the only way to get to 5,8 is from 4,8 */
          bool condition_on_58 = ((x == 4 && y == 8) || (x == 8 && y == 4) || (!(nx == 5 && ny == 8) && !(nx == 8 && ny == 5)));
          if (not_too_skinny and skininess_change_good and volumn_change_good and condition_on_58){
            std::array<unsigned, 2> key  {{x,y}};
            std::array<unsigned, 2> value {{nx, ny}};
            micro_tile_edges[ key ].push_back( value );
          }
        }
      }
    }
  }
  
  for (auto & micro_tile : micro_tile_edges[ {{params.at("micro_tile_height") , params.at("micro_tile_width") }} ]){
    
    auto micro_h = micro_tile[0];
    auto micro_w = micro_tile[1];
    
    /* To each of the one-step micro tile edges, we can also (with p = 0.333) 
     * change n_work_items_per_c_elm in the same step, provided that n_work_items_per_c_elm 
     * increases by 1 if the area of the micro tile decreases, and v.v. */
    std::vector<unsigned> k_splits_to_consider;
    if (micro_h*micro_w < params["micro_tile_height"]*params["micro_tile_width"] && params["micro_tile_height"]*params["micro_tile_width"] < 36){
      if (rand()%3 == 0){ //TODO : is using rand bad?
        k_splits_to_consider = {params["n_work_items_per_c_elm"], params["n_work_items_per_c_elm"] + 1};
      }
      else{
        k_splits_to_consider = {params["n_work_items_per_c_elm"]};
      }
    }
    
    else if (micro_h*micro_w > params["micro_tile_height"]*params["micro_tile_width"] && params["n_work_items_per_c_elm"] > 1 ){
      if (rand()%3 == 0){
        k_splits_to_consider = {params["n_work_items_per_c_elm"], params["n_work_items_per_c_elm"] - 1};
      }
      else{
        k_splits_to_consider = {params["n_work_items_per_c_elm"]};
      }
    }
    
    else{
      k_splits_to_consider = {params["n_work_items_per_c_elm"]};
    }
    
    for (auto k_split : k_splits_to_consider){
      HyperParams hp(params);
      hp.params["micro_tile_height"] = micro_h;
      hp.params["micro_tile_width"] = micro_w;
      hp.params["macro_tile_height"] = micro_h*n_h0;
      hp.params["macro_tile_width"] = micro_w*n_w0;
      hp.params["n_work_items_per_c_elm"] = k_split;
      
      /* Observations suggest that k-split > 1 does not work very well with ufo. */
      if (k_split > 1){
        hp.params["unroll_for_offset"] = 0;
      }
       
      one_aways.push_back(hp);
    }
  }


  /* ***************** n_work_items_per_c_elms ************************
   * These hold the tile size constant, and explore just n_work_items_per_c_elm
   */
   
  std::map<unsigned, std::vector<unsigned> > map_n_work_items_per_c_elms_1_away;
  map_n_work_items_per_c_elms_1_away[1] = {2};
  map_n_work_items_per_c_elms_1_away[2] = {1,3,4};
  map_n_work_items_per_c_elms_1_away[3] = {2,4,6};
  map_n_work_items_per_c_elms_1_away[4] = {2,3,5,6};
  map_n_work_items_per_c_elms_1_away[5] = {4,6,7,8};
  map_n_work_items_per_c_elms_1_away[6] = {4,5,7,8};
  map_n_work_items_per_c_elms_1_away[7] = {6,8};
  map_n_work_items_per_c_elms_1_away[8] = {5,7,9,10};
  map_n_work_items_per_c_elms_1_away[9] = {8,10};
  map_n_work_items_per_c_elms_1_away[10] = {8,14};
  map_n_work_items_per_c_elms_1_away[14] = {8,10};
  
  for (auto & n_work_items_per_c_elm : map_n_work_items_per_c_elms_1_away[params.at("n_work_items_per_c_elm")]){
    HyperParams hp(params);
    hp.params["n_work_items_per_c_elm"] = n_work_items_per_c_elm;
    /* Observations suggest that k-split > 1 does not work very well with ufo. */
    if (n_work_items_per_c_elm > 1){
      hp.params["unroll_for_offset"] = 0;
    }
      
    one_aways.push_back(hp);
  }

  /* ***************** macro tile sizes ********************************/

  /* For Nvidia, where a wavefront (warp) is 32, should this be different? TODO */
  
  /* The standard 8x8 and 16x16 tiling schemes.*/
  std::vector<unsigned> wg_hw_s = {8, 16};
  for (auto & wg_hw : wg_hw_s){
    HyperParams hp(params);
    hp.params["macro_tile_height"] = wg_hw*hp.params.at("micro_tile_height");
    hp.params["macro_tile_width"] = wg_hw*hp.params.at("micro_tile_width");          
    one_aways.push_back(hp);
  }
  
  /* Currently, if C has m or n less than 16, an error is thrown. */
  
  /* **************** unrolls *****************************************
   * */
  std::vector<unsigned> unrolls = {8, 16};
  std::map<unsigned, std::vector<unsigned>> unroll_map;
  unroll_map[8]  = {16};
  unroll_map[16] = {8,32};
  unroll_map[32] = {16, 48};
  unroll_map[48] = {32, 64};
  unroll_map[64] = {48};
  
  for (auto unroll : unroll_map.at(params.at("unroll"))){
    HyperParams hp(params);
    hp.params["unroll"] = unroll;
    /* (weak) observations suggest that unroll > 8 does not work well with ufo. */
    if (unroll > 8){
      hp.params["unroll_for_offset"] = 0;
    }
    one_aways.push_back(hp);
  }
  
  if (params.at("n_work_items_per_c_elm") >= 4){ //if n_work_items_per_c_elm is 4,5,6,7,8,9,10 ... consider making it 2,2,2,2,4,4,4,4 ... with an "increase" in unroll_map
    HyperParams hp_2(params);
    hp_2.params["unroll"] = unroll_map.at(params.at("unroll")).back();
    hp_2.params["n_work_items_per_c_elm"] = 2*(params.at("n_work_items_per_c_elm")/4);
    hp_2.params["unroll_for_offset"] = 0; /* this is a mere whim */
    one_aways.push_back(hp_2);
  }  
  
  /* **************** pads *****************************************
   * considering any pad other than 1 is  probably a waste of time, 
   * as I've never seen any pad significantly outperform 1. 
   * */
  std::vector<unsigned> pads = {1}; //2
  for (auto & pad : pads){
    HyperParams hp(params);
    hp.params["pad"] = pad;
    one_aways.push_back(hp);
  }

  /* **************** group allocation *****************************
   * I have seen all of 2 (row-wise) , 1 (column-wise) and
   * 3 (column-wise within row-wise), performing strictly than the
   * other 2
   * */
  std::vector<unsigned> group_allocations = {1,2,3};
  for (auto & group_allocation : group_allocations){
    HyperParams hp(params);
    hp.params["group_allocation"] = group_allocation;
    one_aways.push_back(hp);
  }

  /* ************** work_item_load_a_pll_to_unrolls ****************
   * TODO : reference to a base explanantion of meta params
  */
  std::vector<unsigned> work_item_load_a_pll_to_unrolls = {0,1};
  for (auto & work_item_load_a_pll_to_unroll : work_item_load_a_pll_to_unrolls){
    HyperParams hp(params);
    hp.params["work_item_load_a_pll_to_unroll"] = work_item_load_a_pll_to_unroll;
    one_aways.push_back(hp);
  }

  /* ************** work_item_load_b_pll_to_unrolls ****************
  */
  std::vector<unsigned> work_item_load_b_pll_to_unrolls = {0,1};    
  for (auto & work_item_load_b_pll_to_unroll : work_item_load_b_pll_to_unrolls){
    HyperParams hp(params);  
    hp.params["work_item_load_b_pll_to_unroll"] = work_item_load_b_pll_to_unroll;
    one_aways.push_back(hp);
  }
  
  /* ************** unroll_pragmas **************************************
   * probably not important. I have seen to be important in combo with UFO
   * I do recollect seeing this make a difference at some point in the past
   * Moreover, who knows what the next generation of compilers will do
  */
  std::vector<unsigned> unroll_pragmas = {0,1};
  for (auto & unroll_pragma : unroll_pragmas){
    HyperParams hp(params);
    hp.params["unroll_pragma"] = unroll_pragma;
    one_aways.push_back(hp);
  }
  
  /* *******************load_to_lds_interwovens **************************
   * definitely an important parameter. 
   * So far, I've seen 0 beating 1
   * */    
  std::vector<unsigned> load_to_lds_interwovens = {0,1};
  for (auto & load_to_lds_interwoven : load_to_lds_interwovens){
    HyperParams hp(params);
    hp.params["load_to_lds_interwoven"] = load_to_lds_interwoven;
    one_aways.push_back(hp);
  }
  
  /* ******************* c_micro_tiles_interwovens ************************
   * An important parameter. 
   * So far, I've seen 1 (ala cobalt) beating 0
   * */
  std::vector<unsigned> c_micro_tiles_interwovens = {0,1};
  for (auto & c_micro_tiles_interwoven : c_micro_tiles_interwovens){
    HyperParams hp(params);
    hp.params["c_micro_tiles_interwoven"] = c_micro_tiles_interwoven;
    one_aways.push_back(hp);
  }
  
  /* ******************* unroll_for_offset ******************************************
   * This can improve performance by a significant amount.
   * Strangely, I have only seen it helping when combined with unroll_pragma true. 
   * My hypothesis for the above is that the compiler is relucant to unroll loops
   * with the additional complexity added by unroll_for_offset = 1.
   * */
  std::vector<unsigned> unroll_for_offsets = {0,1};
  for (auto & unroll_for_offset : unroll_for_offsets){
    HyperParams hp2(params);
    hp2.params["unroll_for_offset"] = unroll_for_offset;
    hp2.params["unroll_pragma"] = true;
    one_aways.push_back(hp2);
  }
  
  
  bool add_custom_edges = true;
  if (add_custom_edges == true){
    
    /* ************************ custom edges ***************************************************
     * This is the place to add some edges experience shows can tunnel out of local minima, or
     * lead to some kernels which have found to be good on some problems  */
    
    auto add_hps = [this, & one_aways](std::string hparamstring){
      add_hyperparam(*this, hparamstring, one_aways);
    };
    
    if (params.at("micro_tile_height")*params.at("micro_tile_width") <=4 ){
      add_hps("Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE6_UFO0");
    }

    if (params.at("micro_tile_height")*params.at("micro_tile_width") <=16 ){
      add_hps("Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ET1_ICE5_UFO0");
    }

    if (params.at("micro_tile_height")*params.at("micro_tile_width") <=20 ){
      add_hps("Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE4_UFO0");
    }
    
    if (params.at("micro_tile_height")*params.at("micro_tile_width") >=16 ){
      add_hps("Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0");
    }
    
    if (params.at("micro_tile_height")*params.at("micro_tile_width") >=8 ){
      add_hps("Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0");
    }

    if (params.at("micro_tile_height") >= params.at("micro_tile_width") &&  params.at("micro_tile_height")*params.at("micro_tile_width") >=10){
      add_hps("Y96_X64_y6_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0");
    }

    if ((params.at("micro_tile_height") == 8 || params.at("micro_tile_height") == 4) && params.at("micro_tile_width") ==  4){
      add_hps("Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0");
    }
    
    if ((params.at("micro_tile_height") == 8 || params.at("micro_tile_height") == 4) && params.at("micro_tile_width") ==  4){    
      add_hps("Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0");
    }
    
    if ((params.at("micro_tile_height")*params.at("micro_tile_height") == 24) && params.at("n_work_items_per_c_elm") >  1){    
      add_hps("Y48_X64_y3_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0");
    }
  
    if (params.at("micro_tile_height") == 3 && params.at("micro_tile_height") < params.at("micro_tile_width")){    
      add_hps("Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0");
    }

    if (params.at("micro_tile_height")*params.at("micro_tile_height") > 5 && params.at("micro_tile_height")*params.at("micro_tile_height") < 48 && params.at("n_work_items_per_c_elm") >  1){    
      add_hps("Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0");
    }
    
    if (gg.tA == gg.isColMajor && gg.tB != gg.isColMajor && params.at("micro_tile_height")*params.at("micro_tile_height") == 64){
      add_hps("Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1");
      add_hps("Y128_X128_y8_x8_U8_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1");
    }

    else if (gg.tA != gg.isColMajor && gg.tB == gg.isColMajor && params.at("micro_tile_height")*params.at("micro_tile_height") == 64){
      add_hps("Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1");
      add_hps("Y128_X128_y8_x8_U8_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1");
    }
  }
  
  /* shuffle them, which bounds the expected time to finding an improvement 
   * (prevents pathological case of all improving kernels at end of vector) 
   * currently, we shuffle after adding custom edges, might consider shuffling
   * before adding, to prevent getting trapped in previously found minima.*/
  std::random_device rd;
  std::default_random_engine default_engine(rd());
  std::shuffle(one_aways.begin(), one_aways.end(), default_engine);
  return one_aways;


}
  
  
std::vector<HyperParams> HyperParams::get_two_aways(const tinygemm::TinyGemmGeometry & gg){
  std::vector<HyperParams> two_aways;
  std::vector<HyperParams> one_aways = get_one_aways(gg);
  for (auto & hp : one_aways){
    std::vector<HyperParams> two_aways_via = hp.get_one_aways(gg);
    for (auto & hp2 : two_aways_via){
      auto blip = std::find(two_aways.begin(), two_aways.end(), hp2);
      if (blip == two_aways.end()) {
        two_aways.push_back(hp2);
      }
      else{
      }
    }
  }

  std::random_device rd;
  std::default_random_engine default_engine(rd());
  std::shuffle(two_aways.begin(), two_aways.end(), default_engine);    
  return two_aways;
}
  
  
/* TODO is this function ever used ? */
std::string HyperParams::get_string(){

  std::stringstream ss;
  ss << "Y" << params["macro_tile_height"] <<  "_X" << params["macro_tile_width"] << "_y" << params["micro_tile_height"] << "_x" << params["micro_tile_width"] << "_U" << params["unroll"] << "_P" << params["pad"] << "_GA" << params["group_allocation"] << "_APLU" << params["work_item_load_a_pll_to_unroll"] << "_BPLU" << params["work_item_load_b_pll_to_unroll"] << "_PU" << params["unroll_pragma"] << "_LIW" << params["load_to_lds_interwoven"] << "_MIW" << params["c_micro_tiles_interwoven"]  << "_ET" << 1 << "_ICE" << params["n_work_items_per_c_elm"] << "_UFO" << params.at("unroll_for_offset");
  
  return ss.str();
}


bool HyperParams::can_be_used_on(const tinygemm::TinyGemmGeometry & gg){
  unsigned m = gg.m;
  unsigned n = gg.n;
  bool tC = gg.tC;
  
  if (params.at("macro_tile_height") < tC ? n : m){
    return false;
  }
  
  if (params.at("macro_tile_width") < tC ? n : m){
    return false;
  }
  
  return true;
}



/* see dev/python/deepbench/deepbench_results.py : this is generated by get_kernel_cache_string, based on results running find with allotted_time =  30 seconds per problem, with three starting kernels for
 * small, medium, large: On a Fiji! 
 * TODO : regenerate with longer runs and more problems.
 * TODO : should not be a single vector, this has linear find time. At least seperate out isColMajor, tA, tB  
 * TODO : make this the starting kernel (get_initial_front) 
 * TODO : figure out how to make cache contain only reduced problems.... very important! */
std::vector<std::tuple<tinygemm::TinyGemmGeometry, std::string> > 
HyperParams::kernel_cache = {
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 128, 3072, 0, 0, 0} , "Y96_X64_y6_x4_U16_P1_GA2_APLU0_BPLU0_PU1_LIW1_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU0_BPLU0_PU1_LIW1_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 32, 4096, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 32, 1024, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 16, 7680, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW0_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 2560, 5124, 5124, 9124, 2560, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 64, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU1_BPLU0_PU0_LIW1_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 128, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 32, 2048, 0, 0, 0} , "Y32_X32_y2_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2560, 7133, 2560, 2560, 7133, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 16, 2048, 0, 0, 0} , "Y8_X16_y1_x2_U32_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 64, 2560, 0, 0, 0} , "Y24_X32_y3_x4_U16_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 2560, 2560, 8457, 35, 0, 0, 0} , "Y48_X48_y3_x3_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 16, 4096, 0, 0, 0} , "Y16_X16_y2_x2_U8_P1_GA2_APLU0_BPLU1_PU1_LIW1_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 7680, 5481, 7680, 7680, 5481, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 2048, 2048, 8457, 35, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA1_APLU1_BPLU1_PU0_LIW1_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 1760, 1760, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 2048, 35, 35, 8457, 2048, 0, 0, 0} , "Y24_X24_y3_x3_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 4096, 5124, 5124, 9124, 4096, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 32, 1760, 0, 0, 0} , "Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU1_LIW1_MIW1_ET1_ICE5_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 2560, 35, 35, 8457, 2560, 0, 0, 0} , "Y24_X40_y3_x5_U32_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 32, 7680, 0, 0, 0} , "Y48_X32_y3_x2_U32_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 4096, 4096, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 1760, 7133, 1760, 1760, 7133, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 4096, 7133, 4096, 4096, 7133, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 5124, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 128, 1024, 0, 0, 0} , "Y32_X64_y2_x4_U48_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 7000, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 64, 4096, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 64, 2560, 0, 0, 0} , "Y96_X64_y6_x4_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 128, 1024, 0, 0, 0} , "Y96_X32_y6_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 64, 3072, 0, 0, 0} , "Y32_X64_y2_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 35, 35, 8457, 2560, 0, 0, 0} , "Y24_X24_y3_x3_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 32, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 64, 4096, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 64, 2048, 0, 0, 0} , "Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU1_LIW1_MIW1_ET1_ICE9_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 16, 1760, 0, 0, 0} , "Y32_X16_y2_x1_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 4096, 35, 35, 8457, 4096, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE6_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 32, 1024, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 35, 1760, 35, 35, 8457, 1760, 0, 0, 0} , "Y24_X48_y3_x6_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU1_BPLU0_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 4096, 4096, 8457, 35, 0, 0, 0} , "Y32_X48_y2_x3_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 16, 2560, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 32, 2048, 0, 0, 0} , "Y48_X32_y3_x2_U64_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE8_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 35, 35, 8457, 4096, 0, 0, 0} , "Y24_X24_y3_x3_U32_P1_GA1_APLU1_BPLU0_PU1_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 7000, 2048, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2048, 2048, 2048, 2048, 16, 2048, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 64, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA3_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 32, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 128, 2048, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 16, 1024, 0, 0, 0} , "Y24_X16_y3_x2_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 16, 2560, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE9_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 5124, 5124, 9124, 2560, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 16, 1760, 0, 0, 0} , "Y24_X16_y3_x2_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 35, 35, 8457, 2048, 0, 0, 0} , "Y24_X16_y3_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 2560, 2560, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 16, 3072, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW0_ET1_ICE9_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 128, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5124, 5124, 2048, 2048, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 16, 1024, 0, 0, 0} , "Y48_X16_y3_x1_U32_P1_GA1_APLU0_BPLU1_PU1_LIW1_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 3072, 7435, 3072, 3072, 7435, 1024, 0, 0, 0} , "Y96_X96_y6_x6_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1760, 1760, 1760, 1760, 7000, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 64, 7680, 0, 0, 0} , "Y80_X32_y5_x2_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 32, 2560, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE9_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 35, 35, 1760, 1760, 8457, 35, 0, 0, 0} , "Y80_X80_y5_x5_U16_P1_GA1_APLU0_BPLU1_PU0_LIW1_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 128, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 5124, 5124, 9124, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA3_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1024, 1024, 3072, 3072, 64, 1024, 0, 0, 0} , "Y32_X32_y2_x2_U48_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3072, 3072, 1024, 1024, 32, 3072, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU1_BPLU0_PU1_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2560, 2560, 2560, 2560, 7000, 2560, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 16, 2560, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE8_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4096, 4096, 4096, 4096, 128, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 16, 4096, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA1_APLU1_BPLU0_PU1_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 5124, 5124, 9124, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 7000, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 1760, 1760, 32, 1760, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4096, 4096, 4096, 4096, 32, 4096, 0, 0, 0} , "Y16_X32_y2_x4_U32_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE8_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7680, 2560, 7680, 7680, 16, 2560, 0, 0, 0} , "Y40_X16_y5_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2048, 2048, 2048, 2048, 7000, 2048, 0, 0, 0} , "Y128_X96_y8_x6_U32_P1_GA1_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 1760, 5124, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 7680, 7680, 32, 2560, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE5_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2048, 7133, 2048, 2048, 7133, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1760, 1760, 35, 35, 8457, 1760, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3072, 1024, 3072, 3072, 64, 1024, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7680, 7680, 2560, 2560, 128, 7680, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2560, 2560, 2560, 2560, 64, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5124, 2048, 5124, 5124, 9124, 2048, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 128, 3072, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 32, 4096, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE8_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 32, 1024, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 16, 7680, 0, 0, 0} , "Y24_X16_y3_x2_U64_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW0_ET1_ICE6_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 2567, 5137, 5124, 9124, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 64, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 128, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 32, 2048, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2565, 7140, 2573, 2560, 7133, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 16, 2048, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 64, 2560, 0, 0, 0} , "Y64_X64_y4_x4_U32_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE6_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 2573, 2560, 8457, 35, 0, 0, 0} , "Y64_X32_y4_x2_U8_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 16, 4096, 0, 0, 0} , "Y32_X16_y4_x2_U32_P1_GA3_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 7685, 5488, 7693, 7680, 5481, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 2061, 2048, 8457, 35, 0, 0, 0} , "Y32_X48_y2_x3_U16_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 1773, 1760, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 2055, 48, 35, 8457, 2048, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 4103, 5137, 5124, 9124, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 32, 1760, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 2567, 48, 35, 8457, 2560, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 32, 7680, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 4109, 4096, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U8_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 1765, 7140, 1773, 1760, 7133, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 4101, 7140, 4109, 4096, 7133, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 5137, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA3_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 128, 1024, 0, 0, 0} , "Y48_X64_y3_x4_U16_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 7000, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 64, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW1_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 7000, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 64, 2560, 0, 0, 0} , "Y96_X64_y6_x4_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 128, 1024, 0, 0, 0} , "Y96_X64_y6_x4_U8_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 64, 3072, 0, 0, 0} , "Y32_X64_y2_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 48, 35, 8457, 2560, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 32, 2560, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 64, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 64, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 16, 1760, 0, 0, 0} , "Y8_X16_y1_x2_U32_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW0_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 32, 1024, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 1767, 48, 35, 8457, 1760, 0, 0, 0} , "Y24_X40_y3_x5_U8_P1_GA1_APLU1_BPLU1_PU1_LIW1_MIW0_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 128, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 40, 4103, 48, 35, 8457, 4096, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 128, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 16, 2560, 0, 0, 0} , "Y32_X16_y4_x2_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 64, 1760, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 32, 2048, 0, 0, 0} , "Y32_X32_y4_x4_U16_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 48, 35, 8457, 4096, 0, 0, 0} , "Y24_X24_y3_x3_U32_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 7000, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2053, 2055, 2061, 2048, 16, 2048, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 64, 2560, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 32, 2560, 0, 0, 0} , "Y64_X32_y4_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE3_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 128, 2048, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 16, 1024, 0, 0, 0} , "Y24_X16_y3_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 16, 2560, 0, 0, 0} , "Y16_X16_y2_x2_U32_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW0_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 5137, 5124, 9124, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 16, 1760, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 48, 35, 8457, 2048, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 2573, 2560, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 16, 3072, 0, 0, 0} , "Y32_X16_y2_x1_U32_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE6_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 128, 4096, 0, 0, 0} , "Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 5129, 5131, 2061, 2048, 9124, 5124, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 16, 1024, 0, 0, 0} , "Y16_X16_y2_x2_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 3077, 7442, 3085, 3072, 7435, 1024, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 128, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 1765, 1767, 1773, 1760, 7000, 1760, 0, 0, 0} , "Y128_X96_y8_x6_U16_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 64, 7680, 0, 0, 0} , "Y128_X64_y8_x4_U32_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE6_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 32, 2560, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE7_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 48, 35, 8457, 1760, 0, 0, 0} , "Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 4109, 4096, 8457, 35, 0, 0, 0} , "Y64_X32_y4_x2_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 5137, 5124, 9124, 2048, 0, 0, 0} , "Y96_X128_y6_x8_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1029, 1031, 3085, 3072, 64, 1024, 0, 0, 0} , "Y96_X64_y6_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 3077, 3079, 1037, 1024, 32, 3072, 0, 0, 0} , "Y32_X32_y2_x2_U32_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 2565, 2567, 2573, 2560, 7000, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 16, 2560, 0, 0, 0} , "Y48_X16_y3_x1_U48_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE8_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 4101, 4103, 4109, 4096, 128, 4096, 0, 0, 0} , "Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 16, 4096, 0, 0, 0} , "Y24_X16_y3_x2_U48_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW0_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 5137, 5124, 9124, 4096, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA3_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 7000, 2560, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 1765, 1767, 1773, 1760, 32, 1760, 0, 0, 0} , "Y64_X32_y4_x2_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 4101, 4103, 4109, 4096, 32, 4096, 0, 0, 0} , "Y24_X32_y3_x4_U32_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 7685, 2567, 7693, 7680, 16, 2560, 0, 0, 0} , "Y40_X16_y5_x2_U8_P1_GA1_APLU1_BPLU0_PU0_LIW1_MIW1_ET1_ICE2_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2053, 2055, 2061, 2048, 7000, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 1767, 5137, 5124, 9124, 1760, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 7693, 7680, 32, 2560, 0, 0, 0} , "Y48_X32_y3_x2_U32_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, true, false, 2053, 7140, 2061, 2048, 7133, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU0_LIW0_MIW1_ET1_ICE1_UFO1" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 40, 42, 1773, 1760, 8457, 35, 0, 0, 0} , "Y32_X48_y2_x3_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 3077, 1031, 3085, 3072, 64, 1024, 0, 0, 0} , "Y96_X64_y6_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 7685, 7687, 2573, 2560, 128, 7680, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, true, false, false, 2565, 2567, 2573, 2560, 64, 2560, 0, 0, 0} , "Y80_X64_y5_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0" ), 
  std::make_tuple<tinygemm::TinyGemmGeometry, std::string> ( {true, false, false, false, 5129, 2055, 5137, 5124, 9124, 2048, 0, 0, 0} , "Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0" ), 

};


 
}} // namespace
