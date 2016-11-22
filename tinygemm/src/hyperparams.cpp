#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <random>

#include "stringutilbase.hpp"
#include "stringutilbase.hpp"
#include "hyperparams.hpp"
#include "defaulthyperparams.hpp"
#include "stringutilbase.hpp"


namespace hyperparams{

HyperParams get_default_big(){
    
  std::map<std::string, unsigned> params;
  params["micro_tile_width"] = 8;  
  params["micro_tile_height"] = 8;
  params["macro_tile_width"] = 128;
  params["macro_tile_height"] = 128; 
  params["unroll"] = 8;
  params["pad"] = 1;    
  params["group_allocation"] = 1;
  params["work_item_load_a_pll_to_unroll"] = 0;
  params["work_item_load_b_pll_to_unroll"] = 1;
  params["unroll_pragma"] = 0;
  params["load_to_lds_interwoven"] = 0;
  params["c_micro_tiles_interwoven"] = 1;
  params["n_work_items_per_c_elm"] = 1;
  params["n_target_active_workgroups"] = 64;
  params["unroll_for_offset"] = 0;

  return HyperParams(params);
}


  

HyperParams get_default_medium(){
  std::map<std::string, unsigned> params;
  params["micro_tile_width"] = 4;  
  params["micro_tile_height"] = 4;
  params["macro_tile_width"] = 32;
  params["macro_tile_height"] = 32; 
  params["unroll"] = 16;
  params["pad"] = 1;    
  params["group_allocation"] = 1;
  params["work_item_load_a_pll_to_unroll"] = 0;
  params["work_item_load_b_pll_to_unroll"] = 1;
  params["unroll_pragma"] = 1;
  params["load_to_lds_interwoven"] = 0;
  params["c_micro_tiles_interwoven"] = 1;
  params["n_work_items_per_c_elm"] = 1;
  params["n_target_active_workgroups"] = 64;
  params["unroll_for_offset"] = 0;
    
  return HyperParams(params);
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


  HyperParams::HyperParams(std::map<std::string, unsigned> params):params(params){
    do_checks();
    

  }
  
  
  void HyperParams::do_checks(){
    for (auto & x : all_hyper_param_names){
      if (params.count(x) == 0){
        std::string errm("The parameter `");
        errm += x;
        errm += "', which should appear as a hyper-parameter but appears not to, should be included\n";
        throw std::runtime_error(errm);
      }
    }
    
    for (auto & x : params){
      auto blip = std::find(all_hyper_param_names.cbegin(), all_hyper_param_names.cend(), x.first);
      if (blip == all_hyper_param_names.cend()) {
      
      //all_hyper_param_names.count(x.first) == 0){
        std::string errm("The parameter `");
        errm += x.first;
        errm += "', which appears in the user-defined list of hyper-parameter, is not recognised\n";
        throw std::runtime_error(errm);        
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
  

  /////* Add, to one_aways, a HyperParams object initialised as default_hp and then modified by parameters */
  ////void add_hyperparam(const HyperParams & default_hp, unsigned Y, unsigned X, unsigned y, unsigned x, unsigned U, unsigned P, unsigned GA, unsigned APLU, unsigned BPLU, unsigned PU, unsigned LIW, unsigned MIW, unsigned ICE, unsigned UFO, std::vector<HyperParams> & one_aways){
    ////HyperParams hp(default_hp.params);
    ////hp.params["micro_tile_height"] = y;
    ////hp.params["micro_tile_width"] = x;
    ////hp.params["macro_tile_height"] = Y;
    ////hp.params["macro_tile_width"] = X;
    ////hp.params["group_allocation"] = GA;
    ////hp.params["work_item_load_a_pll_to_unroll"] = APLU;
    ////hp.params["work_item_load_b_pll_to_unroll"] = BPLU;
    ////hp.params["unroll_pragma"] = PU;
    ////hp.params["load_to_lds_interwoven"] = LIW;
    ////hp.params["c_micro_tiles_interwoven"] = MIW;
    ////hp.params["unroll"] = U;
    ////hp.params["n_work_items_per_c_elm"] = ICE;
    ////hp.params["pad"] = P;
    ////hp.params["unroll_for_offset"] = UFO;
    ////one_aways.push_back(hp);
  ////}
  
  void add_hyperparam(const HyperParams & default_hp, std::string hyperstring, std::vector<HyperParams> & one_aways){
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
    one_aways.push_back(hp);
  }
    
  
  std::vector<HyperParams> HyperParams::get_one_aways(const gemmgeometry::Geometry & gg){
    
    if (gg.m < 16 || gg.n < 16){
      throw std::runtime_error("Currently, if matrix C has a dimension less that 16, it is not supported. If you are seeing this, please remind to do fix it..  jnewling@idiap.ch ");
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
              micro_tile_edges[ {x,y} ].push_back( {nx, ny} );
            }
          }
        }
      }
    }
    
    for (auto & micro_tile : micro_tile_edges[ {params.at("micro_tile_height") , params.at("micro_tile_width") } ]){
      
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
       * This is the place to add some edges which experience shows can tunnel out of local minima, or
       * lead to some kernels which have found to be very good on some problems  */
      
      auto add_hps = [this, & one_aways](std::string hparamstring){
        add_hyperparam(*this, hparamstring, one_aways);
      };
      
      /* custom edge 2. Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE6_UFO0  */
      if (params.at("micro_tile_height")*params.at("micro_tile_width") <=4 ){
        add_hps("Y16_X16_y2_x2_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE6_UFO0");
      }
  
      /* custom edge 3. Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ET1_ICE5_UFO0 */    
      if (params.at("micro_tile_height")*params.at("micro_tile_width") <=16 ){
        add_hps("Y48_X32_y3_x2_U16_P1_GA2_APLU1_BPLU0_PU0_LIW0_MIW1_ET1_ICE5_UFO0");
      }
  
      /* custom edge 4. Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE4_UFO0 */
      if (params.at("micro_tile_height")*params.at("micro_tile_width") <=20 ){
        //add_hyperparam(*this, 64, 64, 4, 4, 16, 1, 2, 0, 0, 0, 1, 1, 4, 0, one_aways);
        add_hps("Y64_X64_y4_x4_U16_P1_GA2_APLU0_BPLU0_PU0_LIW1_MIW1_ET1_ICE4_UFO0");
      }
      
      /* custom edge 5. Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0 */
      if (params.at("micro_tile_height")*params.at("micro_tile_width") >=16 ){
        add_hps("Y128_X128_y8_x8_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0");
      }
      
      /* custom edge 6  Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0 */
      if (params.at("micro_tile_height")*params.at("micro_tile_width") >=8 ){
        add_hps("Y80_X64_y5_x4_U16_P1_GA2_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE2_UFO0");
      }
  
      /* custom edge 7   Y96_X64_y6_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0 */
      if (params.at("micro_tile_height") >= params.at("micro_tile_width") &&  params.at("micro_tile_height")*params.at("micro_tile_width") >=10){
        add_hps("Y96_X64_y6_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE4_UFO0");
      }
  
            
      /* custom edge 8 Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0 */
      if ((params.at("micro_tile_height") == 8 || params.at("micro_tile_height") == 4) && params.at("micro_tile_width") ==  4){
        add_hps("Y128_X64_y8_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE3_UFO0");
      }
      
  
      /* custom edge 9 Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0 */
      if ((params.at("micro_tile_height") == 8 || params.at("micro_tile_height") == 4) && params.at("micro_tile_width") ==  4){    
        add_hps("Y64_X64_y4_x4_U16_P1_GA3_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0");
      }
      
  
      /* custom edge 10 Y48_X64_y3_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0 */
      if ((params.at("micro_tile_height")*params.at("micro_tile_height") == 24) && params.at("n_work_items_per_c_elm") >  1){    
        add_hps("Y48_X64_y3_x4_U16_P1_GA2_APLU0_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0");
      }
    
      /* custom edge 11 Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0 */
      if (params.at("micro_tile_height") == 3 && params.at("micro_tile_height") < params.at("micro_tile_width")){    
        add_hps("Y24_X40_y3_x5_U16_P1_GA1_APLU1_BPLU1_PU0_LIW0_MIW1_ET1_ICE1_UFO0");
      }
      
      /* custom edge 12 Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0 */
      if (params.at("micro_tile_height")*params.at("micro_tile_height") > 5 && params.at("micro_tile_height")*params.at("micro_tile_height") < 48 && params.at("n_work_items_per_c_elm") >  1){    
        add_hps("Y64_X64_y4_x4_U16_P1_GA1_APLU0_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO0");
      }
      
      /* custom edge 13 Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1 */
      /* custom edge 14 Y128_X128_y8_x8_U8_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1 */
      if (gg.tA == gg.isColMajor && gg.tB != gg.isColMajor && params.at("micro_tile_height")*params.at("micro_tile_height") == 64){
        add_hps("Y128_X128_y8_x8_U8_P1_GA1_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1");
        add_hps("Y128_X128_y8_x8_U8_P1_GA2_APLU1_BPLU1_PU1_LIW0_MIW1_ET1_ICE1_UFO1");
      }

      /* custom edge 15 Y128_X128_y8_x8_U8_P1_GA1_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1 */
      /* custom edge 16 Y128_X128_y8_x8_U8_P1_GA2_APLU0_BPLU0_PU1_LIW0_MIW1_ET1_ICE1_UFO1 */
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
  
  
  std::vector<HyperParams> HyperParams::get_two_aways(const gemmgeometry::Geometry & gg){
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
  
  
  bool HyperParams::can_be_used_on(const gemmgeometry::Geometry & gg){
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



  
  
  std::vector<HyperParams> get_initial_front(const gemmgeometry::Geometry & gg, bool enforce_deterministic){

    /* TODO : initial front should be determined from a cache file. What follows can correspond to an initial cache */
    std::vector<HyperParams> initial_front;
    
    
    /* if the matrix is `big' (C is 1024**2) and not too skinny, add the best (Catalyst) kernel `big matrix' 
     * see figure baidu_micro_ksplit_best_hists.pdf for `motivation' */
    float log_c_surface_area = log2(static_cast<float>(gg.m*gg.n));
    if (log_c_surface_area > 20 && gg.m >= 128 && gg.n >= 128){
      HyperParams hp = get_default_big();
      initial_front.push_back(hp);
    }
    
    else if (log_c_surface_area > 17 && gg.m >= 32 && gg.n > 32){
      HyperParams hp = get_default_medium();
      initial_front.push_back(hp);
    }
    
    else if (gg.m >= 16 && gg.n >= 16){
      HyperParams hp = get_default_small(enforce_deterministic);
      initial_front.push_back(hp);      
    }
    
        
    else {
      /* The case  of  (gg.m < 16 || gg.n < 16) */
      throw std::runtime_error("Currently, we do not support matrices which are skinnier (m or n) than 16. This can easily be fixed... please contanct me at jnewling@idiap.ch ");
    }
    return initial_front;
  
  }

 
} // namespace
