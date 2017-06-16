#include <vector>
#include <sstream>
#include <MIOpenGEMM/tiling.hpp>
#include <MIOpenGEMM/error.hpp>


#include <iostream>
namespace MIOpenGEMM{
namespace tiling{

std::vector<unsigned> get_multiples(unsigned N){
  std::vector<unsigned> multiples;
  for (unsigned k = N; k > 0; --k){
    if (N%k == 0){
      multiples.push_back(k);
    }
  }
  return multiples;
}

void set_tile_dimensions_no_checks(unsigned & tH, unsigned & tW, unsigned TH, unsigned TW, unsigned tS){
  for (auto & multiple_of_TH : get_multiples(TH)){
    if ((tS % multiple_of_TH == 0) && ((tS / multiple_of_TH) <= TW)){
      tH = multiple_of_TH;
      tW = tS / tH;
      break;
    }
  }
}

std::tuple<bool, std::string> get_tileability(unsigned TH, unsigned TW, unsigned tS){
  
  std::stringstream ss_tileable_status;
  
  if (tS == 0){
    throw miog_error("In get_tileability, and tS is zero. This is worse than non-tileable, there is probably a bad input parameter.");
  }

  std::string set_ds("");  
  std::stringstream input_ss;
  input_ss << "\n" << "TH : " << TH << " TW : " << TW << " tS : " << tS;
  std::string input_string = input_ss.str();

  if ((TH*TW) % tS  != 0){
    ss_tileable_status << "Areas of micro and macro tiles are incompatible : " << input_string;
    std::make_tuple(false, ss_tileable_status.str());
  }
  
  unsigned tH, tW;
  set_tile_dimensions_no_checks(tH, tW, TH, TW, tS);
  
  
  if (tH == 0){
    ss_tileable_status << "Impossible tiling problem in get_tile_dimensions : " << input_string;
    std::make_tuple(false, ss_tileable_status.str());
  }
  

  if (tW  > tH){
    // this would be a pedantic error: no `tall' tile. best `wide' one " + bla
  }
  
  if (TW % tW != 0 || TH % TH != 0 || tW*tH != tS){
    std::stringstream err_ss;
    err_ss 
    << "Problem in get_tileability. This isn't even non-tileable, this is a logic error. The found micro tile size is not consistent with the macro tile : "
    << input_string << "   tH : " << tH << " tW  " << tW;
    throw miog_error(err_ss.str()); 
  }
  
  /* ran the gauntlet successfully */
  return std::make_tuple(true, ""); 
  
}

void set_tile_dimensions(unsigned & tH, unsigned & tW, unsigned TH, unsigned TW, unsigned tS, bool tall){

  bool is_tileable;
  std::string tileable_status;
  
  std::tie(is_tileable, tileable_status) = get_tileability(TH, TW, tS);
  if (is_tileable == false){
    throw miog_error("In set_tile_dimensions, and the problem is not tileable. Call get_tileability as a check before set_tile_dimensions to catch this case without throwing an error. The string returned from set_tile_dimensions was : " + tileable_status);
  }
  
  /* switch (tW <-> tH) and (TW <-> TH) */
  if (tall == false)  {
    set_tile_dimensions_no_checks(tW, tH, TW, TH, tS);
  }

  else{  
    set_tile_dimensions_no_checks(tH, tW, TH, TW, tS);
  }
}


}
}
