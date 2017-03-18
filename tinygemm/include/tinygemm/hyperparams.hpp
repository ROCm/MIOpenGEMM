#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>
#include <random>

#include "tinygemmgeometry.hpp"

namespace tinygemm{


enum EnumMAC {a4b8 = 0, a8b4 = 1, a8b8 = 2, a8b16 = 3, a16b8 = 4, a16b16 = 5};
enum EnumGA  {byrow = 1, bycol = 2, sucol = 3};

/* if you're going to add a parameter here, make sure to add it BEFORE the final count */
enum EnumChiralHP {MIC = 0, PAD, PLU, LIW, MIW, WOS, nChiralHPs};
/* the strings must appear in same order as enum */
const std::vector<std::string> ChiralHPKeys = {"MIC", "PAD", "PLU", "LIW", "MIW", "WOS"};
// TODO : confirm somewhere that nChiralHPs = chiralHPKeys.size(), etc ditto

enum EnumNonChiralHP {UNR = 0, GAL, PUN, ICE, NAW, UFO, MAC, nNonChiralHPs};
const std::vector<std::string> NonChiralHPKeys = {"UNR", "GAL", "PUN", "ICE", "NAW", "UFO", "MAC"};

const std::map<char, const std::vector<std::string> & > HPKeys = {
{'A', ChiralHPKeys},  {'B', ChiralHPKeys}, {'C', NonChiralHPKeys} };

std::map<char, unsigned> nHPs = {{'A', nChiralHPs}, {'B', nChiralHPs}, {'C', nNonChiralHPs}};
  

enum EnumMatrix {matA = 0, matB, matC, nMatrices};
const std::map<char, EnumMatrix> matNums = {{'C', matC}, {'B', matB}, {'A', matA}};



namespace hyperparams{
  
class Graph{
  
public:
  /* example : graph[MIC] is a map; graph[MIC][1] --> {2,3,4} */
  const std::vector<std::map<unsigned, std::vector<unsigned> > > graph;
  /* example : range[MIC] --> {1,2,3,4,5,6,7,8} */
  const std::vector<std::vector<unsigned> > range;  
  void update_range();
};


Graph get_g_chiral();
Graph get_g_non_chiral();



class XHPs{
  
  protected:
    const std::vector<unsigned> & hpkeys;
    const Graph & hpgraph;
      
  public:
    std::vector<unsigned> values;
    std::vector<unsigned> importances;
    std::string get_string() const;
    void check() const;
    
    XHPs(const std::vector<unsigned> & hpkeys_, const Graph & hpgraph_, unsigned nHPsTest):hpkeys(hpkeys_), hpgraph(hpgraph_){
      if (nHPsTest != hpkeys.size() || nHPsTest != hpgraph.graph.size() ){
        throw tinygemm_error("There is a discrepency in the number of hyper parameters in XHPs constructor");
      }
    }
    
};


class HyperParams{

private:
  std::vector<XHPs> v_xhps;

public:

  const XHPs & at(EnumMatrix matX) const {return  v_xhps[matX]; }
  XHPs & at(EnumMatrix matX) {return  v_xhps[matX]; }
  
  HyperParams(const std::map<char, std::map<std::string, unsigned> > & );

  /* take in hyper-parameter string and return a HyperParam object */
  HyperParams(const std::string & hyperstring);
  
  HyperParams() = delete;
  
  bool operator == (const HyperParams & hpr);
  
  std::vector<HyperParams> get_one_aways();

  void check_map_keys(const std::map<char, std::map<std::string, unsigned> > & params);

  std::string get_string() const;
  

};  


HyperParams get_default();

}
}

#endif


