#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>
#include <random>
#include <functional>

#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{

enum class FindStartType {Default, Random};


/* design note: a safer choice than namespacing enums is to use enum class. */ 
/* but this would involve static casting from the enumerated type to unsigned all the time */

namespace nsMAC{
  enum eMAC {a4b8 = 0, a8b4 = 1, a8b8 = 2, a8b16 = 3, a16b8 = 4, a16b16 = 5};
}

namespace nsGAL{
  enum eGAL {byrow = 1, bycol = 2, sucol = 3};
}

namespace nsHP{
  /* if you're going to add a parameter here, make sure to add it BEFORE the final count */
  enum eChiral {MIC = 0, PAD, PLU, LIW, MIW, WOS, nChiralHPs};
  enum eNonChiral {UNR = 0, GAL, PUN, ICE, NAW, UFO, MAC, nNonChiralHPs};  
  //enum eMatrix {matA = 0, matB, matC, nMatrices};
  enum eSpecial {undefined = -1};
  enum eBinary {no = 0, yes = 1};
  enum eSubGType {SubGChiralA, SubGChiralB, SubGNonChiral, nGraph};
}

namespace hyperparams{

class Graph{

  private:
    ASubG asubg;
    BSubG bsubg;
    CSubG csubg;

  public:
    tinygemm::TinyGemmGeometry * ptr_gg;

    std::vector<SubG * > p_subgs;
    std::map <char, unsigned> graphind;  
    std::vector<std::pair< std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> > > coupled_parameters;

    Graph(const tinygemm::TinyGemmGeometry & gg); 
    std::vector<std::vector<unsigned>> get_params_from_string(const std::string & hyperstring, bool expect_full_hyperstring);
    
};




class SubG{

public:
  SubG(unsigned nHPs, const tinygemm::TinyGemmGeometry & gg); //yes
 
  unsigned nHPs; 
  const tinygemm::TinyGemmGeometry * ptr_gg;
  std::vector<std::string> Keys;
  std::map<std::string, unsigned> Vals;
  
  /* all the possible values of a hyper parameter */
  /* example : range[nsHP::MIC] --> {1,2,3,4,5,6,7,8} */
  std::vector<std::vector<unsigned> > range;  

  /* all the possible edges from all the possible hyper parameter */
  /* example : edges[nsHP::MIC] is a map; edges[nsHP::MIC][1] --> {2,3,4} */
  std::vector<std::map<unsigned, std::vector<unsigned> > > edges;

  /* a subset of range, the possible values returned on a request for a random value */  
  /* example : start_range[nsHP::MIC] --> {2,8}. It can depend on geometry (from initialisation) */  
  std::vector<std::vector<unsigned> > start_range;
  
  void initialise();
  void initialise_range_from_edges(); //yes
  void confirm_start_is_subset(); //TODO
  virtual void initialise_maps() = 0;
  virtual void set_edges() = 0;
  virtual void set_start_range() = 0;

};


class CSubG : public SubG{
  public:
    CSubG(const tinygemm::TinyGemmGeometry & gg);
    virtual void initialise_maps() override final; //yes
    virtual void set_edges() override final;
    virtual void set_start_range() override final;

};



class ChiralSubG : public SubG{
  public: 
    ChiralSubG(const tinygemm::TinyGemmGeometry & gg);  
    virtual void initialise_maps() override final; //yes
    virtual void set_edges() override final;
    virtual void set_start_range() override final;
    virtual void set_chirality_specific() = 0;
};

class ASubG : public ChiralSubG{
  public:
    virtual void set_chirality_specific() override final;
};

class BSubG : public ChiralSubG{
  public:
    virtual void set_chirality_specific() override final;
};







class XHPs{
  
  public:
    std::vector<unsigned> vs;
    std::vector<unsigned> importances;
    XHPs(unsigned nHPs) vs {nHPs}, importances {nHPs} {}
};


class HyperParams{

private:
  const Graph * p_graph;
  std::vector<XHPs> v_xhps;
  HyperParams(const std::vector<std::vector<unsigned>> & params); 
 
public:
  void replace_undefined_randomly();
  void replace(const std::vector<std::vector<unsigned>> & partial_params);
  void replace_where_source_defined(const std::vector<std::vector<unsigned>> & params);
  bool satisfies_where_source_defined(const std::vector<std::vector<unsigned>> & params);
 
  const XHPs & at(nsHP::eMatrix matX) const {return  v_xhps[matX]; }
  XHPs & at(nsHP::eMatrix matX) {return  v_xhps[matX]; }
  
  //TODO : deprecate these
  const XHPs & at(char X) const {
    X = (X == 'a' ? 'A' : X);
    X = (X == 'b' ? 'B' : X); 
    X = (X == 'c' ? 'C' : X);     
    return v_xhps[suHP.MatVals.at(X)]; 
  }
  
  XHPs & at(char X) {
    X = (X == 'a' ? 'A' : X);
    X = (X == 'b' ? 'B' : X); 
    X = (X == 'c' ? 'C' : X);     
    return v_xhps[suHP.MatVals.at(X)]; 
  }
 
  /* take in hyper-parameter string and return a HyperParam object */
  HyperParams(const std::string & hyperstring);
  HyperParams() = default;
  
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways();
  std::string get_string() const;
  void checks() const;

};  


HyperParams get_hp_start(FindStartType fst, std::string constraint_string, const Graph * const graphs);



}
}

#endif


