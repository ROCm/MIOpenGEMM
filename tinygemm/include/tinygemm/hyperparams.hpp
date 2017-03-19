#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>
#include <random>

#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{


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
  enum eNonChiral {UNR = 0, GAL, PUN, ICE, NAW, UFO, MAC, nNonChiralKeys};  
  enum eMatrix {matA = 0, matB, matC, nMatrices};
}

namespace hyperparams{

struct SUHP{
/* where for example, ChiralKeys[PAD] is "PAD" */
std::vector<std::string> ChiralKeys;
std::map<std::string, unsigned> ChiralVals;

std::vector<std::string> NonChiralKeys;
std::map<std::string, unsigned> NonChiralVals;

std::vector<char> MatKeys;
std::map<char, unsigned> MatVals;

std::vector<std::vector<std::string>> HPKeys;
std::vector<std::map<std::string, unsigned>> HPVals;
std::vector<unsigned> nHPs;



SUHP();
};

extern const SUHP suHP;



class Graph{
  
public:
  /* example : graph[nsHP::MIC] is a map; graph[nsHP::MIC][1] --> {2,3,4} */
  std::vector<std::map<unsigned, std::vector<unsigned> > > graph;
  /* example : range[nsHP::MIC] --> {1,2,3,4,5,6,7,8} */
  std::vector<std::vector<unsigned> > range;  
  void update_range();
};



class XHPs{
  
  public:
    /* design choice. I think that pointers to vectors ugly. But, using const ref class variables means manually enforcing copy/default constructors */ 
    const std::vector<std::string> * ptr_hpkeys;
    const Graph * ptr_hpgraph;
    std::vector<unsigned> vs;
    std::vector<unsigned> importances;
    std::string get_string() const;
    void check() const;
    
    XHPs(
    const std::vector<std::string> * ptr_hpkeys_, const Graph * ptr_hpgraph_, unsigned nHPsIn): 
    ptr_hpkeys(ptr_hpkeys_), ptr_hpgraph(ptr_hpgraph_)
    {
      if (nHPsIn != ptr_hpkeys->size() || nHPsIn != ptr_hpgraph->graph.size() ){
        throw tinygemm_error("There is a discrepency in the number of hyper parameters in XHPs constructor");
      }
      vs.resize(nHPsIn);
      importances.resize(nHPsIn);
    }
};


class HyperParams{

private:
  std::vector<XHPs> v_xhps;
  HyperParams(const std::vector<std::vector<unsigned>> & params); 
  
public:

  
 
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
  
  HyperParams() = delete;
  
  bool operator == (const HyperParams & hpr);
  
  std::vector<HyperParams> get_one_aways();

  std::string get_string() const;
  
  void checks() const;
  

};  


HyperParams get_default();

}
}

#endif


