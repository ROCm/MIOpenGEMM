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
  enum eMatrix {matA = 0, matB, matC, nMatrices};
  enum eSpecial {undefined = -1};
  enum eBinary {no = 0, yes = 1};
  enum eGraphType {ChiralType, NonChiralType, nGraphTypes};
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

/* for example, we want to couple [matA][MIC] with [matB][MIC]  */
/* design choice : going with pairs as opposed to tuples as they have nicer initializer_list behavioUr*/
std::vector<std::pair< std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> > > coupled_parameters;


//std::vector<std::vector<std::string>> GraphKeys;

SUHP();
};

extern const SUHP suHP;



class Graph{

protected:
  const std::vector<std::string> * ptr_graphkeys;
  virtual void set_ptr_graphkeys() = 0;
  
public:

  
  Graph(unsigned size);
  
  /* all the possible values of a hyper parameter */
  /* example : range[nsHP::MIC] --> {1,2,3,4,5,6,7,8} */
  std::vector<std::vector<unsigned> > range;  

  /* all the possible edges from all the possible hyper parameter */
  /* example : graph[nsHP::MIC] is a map; graph[nsHP::MIC][1] --> {2,3,4} */
  std::vector<std::map<unsigned, std::vector<unsigned> > > graph;

  /* a subset of range, the possible values returned on a request for a random value */  
  /* example : default_start_range[nsHP::MIC] --> {2,8}. Note that if the default start range 
   * is dependent on geometry, the gg_start_range will be used instead, with default_start_range element empty*/  
  std::vector<std::vector<unsigned> > default_start_range;  
  
  std::vector<std::function<std::vector<unsigned>(const tinygemm::TinyGemmGeometry & gg)>> gg_start_range;
  
  std::vector<unsigned> get_start_range(unsigned hpi, const tinygemm::TinyGemmGeometry & gg) const;

  void update_range();
  
  void initialise();
  
  void confirm_start_is_subset();

  virtual void set_edges() = 0;

  virtual void set_start_range() = 0;

};

class NonChiralGraph : public Graph{
  public:
    NonChiralGraph();
    virtual void set_ptr_graphkeys() override final;
    virtual void set_edges() override final;
    virtual void set_start_range() override final;

};

class ChiralGraph : public Graph{
  public: 
    ChiralGraph();  
    virtual void set_ptr_graphkeys() override final;
    virtual void set_edges() override final;
    virtual void set_start_range() override final;
    virtual void set_chirality_specific() = 0;
};

class AChiralGraph : public ChiralGraph{
  public:
    //AChiralGraph();
    virtual void set_chirality_specific() override final;
};

class BChiralGraph : public ChiralGraph{
  public:
    //BChiralGraph();
    virtual void set_chirality_specific() override final;
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

  HyperParams(const std::vector<std::vector<unsigned>> & params); 
  std::vector<XHPs> v_xhps;

  

  
  

  
public:

   
  void replace_undefined_randomly(const tinygemm::TinyGemmGeometry & gg);
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
  
  

  
  HyperParams();  
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways();
  std::string get_string() const;
  void checks() const;

};  


//HyperParams get_default(const tinygemm::TinyGemmGeometry & gg, std::string constraint_string);
HyperParams get_hp_start(FindStartType fst, std::string constraint_string, const tinygemm::TinyGemmGeometry & gg);

std::vector<std::vector<unsigned>> get_params_from_string(const std::string & hyperstring, bool expect_full_hyperstring);

}
}

#endif


