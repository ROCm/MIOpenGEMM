#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <map>

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
  enum eSpecial {undefined = -1};
  enum eBinary {no = 0, yes = 1};
  enum eMat {matA, matB, matC, nSubGs};
}

namespace hyperparams{

class SubG{

public:
  SubG(unsigned nHPs, const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull);
 
  SubG() = default;
  


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

  std::string subg_cs;
  bool subg_csfull;
  std::vector<unsigned> constraints;
  
    
  void initialise();
  void set_constraints();
  void initialise_range_from_edges();
  void confirm_start_is_subset();
  virtual void initialise_maps() = 0;
  virtual void set_edges() = 0;
  virtual void set_start_range() = 0;
  
  std::string get_string(unsigned hpi);

    

};


class CSubG : public SubG{
  public:
    CSubG() = default;
    CSubG(const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull);
    virtual void initialise_maps() override final;
    virtual void set_edges() override final;
    virtual void set_start_range() override final;

};



class ChiralSubG : public SubG{
  public: 
    ChiralSubG() = default;
    ChiralSubG(const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull);  
    virtual void initialise_maps() override final;
    virtual void set_edges() override final;
    virtual void set_start_range() override final;
    virtual void set_chirality_specific() = 0;
};

class ASubG : public ChiralSubG{
  public:
    ASubG() = default;
    ASubG(const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull):ChiralSubG(gg, cs, csfull){}
    virtual void set_chirality_specific() override final;
};

class BSubG : public ChiralSubG{
  public:
    BSubG() = default;
    BSubG(const tinygemm::TinyGemmGeometry & gg, std::string cs, bool csfull):ChiralSubG(gg, cs, csfull){}
    virtual void set_chirality_specific() override final;
};


class Graph{

  private:
    ASubG asubg;
    BSubG bsubg;
    CSubG csubg;

  public:
    const tinygemm::TinyGemmGeometry * ptr_gg;

    std::vector<SubG * > p_subgs;
    
    std::map <char, unsigned> graphind;  
    std::vector<char> graphchar;
    
    std::vector<std::pair< std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> > > coupled_parameters;

    Graph(const tinygemm::TinyGemmGeometry & gg, std::string constraint_string, bool full_constraints_expected);

};






class XHPs{
  
  public:
    std::vector<unsigned> vs;
    XHPs(unsigned nHPs)  { vs = std::vector<unsigned>(nHPs, nsHP::undefined); }
      
};


class HyperParams{

private:
  const Graph * p_graph;
  std::vector<XHPs> v_xhps;
  //HyperParams(const std::vector<std::vector<unsigned>> & params); 
 
public:
  void replace_undefined_randomly();
  void replace(const std::vector<std::vector<unsigned>> & partial_params);
  void replace_where_source_defined(const std::vector<std::vector<unsigned>> & params);
  bool in_graph();
   
 
  const XHPs & at(nsHP::eMat subgtype) const {return  v_xhps[subgtype]; }
  XHPs & at(nsHP::eMat subgtype) {return  v_xhps[subgtype]; }
  
  //TODO : deprecate these
  const XHPs & at(char X) const {
    X = (X == 'a' ? 'A' : X);
    X = (X == 'b' ? 'B' : X); 
    X = (X == 'c' ? 'C' : X);     
    return v_xhps[p_graph->graphind.at(X)]; 
  }
  
  XHPs & at(char X) {
    X = (X == 'a' ? 'A' : X);
    X = (X == 'b' ? 'B' : X); 
    X = (X == 'c' ? 'C' : X);     
    return v_xhps[p_graph->graphind.at(X)]; 
  }
    
  HyperParams(const Graph & graph);
  
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways();
  std::string get_part_string(char X) const;
  std::string get_string() const;
  void checks() const;

};  


HyperParams get_hp_start(FindStartType fst, const Graph & graph);



}
}

#endif


