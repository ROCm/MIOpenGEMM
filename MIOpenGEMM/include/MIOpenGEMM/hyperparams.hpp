#ifndef HYPERKERNELPARAMS_HPP
#define HYPERKERNELPARAMS_HPP

#include <vector>
#include <array>
#include <map>

#include <functional>

#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/openclutil.hpp>

#include <array>


namespace MIOpenGEMM{

enum class FindStartType {Default, Random};

/* design note: a safer choice than namespacing enums is to use enum class. */ 
/* but this would involve static casting from the enumerated type to unsigned all the time */


namespace nsMAC{

std::tuple<bool, std::string, std::array<unsigned, 2>> get_mac_grid(unsigned mac, unsigned skew);

}



namespace nsGAL{
  enum eGAL {byrow = 1, bycol = 2, sucol = 3};
}

namespace nsHP{
  /* if you're going to add a parameter here, make sure to add it BEFORE the final count */
  enum eChiral {MIC = 0, PAD, PLU, LIW, MIW, WOS, nChiralHPs};
  enum eNonChiral {UNR = 0, GAL, PUN, ICE, NAW, UFO, MAC, SKW, nNonChiralHPs};  
  enum eSpecial {undefined = -1};
  enum eBinary {no = 0, yes = 1};
}



namespace hyperparams{



class KeysVals{
  public:
    std::vector<std::string> keys;
    std::map <std::string, unsigned> vals;
    unsigned nHPs; 
};

std::vector<unsigned> get_constraints(std::string subg_cs, bool subg_csfull, const KeysVals * p_kv, char subg_hash);

extern const std::map <char, unsigned> graphind;  
extern const std::vector<char> graphchar;

extern const KeysVals chiral_kv;
extern const KeysVals non_chiral_kv;


class SubG{

public:
  SubG(unsigned nHPs, const Geometry & gg, std::string cs, bool csfull, const openclutil::OpenCLDeviceInfo * ptr_devinfo);
 
  SubG() = default;

  void apply_constraints();

  unsigned nHPs; 
  const Geometry * ptr_gg;
  //const std::vector<std::string> * 
  
  const KeysVals * ptr_keys_vals;
  //const std::map<std::string, unsigned> * ptr_vals;
  

  /* all the possible edges from all the possible hyper parameter */
  /* example : edges[nsHP::MIC] is a map; edges[nsHP::MIC][1] --> {2,3,4} */
  std::vector<std::map<unsigned, std::vector<unsigned> > > edges;

  /* all the possible values of a hyper parameter */
  /* example : range[nsHP::MIC] --> {1,2,3,4,5,6,7,8} */
  std::vector<std::vector<unsigned> > range;  

  /* a subset of range, the possible values returned on a request for a random value */  
  /* example : start_range[nsHP::MIC] --> {2,8}. It can depend on geometry (from initialisation) */  
  std::vector<std::vector<unsigned> > start_range;

  std::string subg_cs;
  bool subg_csfull;
  
  const openclutil::OpenCLDeviceInfo * ptr_devinfo;
  
  std::vector<unsigned> constraints;
  
    
  void initialise();
  void set_constraints();
  void initialise_range_from_preconstraint_edges();
  void initialise_start_range_from_range();    
  void confirm_start_is_subset();
  virtual void initialise_maps() = 0;
  virtual void set_preconstraint_edges() = 0;
  /* used if start range should be a strict subset of range */
  virtual void manual_override_start_range() = 0;
  virtual char get_char() = 0;
  
  std::string get_string(unsigned hpi);
  std::string get_edges_string(unsigned hpi);
  std::string get_range_string(unsigned hpi);
  std::string get_start_range_string(unsigned hpi);

  
  void force_start_node(std::vector<unsigned>);
};


class CSubG : public SubG{
  public:
    CSubG() = default;
    CSubG(const Geometry & gg, std::string cs, bool csfull, const openclutil::OpenCLDeviceInfo * ptr_devinfo);
    virtual void initialise_maps() override final;
    virtual void set_preconstraint_edges() override final;
    virtual void manual_override_start_range() override final;
    virtual char get_char() override final {return 'C';}
};



class ChiralSubG : public SubG{
  public: 
    ChiralSubG() = default;
    ChiralSubG(const Geometry & gg, std::string cs, bool csfull, const openclutil::OpenCLDeviceInfo * ptr_devinfo);  
    virtual void initialise_maps() override final;
    virtual void set_preconstraint_edges() override final;
    virtual void manual_override_start_range() override final;
    void set_chirality_specific_start_range_base(unsigned non_unroll_dimension);
    virtual void set_chirality_specific_start_range() = 0;
    virtual char get_char() = 0;

};

class ASubG : public ChiralSubG{
  public:
    ASubG() = default;
    ASubG(const Geometry & gg, std::string cs, bool csfull, const openclutil::OpenCLDeviceInfo * ptr_devinfo_):ChiralSubG(gg, cs, csfull, ptr_devinfo_){}
    virtual void set_chirality_specific_start_range() override final;
    virtual char get_char() override final {return 'A';}
    
};

class BSubG : public ChiralSubG{
  public:
    BSubG() = default;
    BSubG(const Geometry & gg, std::string cs, bool csfull, const openclutil::OpenCLDeviceInfo * ptr_devinfo_):ChiralSubG(gg, cs, csfull, ptr_devinfo_){}
    virtual void set_chirality_specific_start_range() override final;
    virtual char get_char() override final {return 'B';}

};




class Graph{

  private:
    ASubG asubg;
    BSubG bsubg;
    CSubG csubg;

  public:
    const Geometry * ptr_gg;

    std::vector<SubG * > p_subgs;
    
    std::string constraints_string_in;
    
    void force_start_node(std::string);
    
    std::vector<std::pair< std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned> > > coupled_parameters;

    Graph(const Geometry & gg, const openclutil::OpenCLDeviceInfo & devinfo, std::string constraint_string, bool full_constraints_expected);

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
 
public:
  void replace_undefined_randomly();
  void replace(const std::vector<std::vector<unsigned>> & partial_params);
  void replace_where_source_defined(const std::vector<std::vector<unsigned>> & params);
  bool in_graph(unsigned mi, unsigned hpi, unsigned value);
  std::tuple<bool, std::string> in_graph();
   
 
  const XHPs & at(nsHP::eMat subgtype) const {return  v_xhps[subgtype]; }
  XHPs & at(nsHP::eMat subgtype) {return  v_xhps[subgtype]; }
  
  nsHP::eMat get_eMat_from_char(char X) const;
    
  HyperParams(const Graph & graph);
  
  bool operator == (const HyperParams & hpr);
  std::vector<HyperParams> get_one_aways();
  std::string get_part_string(char X) const;
  std::string get_string() const;
  void checks() const;

};  


std::vector< std::vector<unsigned> > get_all_constraints(std::string constraints_string);

}
}

#endif


