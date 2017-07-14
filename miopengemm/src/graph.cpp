#include <algorithm>
#include <sstream>
#include <miopengemm/architests.hpp>
#include <miopengemm/derivedparams.hpp>
#include <miopengemm/graph.hpp>
#include <miopengemm/macgrid.hpp>
#include <miopengemm/randomutil.hpp>
#include <miopengemm/stringutilbase.hpp>

namespace MIOpenGEMM
{

RandomUtil radutil17;

std::vector<HyPas> Graph::get_neighbors(const HyPas& hp0) const
{

  std::vector<HyPas> neighbors(get_one_aways(hp0));
  radutil17.shuffle(0, neighbors.size(), neighbors);

  auto p_coupled_away = get_p_coupled_away(hp0);
  radutil17.shuffle(0, p_coupled_away.size(), p_coupled_away);
  neighbors.insert(neighbors.end(), p_coupled_away.begin(), p_coupled_away.end());

  auto mic_mac_transformed = get_mic_mac_transformed(hp0);
  radutil17.shuffle(0, mic_mac_transformed.size(), mic_mac_transformed);
  neighbors.insert(neighbors.end(), mic_mac_transformed.begin(), mic_mac_transformed.end());

  return neighbors;
}

std::vector<HyPas> Graph::get_p_coupled_away(const HyPas& hp0) const
{

  std::vector<HyPas> p_coupled_away;

  // by changing two hyper-parameters
  for (auto& coup : p_coupled)
  {

    auto first       = std::get<0>(coup);
    auto first_m     = std::get<0>(first);
    auto first_p     = std::get<1>(first);
    auto first_value = hp0.sus[first_m].vs[first_p];

    auto second       = std::get<1>(coup);
    auto second_m     = std::get<0>(second);
    auto second_p     = std::get<1>(second);
    auto second_value = hp0.sus[second_m].vs[second_p];

    for (auto& new_first_val : at(first_m).edges[first_p].at(first_value))
    {
      for (auto& new_second_val : at(second_m).edges[second_p].at(second_value))
      {

        // only if one increases and one decreases
        if ((new_second_val > second_value) != (new_first_val > first_value))
        {
          HyPas hp1(hp0);
          hp1.sus[first_m].vs[first_p]   = new_first_val;
          hp1.sus[second_m].vs[second_p] = new_second_val;
          p_coupled_away.push_back(hp1);
        }
      }
    }
  }

  return p_coupled_away;
}

// changing MAC and one or both MICs, so as to semi-preserve the overall shape of the macro tile
std::vector<HyPas> Graph::get_mic_mac_transformed(const HyPas& hp0) const
{

  std::vector<HyPas> mmt;

  size_t        curr_mac = hp0.sus[Mat::E::C].vs[NonChi::E::MAC];
  macgrid::Grid curr_grid(curr_mac, hp0.sus[Mat::E::C].vs[NonChi::E::SKW]);

  for (auto& newmac : at(Mat::E::C).edges[NonChi::E::MAC].at(curr_mac))
  {
    macgrid::Grid new_grid(newmac, hp0.sus[Mat::E::C].vs[NonChi::E::SKW]);
    if (!new_grid.is_good)
    {
      continue;
    }

    double delta_na =
      static_cast<double>(new_grid.at(Mat::E::A)) / static_cast<double>(curr_grid.at(Mat::E::A));
    double delta_nb =
      static_cast<double>(new_grid.at(Mat::E::B)) / static_cast<double>(curr_grid.at(Mat::E::B));

    // mica scaled so that the macro tile remains ~ the same in the a dimension
    size_t curr_mica = hp0.sus[Mat::E::A].vs[Chi::E::MIC];
    size_t new_mica  = static_cast<size_t>(static_cast<double>(curr_mica) / delta_na);

    // micb scaled so that the macro tile remains the same in the b dimension
    size_t curr_micb = hp0.sus[Mat::E::B].vs[Chi::E::MIC];
    size_t new_micb  = static_cast<size_t>(static_cast<double>(curr_micb) / delta_nb);
    // if the new micro tile (a) is different and valid, add it
    if (new_mica != curr_mica && contains(Mat::E::A, Chi::E::MIC, new_mica))
    {
      HyPas hp1(hp0);
      hp1.sus[Mat::E::C].vs[NonChi::E::MAC] = newmac;
      hp1.sus[Mat::E::A].vs[Chi::E::MIC]    = new_mica;
      mmt.push_back(hp1);
    }

    if (new_micb != curr_micb && contains(Mat::E::B, Chi::E::MIC, new_micb))
    {
      HyPas hp1(hp0);
      hp1.sus[Mat::E::C].vs[NonChi::E::MAC] = newmac;
      hp1.sus[Mat::E::B].vs[Chi::E::MIC]    = new_micb;
      mmt.push_back(hp1);

      if (new_mica != curr_mica && contains(Mat::E::A, Chi::E::MIC, new_mica))
      {
        HyPas hp2(hp1);
        hp2.sus[Mat::E::A].vs[Chi::E::MIC] = new_mica;
        mmt.push_back(hp2);
      }
    }
  }
  return mmt;
}

std::vector<HyPas> Graph::get_one_aways(const HyPas& hp0) const
{

  std::vector<HyPas> one_aways;

  // the true one aways
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
    {
      size_t v0 = hp0.sus[emat].vs.at(i);
      for (auto& x : at(emat).edges.at(i).at(v0))
      {
        HyPas hp1(hp0);
        hp1.sus[emat].vs[i] = x;
        one_aways.push_back(hp1);
      }
    }
  }

  return one_aways;
}

void SuHy::checks() const
{
  if (vs.size() != Mat::mat_to_xchi(emat)->N)
  {
    throw miog_error("size of vs array of SuHy is not as expected, internal logic error");
  }

  for (const auto& v : vs)
  {
    if (v == Status::E::UNDEFINED)
    {
      throw miog_error("UNDEFINED in vs of SuHy, internal logic error");
    }
  }
}

void HyPas::checks() const
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sus[emat].checks();
  }
}

std::string get_location_string(Mat::E emat, size_t hpi)
{

  if (hpi > Mat::mat_to_xchi(emat)->N)
  {
    throw miog_error("invalid hpi in get_location_string, internal logic error");
  }

  std::stringstream basess;
  basess << " Sub-graph: " << Mat::M.name[emat]
         << ". Hyper-p: " << Mat::mat_to_xchi(emat)->name[hpi];
  return basess.str();
}

void SuGr::checks() const
{

  std::stringstream errm;

  for (unsigned i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {

    std::vector<size_t> eks;
    for (auto& x : edges[i])
    {
      eks.push_back(std::get<0>(x));
    }

    // eks should be (a subset of) union(range, start_range)
    for (auto x : eks)
    {
      if (std::find(start_range[i].begin(), start_range[i].end(), x) == start_range[i].end() &&
          std::find(range[i].begin(), range[i].end(), x) == range[i].end())
      {
        errm << get_location_string(emat, i) << ". Edge key " << x
             << " in in neither range nor edge_range."
             << " Being pedantic and bailing. ";
        errm << get_string(i);
        throw miog_error(errm.str());
      }
    }

    // range must be a subset of edge keys
    for (auto x : range[i])
    {
      if (std::find(eks.begin(), eks.end(), x) == eks.end())
      {
        errm << get_location_string(emat, i) << ". Range value " << x << " is not an edge key.\n"
             << get_string(i);
        throw miog_error(errm.str());
      }
    }

    // start_range must be a subset of edge keys
    for (auto x : start_range[i])
    {
      if (std::find(eks.begin(), eks.end(), x) == eks.end())
      {
        errm << get_location_string(emat, i) << ". Start range value " << x
             << " is not an edge key." << get_string(i);
        throw miog_error(errm.str());
      }
    }

    // ends of all edges must be in range
    for (auto& x : edges[i])
    {
      for (auto& y : std::get<1>(x))
      {
        if (std::find(range[i].begin(), range[i].end(), y) == range[i].end())
        {
          errm << get_location_string(emat, i) << ". Detected dangling edge end,  " << y << '.'
               << get_string(i);
        }
      }
    }
  }
}

void Graph::checks() const
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    at(emat).checks();
  }
}

void SuGr::apply_constraint()
{

  // start range logic :
  // if in range, leave edges.
  // if not in range, fully connect outbound.
  for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {
    if (ptr_constraint->start_range[i] != Status::E::UNDEFINED)
    {

      if (ptr_constraint->range[i] != Status::E::UNDEFINED)
      {
        std::stringstream errm;
        errm << "If a range constraint is provided, no start range constraint is allowed. ";
        errm << "Range string :\n " << ptr_constraint->get_r_str() << '\n';
        errm << "Start range string :\n " << ptr_constraint->get_sr_str() << '\n';
        errm << get_string(i) << "\n";
        throw miog_error(errm.str());
      }
      size_t new_unique_start = ptr_constraint->start_range[i];
      start_range[i]          = {new_unique_start};
      if (std::find(range[i].begin(), range[i].end(), new_unique_start) == range[i].end())
      {
        edges[i][new_unique_start] = range[i];
      }
    }
  }

  // range logic :
  // simply replace with unique value x, edges is {x:{}}.
  // also, it overrides start_range
  for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {
    if (ptr_constraint->range[i] != Status::E::UNDEFINED)
    {
      size_t new_unique = ptr_constraint->range[i];
      range[i]          = {new_unique};
      start_range[i]    = {new_unique};
      edges[i]          = {{{new_unique}, {}}};
    }
  }
}

SuHy SuGr::get_random_start() const
{
  std::vector<size_t> hpvs(Mat::mat_to_xchi(emat)->N);
  for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {
    auto index = radutil17.get_from_range(start_range[i].size());
    hpvs[i]    = start_range[i][index];
  }
  return SuHy(emat, std::move(hpvs));
}

HyPas Graph::get_random_start() const
{
  return HyPas({at(Mat::E::A).get_random_start(),
                at(Mat::E::B).get_random_start(),
                at(Mat::E::C).get_random_start()});
}

HyPas Graph::get_random_valid_start() const
{

  HyPas hp0(get_random_start());

  bool              found = false;
  size_t            iter  = 0;
  std::stringstream ss;

  while (found == false && iter < max_n_iter)
  {
    hp0 = get_random_start();
    hp0.checks();  // TODO WHEN PASSING : this should not be necessary

    Derivabilty dble(hp0, geometry);
    if (!dble.is_derivable)
    {
      ss << '\n' << hp0.get_string() << " isn't deriveable : " << dble.msg;
    }

    else
    {
      auto             dp = DerivedParams(hp0, geometry);
      architests::Stat atr(devinfo, dp, geometry, hp0);

      if (!atr.is_good)
      {
        ss << '\n' << hp0.get_string() << "failed architests : " << atr.msg;
      }
      else
      {
        found = true;
      }
    }
    ++iter;
  }

  // force the graph starting parameters
  if (!found)
  {
    std::stringstream base_ss;
    base_ss << "\nStruggling to find hp satisying geometry, constraints and architecture."
            << " The number of attempts made : " << max_n_iter << '.'
            << " To view the full output of hps tried, "
            << " and reasons for not being derivable, modify the code here -- "
            << " (add ss.str() to this string). Will attempt to obtain generic hp. ";

    throw miog_error(base_ss.str());
  }
  else
  {
    mowri << "#trials to find viable hp in graph : " << iter << Endl;
  }
  return hp0;
}

std::string SuGr::get_string(size_t hpi) const
{
  std::stringstream ss;
  ss << get_edges_string(hpi) << '\n' << get_range_string(hpi) << get_start_range_string(hpi);
  return ss.str();
}

void SuGr::ss_init(size_t hpi, std::stringstream& ss, std::string name) const
{
  if (hpi >= Mat::mat_to_xchi(emat)->N)
  {
    throw miog_error("index too large while obtaining edges string, interal logic error");
  }
  ss << '\n' << stringutil::get_star_wrapped(name) << '\n';
}

std::string SuGr::get_edges_string(size_t hpi) const
{
  std::stringstream ss;
  ss_init(hpi, ss, "EDGES");
  for (auto& x : edges[hpi])
  {
    ss << std::get<0>(x) << " : { ";
    for (auto& y : std::get<1>(x))
    {
      ss << y << ' ';
    }
    ss << "}\n";
  }
  return ss.str();
}

std::string SuGr::get_range_string(size_t hpi) const
{
  std::stringstream ss;
  ss_init(hpi, ss, "RANGE");
  stringutil::add_v_string(ss, range[hpi]);
  return ss.str();
}

std::string SuGr::get_start_range_string(size_t hpi) const
{
  std::stringstream ss;
  ss_init(hpi, ss, "START RANGE");
  stringutil::add_v_string(ss, start_range[hpi]);
  return ss.str();
}

bool SuHy::operator==(const SuHy& rhs) const { return vs == rhs.vs; }

bool HyPas::operator==(const HyPas& rhs) const { return sus == rhs.sus; }

std::string HyPas::get_string() const
{

  std::stringstream ss;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    if (emat != Mat::E::A)
    {
      ss << "__";
    }
    ss << Mat::M.name[emat] << '_' << sus[emat].get_string();
  }
  return ss.str();
}

std::string Constraints::get_combo_str(const str_array& strs) const
{
  std::stringstream ss;
  bool              empty = true;
  for (auto x : strs)
  {
    if (x != "")
    {
      if (!empty)
      {
        ss << "__";
      }
      ss << x;
      empty = false;
    }
  }
  return ss.str();
}

std::string Constraints::get_r_str() const
{
  str_array strs;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    strs[emat] = sub[emat].get_r_str();
  }
  return get_combo_str(strs);
}

std::string Constraints::get_sr_str() const
{
  str_array strs;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    strs[emat] = sub[emat].get_sr_str();
  }
  return get_combo_str(strs);
}

void SuHy::replace_where_defined(const Constraint& constraint)
{
  if (constraint.emat != emat)
  {
    throw miog_error("constraint is not for same subgraph, internal logic error");
  }
  for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {
    if (constraint.range[i] != Status::E::UNDEFINED)
    {
      vs[i] = constraint.range[i];
    }
  }
}

std::string get_str(Mat::E emat, const std::vector<size_t>& vs)
{
  std::stringstream ss;
  bool              isempty = true;
  for (size_t i = 0; i < Mat::mat_to_xchi(emat)->N; ++i)
  {
    if (vs[i] != Status::E::UNDEFINED)
    {
      if (!isempty)
      {
        ss << '_';
      }
      ss << Mat::mat_to_xchi(emat)->name[i] << vs[i];
      isempty = false;
    }
  }
  return ss.str();
}

std::string Constraint::get_r_str() const { return get_str(emat, range); }

std::string Constraint::get_sr_str() const { return get_str(emat, start_range); }

std::string SuHy::get_string() const { return get_str(emat, vs); }

Constraint::Constraint(Mat::E e)
  : emat(e),
    range(Mat::mat_to_xchi(emat)->N, Status::E::UNDEFINED),
    start_range(Mat::mat_to_xchi(emat)->N, Status::E::UNDEFINED)
{
}

Constraint::Constraint(Mat::E e, const std::string& r) : Constraint(e)
{
  range = get_hy_v(r, false, emat);
}

Constraint::Constraint(Mat::E e, const std::string& r, const std::string& sr) : Constraint(e, r)
{
  start_range = get_hy_v(sr, false, emat);
}

// TODO : Constraints and Hyperparams are similar, consider inheritance

Constraints::Constraints(const str_array& cr_strings)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sub[emat] = Constraint(emat, cr_strings[emat]);
  }
}

Constraints::Constraints(const str_array& cr, const str_array& csr)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sub[emat] = Constraint(emat, cr[emat], csr[emat]);
  }
}

// included for deprecation reasons
std::array<std::string, Mat::E::N> get_substrings(const std::string& rconcat)
{

  std::array<std::string, Mat::E::N> substrings;
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    substrings[emat] = "";
  }

  auto              megafrags = stringutil::split(rconcat, "__");
  std::stringstream ss;
  for (auto& megafrag : megafrags)
  {
    if (Mat::M.val.count(megafrag[0]) == 0)
    {
      ss << "\nWhile reading hyperstring in get-params-from-string,\n";
      ss << "the leading char, `" << megafrag[0] << "', was not recognised.\n";
      throw miog_error(ss.str());
    }
    Mat::E emat    = static_cast<Mat::E>(Mat::M.val.at(megafrag[0]));
    size_t minsize = std::string("m__hv").size();

    if (megafrag.size() < minsize)
    {
      ss << "sub constraint " << megafrag << " is too short, something is wrong. \n";
      throw miog_error(ss.str());
    }
    substrings[emat] = megafrag.substr(2);
  }

  return substrings;
}

Constraints::Constraints(const std::string& rconcat) : Constraints(get_substrings(rconcat)) {}

HyPas::HyPas(const std::string& rconcat) : HyPas(get_substrings(rconcat)) {}

std::vector<size_t> get_hy_v(std::string hy_s, bool hy_s_full, Mat::E emat)
{

  auto p_kv = Mat::mat_to_xchi(emat);

  std::vector<size_t> hy_v(p_kv->N, Status::E::UNDEFINED);

  std::vector<std::string> keyvalfrags;
  if (hy_s.compare(""))
  {
    keyvalfrags = stringutil::split(hy_s, "_");
  }

  // MIC, etc
  std::string key;
  // 6, etc
  size_t val;

  auto start = p_kv->name.begin();
  auto end   = p_kv->name.end();
  for (auto& x : keyvalfrags)
  {
    std::tie(key, val) = stringutil::splitnumeric(x);
    if (std::find(start, end, key) == end)
    {
      std::stringstream ss;
      ss << "While processing the constraint string for Sub Graph `" << Mat::M.name[emat] << "', ";
      ss << "the unrecognised key `" + key << "' was not encountered. \n";
      throw miog_error(ss.str());
    }

    size_t keyindex = p_kv->val.at(key);

    if (keyindex >= p_kv->N)
    {
      throw miog_error("keyindex exceeds number of sub graph hyper params, internal logic error ");
    }

    hy_v[keyindex] = val;
  }

  // A special test in the case that constraints
  // are supposed to be comprehensive
  if (hy_s_full == true)
  {
    for (size_t hpi = 0; hpi < p_kv->N; ++hpi)
    {
      if (hy_v[hpi] == Status::E::UNDEFINED)
      {
        std::stringstream ss;
        ss << "While processing the constraints string of SubG `" << Mat::M.name[emat] << "', ";
        ss << "the parameter `" << p_kv->name[hpi]
           << "' appeared to be unset. Values must all be set as "
           << "hy_s_full is true ";
        throw miog_error(ss.str());
      }
    }
  }

  return hy_v;
}

const std::map<size_t, std::vector<size_t>> g_binary = {{Binary::E::NO, {Binary::E::YES}},
                                                        {Binary::E::YES, {Binary::E::NO}}};

Graph::Graph(const Geometry&         gg,
             const oclutil::DevInfo& di,
             const Constraints&      cs,
             owrite::Writer&         mowri_)
  : asubg(gg, cs.sub[Mat::E::A], devinfo),
    bsubg(gg, cs.sub[Mat::E::B], devinfo),
    csubg(gg, cs.sub[Mat::E::C], devinfo),
    geometry(gg),
    devinfo(di),
    constraints(cs),
    mowri(mowri_)
{
  asubg.initialise();
  bsubg.initialise();
  csubg.initialise();

  p_coupled.push_back({{Mat::E::A, Chi::E::MIC}, {Mat::E::B, Chi::E::MIC}});
  p_coupled.push_back({{Mat::E::C, NonChi::E::UFO}, {Mat::E::C, NonChi::E::PUN}});
  p_coupled.push_back({{Mat::E::C, NonChi::E::UNR}, {Mat::E::C, NonChi::E::ICE}});
}

bool Graph::contains(Mat::E emat, size_t hpi, size_t value) const
{
  if (emat >= Mat::E::N)
  {
    throw miog_error("emat not recognised in Graph contains, internal logic error");
  }
  return at(emat).contains(hpi, value);
}

void HyPas::replace_where_defined(const Constraints& constraints)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sus[emat].replace_where_defined(constraints.sub[emat]);
  }
}

bool Graph::contains(const HyPas& hp) const
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    if (!at(emat).contains(hp.sus[emat]))
    {
      return false;
    }
  }
  return true;
}

const SuGr& Graph::at(size_t emat) const
{
  switch (emat)
  {
  case Mat::E::A: return asubg;
  case Mat::E::B: return bsubg;
  case Mat::E::C: return csubg;
  default: throw miog_error("unrecogised Mat::E in p_subgs");
  }
}

void SuGr::initialise_range()
{
  range.resize(edges.size());
  for (size_t hpi = 0; hpi < edges.size(); ++hpi)
  {
    for (auto& x : edges[hpi])
    {
      range[hpi].push_back(x.first);
    }
  }
}

void SuGr::initialise_start_range()
{
  start_range.resize(range.size());
  for (size_t hpi = 0; hpi < range.size(); ++hpi)
  {
    for (auto& x : range[hpi])
    {
      start_range[hpi].push_back(x);
    }
  }
}

void SuGr::initialise()
{
  initialise_edges();  // virtual
  initialise_range();
  initialise_start_range();
  refine_start_range();  // virtual
  checks();
  apply_constraint();
  checks();
}

void ChiSuGr::initialise_edges()
{

  edges[Chi::E::MIC] = {{1, {2, 3}},
                        {2, {1, 3, 4}},
                        {3, {1, 2, 4}},
                        {4, {2, 3, 5, 6}},
                        {5, {2, 4, 6}},
                        {6, {4, 5, 8}},
                        {8, {4, 6}}};

  edges[Chi::E::PAD] = {{0, {1}},
                        {1, {0, 2}},
                        {
                          2, {1},
                        }};

  edges[Chi::E::PLU] = {g_binary};
  edges[Chi::E::LIW] = {g_binary};
  edges[Chi::E::MIW] = {g_binary};

  edges[Chi::E::WOS] = {{Scratch::E::UNUSED, {Scratch::E::COPY, Scratch::E::NFORM}},
                        {Scratch::E::COPY, {Scratch::E::UNUSED, Scratch::E::NFORM}},
                        {Scratch::E::NFORM, {Scratch::E::UNUSED, Scratch::E::COPY}}};
}

void CSuGr::initialise_edges()
{

  edges[NonChi::E::UNR] = {{8, {16}}, {16, {8, 32}}, {32, {16, 64}}, {64, {16, 32}}};
  edges[NonChi::E::NAW] = {{64, {16}}, {16, {64}}};
  edges[NonChi::E::GAL] = {
    {GroupAllocation::E::BYROW, {GroupAllocation::E::BYCOL, GroupAllocation::E::SUCOL}},
    {GroupAllocation::E::BYCOL, {GroupAllocation::E::BYROW, GroupAllocation::E::SUCOL}},
    {GroupAllocation::E::SUCOL, {GroupAllocation::E::BYROW, GroupAllocation::E::BYCOL}}};

  // MAC and SKW

  if (ptr_devinfo->device_name == "unknown_default_constructed")
  {
  }

  else if (ptr_devinfo->wg_atom_size != 64 && ptr_devinfo->wg_atom_size != 32)
  {
    std::stringstream ss;
    ss << "(device_name : " << ptr_devinfo->device_name << ")  "
       << "Setting up the edge search graph in set_preconstraint_edges, and it "
          "seems like the "
          "atomic wg size is neither 32 or 64. Is this correct ?? If so, "
          "consider changing here or "
          "raise an issue";
    throw miog_error(ss.str());
  }

  // very small / thin matrices
  else if (ptr_gg->m * ptr_gg->n < 32 * 32 || ptr_gg->m < 16 || ptr_gg->n < 16)
  {
    edges[NonChi::E::MAC] = {
      {1, {4, 16}}, {4, {1, 16, 64}}, {16, {4, 64}}, {64, {16, 256}}, {256, {64}},
    };

    edges[NonChi::E::SKW] = {

      {7, {8}},
      {8, {7, 9}},
      {9, {8, 10}},
      {10, {9, 11}},
      {11, {10, 12}},
      {12, {11, 13}},
      {13, {12}},

    };
  }

  else if (ptr_devinfo->wg_atom_size == 64)
  {
    edges[NonChi::E::MAC] = {{64, {256}}, {256, {64}}};
    edges[NonChi::E::SKW] = {{9, {10}}, {10, {9, 11}}, {11, {10}}};
  }

  else if (ptr_devinfo->wg_atom_size == 32)
  {
    edges[NonChi::E::MAC] = {{32, {64, 256}}, {64, {32, 128, 256}}, {128, {64, 256}}, {256, {64}}};
    edges[NonChi::E::SKW] = {{9, {10}}, {10, {9, 11}}, {11, {10, 12}}, {12, {10, 11}}};
  }

  else
  {
    throw miog_error("wg_atom_size is neither 32 or 64, how can this be? I "
                     "thought we'd already "
                     "checked this. (Logic error)");
  }

  edges[NonChi::E::ICE] = {{1, {2}},
                           {2, {1, 3, 4}},
                           {3, {1, 2, 4, 6}},
                           {4, {1, 3, 5, 7}},
                           {5, {1, 2, 4, 6, 8}},
                           {6, {1, 3, 5, 7, 9}},
                           {7, {4, 6, 8, 10}},
                           {8, {1, 5, 7, 9, 11}},
                           {9, {6, 8, 10, 12}},
                           {10, {1, 7, 9, 11, 13}},
                           {11, {8, 10, 12, 14}},
                           {12, {1, 9, 11, 13, 14}},
                           {13, {10, 12, 14}},
                           {14, {1, 11, 13}}};

  edges[NonChi::E::PUN] = {g_binary};
  edges[NonChi::E::UFO] = {g_binary};
}

void ChiSuGr::refine_start_range()
{
  start_range[Chi::E::PAD] = {1, 2};
  start_range[Chi::E::LIW] = {Binary::E::NO};
  start_range[Chi::E::MIW] = {Binary::E::YES};
  start_range[Chi::E::WOS] = {Scratch::E::UNUSED, Scratch::E::COPY, Scratch::E::NFORM};

  set_start_mic();
  // refine_start_range_chirality_specific();
}

void CSuGr::refine_start_range()
{

  start_range[NonChi::E::UNR] = {8, 16};
  start_range[NonChi::E::ICE] = {1};
  start_range[NonChi::E::UFO] = {Binary::E::NO};

  if ((ptr_gg->m) > 200 && (ptr_gg->n) > 200)
  {

    if (ptr_devinfo->wg_atom_size == 32)
    {
      start_range[NonChi::E::SKW] = {macgrid::skew0, macgrid::skew0 + 1};
    }

    else
    {
      start_range[NonChi::E::SKW] = {macgrid::skew0};
    }
  }
}

void ChiSuGr::set_start_mic()
{

  size_t non_unroll_dimension = ptr_gg->get_non_k_dim(emat);

  std::vector<size_t> basemic = {8, 6};
  if (non_unroll_dimension < 256)
  {
    basemic.push_back(5);
    basemic.push_back(4);
  }

  if (non_unroll_dimension < 128)
  {
    basemic.push_back(3);
    basemic.push_back(2);
  }

  if (non_unroll_dimension < 64)
  {
    basemic.push_back(1);
  }

  start_range[Chi::E::MIC] = {};
  for (auto& x : basemic)
  {
    if (x <= non_unroll_dimension)
    {
      start_range[Chi::E::MIC].push_back(x);
    }
  }
}

SuGr::SuGr(Mat::E e, const Geometry& gg, const Constraint& ct, const oclutil::DevInfo& di)
  : emat(e),
    ptr_gg(&gg),
    ptr_constraint(&ct),
    ptr_devinfo(&di),
    edges(Mat::mat_to_xchi(emat)->N),
    range(Mat::mat_to_xchi(emat)->N),
    start_range(Mat::mat_to_xchi(emat)->N)
{
}

CSuGr::CSuGr(const Geometry& gg, const Constraint& ct, const oclutil::DevInfo& di)
  : SuGr(Mat::E::C, gg, ct, di)
{
}

ChiSuGr::ChiSuGr(Mat::E e, const Geometry& gg, const Constraint& ct, const oclutil::DevInfo& di)
  : SuGr(e, gg, ct, di)
{
}

ASuGr::ASuGr(const Geometry& gg, const Constraint& constraint, const oclutil::DevInfo& devinfo)
  : ChiSuGr(Mat::E::A, gg, constraint, devinfo)
{
}

BSuGr::BSuGr(const Geometry& gg, const Constraint& constraint, const oclutil::DevInfo& devinfo)
  : ChiSuGr(Mat::E::B, gg, constraint, devinfo)
{
}

SuHy::SuHy(Mat::E e) : emat(e), vs(Mat::mat_to_xchi(emat)->N, Status::E::UNDEFINED) {}

SuHy::SuHy(Mat::E e, const std::string& hyperstring) : SuHy(e)
{
  vs = get_hy_v(hyperstring, true, emat);
}

SuHy::SuHy(Mat::E e, std::vector<size_t>&& vs_) : emat(e), vs(vs_) {}

HyPas::HyPas(const str_array& hyperstrings)
{
  for (auto emat : {Mat::E::A, Mat::E::B, Mat::E::C})
  {
    sus[emat] = SuHy(emat, hyperstrings[emat]);
  }
}

HyPas::HyPas(std::array<SuHy, Mat::E::N>&& suhys) : sus(suhys) {}

bool SuGr::contains(size_t hpi, size_t val) const
{
  if (range.size() >= hpi)
  {
    throw miog_error("in SuGr::contains, range size smaller than hpi, internal logic err");
  }
  bool x =
    (std::find(range[hpi].begin(), range[hpi].end(), val) == range[hpi].end()) ? false : true;
  return x;
}

bool SuGr::contains(const SuHy& suhy) const
{
  for (size_t hpi = 0; hpi < Mat::mat_to_xchi(emat)->N; ++hpi)
  {
    if (!contains(hpi, suhy.vs[hpi]))
    {
      return false;
    }
  }
  return true;
}
}
