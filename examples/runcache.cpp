/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <chrono>
#include <string>
#include <thread>
#include <miopengemm/geometries.hpp>
#include <miopengemm/kernelcache.hpp>
#include <miopengemm/kernelcachemerge.hpp>
#include <miopengemm/redirection.hpp>
#include <miopengemm/setabcw.hpp>
#include <miopengemm/tinytwo.hpp>

template <typename TFl>
int runcache_v2(std::map<char, std::vector<std::string>>& filters)
{

  using namespace MIOpenGEMM;

  Offsets offsets      = get_zero_offsets();
  auto&&  kernel_cache = get_kernel_cache();
  auto    cache_keys   = kernel_cache.get_keys();

  // Set device and plaform
  CLHint devhint;
  if (filters['P'].size() != 1 || filters['D'].size() != 1)
  {
    throw miog_error("P and D (platform and device) should be one each (or neither)");
  }

  if (filters['P'][0] == "-1" and filters['D'][0] != "-1")
  {
    throw miog_error("D appears to be set, but P not. Set P (probably to 0)");
  }

  if (filters['P'][0] != "-1" and filters['D'][0] == "-1")
  {
    throw miog_error("P appears to be set, but D not. Set D (maybe to 0)");
  }

  if (filters['P'][0] != "-1" and filters['D'][0] != "-1")
  {
    size_t p_id = std::stoi(filters['P'][0]);
    size_t d_id = std::stoi(filters['D'][0]);
    std::cout << "Selected platform : " << p_id << " and selected device : " << d_id << std::endl;
    devhint = CLHint(p_id, d_id);
  }
  else
  {
    std::cout << "-P and -D not passed, will try to guess platform and device" << std::endl;
  }

  if (std::count(filters['d'].begin(), filters['d'].end(), "a") == 0)
  {
    std::vector<std::string> dev_filters;
    for (auto& x : filters['d'])
    {
      if (x == "d")
      {
        owrite::Writer   silent_mowri(Ver::E::TERMINAL, "");
        oclutil::DevInfo devinfo(devhint, silent_mowri);
        dev_filters.push_back(devinfo.device_name);
      }
      else
      {
        dev_filters.push_back(x);
      }
    }
    std::cout << "Will filter cache for devices:   {";
    for (auto& x : dev_filters)
    {
      std::cout << "  " << x << "  ";
    }
    std::cout << "}\n";
    filter_device(cache_keys, dev_filters);
  }
  else
  {
    std::cout << "Will consider all devices in cache\n";
  }

  if (std::count(filters['g'].begin(), filters['g'].end(), "a") == 0)
  {

    std::vector<std::string> geometries_to_filter;

    bool                  filter_db = false;
    std::vector<Geometry> dbgs;

    if (std::count(filters['g'].begin(), filters['g'].end(), "d") != 0)
    {
      {
        dbgs      = get_deepbench(0);
        filter_db = true;
        geometries_to_filter.push_back("deepbench");
      }
    }

    bool filter_square = false;
    if (std::count(filters['g'].begin(), filters['g'].end(), "s") != 0)
    {
      filter_square = true;
      geometries_to_filter.push_back("square");
    }

    std::cout << "Will filter cache for geometries:  {";
    for (auto& x : geometries_to_filter)
    {
      std::cout << "  " << x << "  ";
    }
    std::cout << "}\n";

    if (filters['m'].size() != 1)
    {
      throw miog_error("m flag should be ONE of 0,+,a.");
    }

    auto ws = filters['m'][0];

    std::cout << "Will filter geometry memories:  {";
    std::cout << "  " << ws << "  ";
    std::cout << "}\n";

    auto                  all_geometries = get_geometries(cache_keys);
    std::vector<Geometry> filtered_geometries;
    for (auto& x : all_geometries)
    {

      auto x2        = x;
      x2.wSpaceSize  = 0;
      auto same_geom = [&x2](const Geometry& gg) { return gg == x2; };

      if (filter_square && x.m == x.n && x.m == x.k)
      {
        filtered_geometries.push_back(x);
      }
      else
      {
        if (ws == "0" && filter_db && std::find(dbgs.begin(), dbgs.end(), x) != dbgs.end())
        {
          filtered_geometries.push_back(x);
        }
        else if ((ws == "+" || ws == "a") &&
                 std::find_if(dbgs.begin(), dbgs.end(), same_geom) != dbgs.end())
        {
          if (ws == "+" && x.wSpaceSize > 0)
          {
            filtered_geometries.push_back(x);
          }
          if (ws == "a")
          {
            filtered_geometries.push_back(x);
          }
        }
        else if (ws != "0" && ws != "a" && ws != "+")
        {
          throw miog_error("unrecognised m flag, it should be one of 0,+,a.");
        }
      }
    }
    filter_geometries(cache_keys, filtered_geometries);
  }
  else
  {
    std::cout << "Will consider all geometry families in cache\n";
  }

  // filtering transpose types
  if (std::count(filters['t'].begin(), filters['t'].end(), "a") == 0)
  {
    std::stringstream ss;
    ss << "will filter transpose cases : { ";
    std::vector<std::array<bool, 2>> v_tt;
    for (auto& x : filters['t'])
    {
      if (x.size() == 2 && (x[0] == 'N' || x[0] == 'T') && (x[1] == 'N' || x[1] == 'T'))
      {
        bool aT = x[0] == 'N' ? false : true;
        bool bT = x[1] == 'N' ? false : true;
        v_tt.push_back({{aT, bT}});
      }
      ss << ' ' << x << ' ';
    }
    ss << " }\n";
    std::cout << ss.str();

    auto                  all_geometries = get_geometries(cache_keys);
    std::vector<Geometry> transpose_correct;
    for (auto& gg : all_geometries)
    {
      for (auto& x : v_tt)
      {
        if (gg.tX[Mat::E::A] == x[0] && gg.tX[Mat::E::B] == x[1])
        {
          transpose_correct.push_back(gg);
          break;
        }
      }
    }
    filter_geometries(cache_keys, transpose_correct);
  }

  else
  {
    std::cout << "Will not filter transposes in cache\n";
  }

  if (cache_keys.size() == 0)
  {
    std::cout << "No cache keys remain after filtering. \n";
    std::cout << "Note that all device keys in the cache are  {";
    for (auto& x : get_devices(kernel_cache.get_keys()))
    {
      std::cout << "  " << x << "  ";
    }
    std::cout << "}\n";
    return -1;
  }

  filter_floattype(cache_keys, sizeof(TFl));
  std::cout << "generating random matrices on CPU ... " << std::flush;
  setabcw::CpuMemBundle<TFl> cmb(get_geometries(cache_keys), offsets);
  std::cout << "done.\n" << std::endl;

  if (std::count(filters['w'].begin(), filters['w'].end(), "a") != 0)
  {
    std::cout << "\nPerforming accuracy tests\n\n";
    owrite::Writer mowri(Ver::E::TERMINAL, "");
    for (size_t i = 0; i < cache_keys.size(); ++i)
    {
      auto ck = cache_keys[i];
      if (ck.gg.floattype == 'f')
      {
        dev::TinyOne<TFl> diva(ck.gg, offsets, cmb.r_mem, mowri, devhint);
        diva.accuracy_test(kernel_cache.at(ck, redirection::get_is_not_canonical(ck.gg)));
        mowri << "\n\n";
      }
    }
  }

  if (std::count(filters['w'].begin(), filters['w'].end(), "b") != 0)
  {
    std::cout << "\nBenchmarking\n\n";
    owrite::Writer mowri(Ver::E::MULTIBENCH, "");
    for (size_t i = 0; i < cache_keys.size(); ++i)
    {
      auto ck = cache_keys[i];
      if (ck.gg.floattype == 'f')
      {
        dev::TinyOne<TFl> diva(ck.gg, offsets, cmb.r_mem, mowri, devhint);
        std::string       prefix = std::to_string(i) + "/" + std::to_string(cache_keys.size());
        prefix.resize(8, ' ');
        std::cout << prefix << " ";
        Timer timer;
        timer.start();
        diva.benchgemm({kernel_cache.at(ck, redirection::get_is_not_canonical(ck.gg))},
                       {{{0, 5}}, {{0, 0.08}}});
        auto x = timer.get_elapsed();
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<size_t>(1200. * x)));
      }
    }
  }

  return 0;
}

int main(int argc, char* argv[])
{
  std::map<char, std::string> args = {
    {'g', "geometries to look for in cache"},
    {'d', "devices to look for in cache (not necessarily this device)"},
    {'w', "what to do"},
    {'m', "workspace memories to consider"},
    {'t', "transpose cases (tA, tB) to consider"},
    {'D', "ID of the device to run on"},
    {'P', "ID of the platform to run on"},

  };

  std::map<char, std::vector<std::string>> options = {
    {'g', {"a (all)", "d (deepbench)", "s (m=n=k)"}},
    {'d', {"a (all)", "d (default)", "any other string"}},
    {'w', {"b (benchmark)", "a (accuracy)"}},
    {'m', {"a (all)", "0 (only zero ws)", "+ (only positive ws"}},
    {'t', {"a (all)", "NN", "NT", "TN", "TT"}},
    {'D', {"a non-negative integer"}},
    {'P', {"a non-negative integer"}},
  };

  std::map<char, std::string> defaults = {
    {'g', "d"}, {'d', "d"}, {'w', "b"}, {'m', "0"}, {'t', "a"}, {'D', "-1"}, {'P', "-1"}};

  // TODO: check here that above maps have same keys.

  std::stringstream hss;
  hss << "\n\n ";
  std::map<char, std::vector<std::string>> filters;
  for (auto& x : args)
  {
    hss << '-' << x.first << " : " << x.second << '\n';
    hss << "options : ";
    for (auto& o : options[x.first])
    {
      hss << " " << o << " ";
    }
    hss << "\ndefault :  " << defaults[x.first] << "\n\n";
    filters[x.first] = {};
  }
  hss << "\nexamples:\n";
  hss << "`runcache -g d -d a -w b`\n";
  hss << " benchmarks all cache entries with a deepbench geometry with zero ms \n";
  hss << "`runcache -g a -d gfx803`\n";
  hss << "`runcache -g a -d gfx803 -P 0 -D 0`\n";
  hss << " as above, on device 0 of platform 0 \n";
  hss << " benchmarks of all cache entries which match device gfx803 and have zero ms \n";
  hss << "`runcache -w b a -m a`\n";
  hss << " benchmarks and accuracy test of all cache entries \n";

  std::string help = hss.str();

  using namespace MIOpenGEMM;

  std::vector<std::string> parsed;
  for (int i = 1; i < argc; ++i)
  {
    parsed.push_back(argv[i]);
  }

  for (auto& x : parsed)
  {
    if (x == "-h" || x == "--help" || x == "-help")
    {
      std::cout << help << std::endl;
      return 0;
    }
  }

  if (argc % 2 != 1)
  {
    throw miog_error("Odd number of keys+vals is incorrect.\n" + help);
  }

  char key = '?';
  for (auto& x : parsed)
  {
    if (x.size() == 2 && x.compare(0, 1, "-") == 0)
    {
      key = x[1];
    }
    else
    {
      if (filters.count(key) == 0)
      {
        std::stringstream errm;
        errm << "Unrecognised key\n" << help;
        throw miog_error(errm.str());
      }
      else
      {
        filters[key].push_back(x);
      }
    }
  }

  for (auto x : defaults)
  {
    if (filters[x.first].size() == 0)
    {
      filters[x.first] = {x.second};
    }
  }

  return runcache_v2<float>(filters);
}
