#ifndef STARK_INFO_HPP
#define STARK_INFO_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "config.hpp"

using json = nlohmann::json;
using namespace std;

class Step
{
public:
    uint64_t nBits;
};

class StarkStruct
{
public:
  uint64_t nBits;
  uint64_t nBitsExt;
  uint64_t nQueries;
  string verificationHashType;
  vector<Step> steps;
};

class PolsSections
{
public:
  uint64_t cm1_n;
  uint64_t cm2_n;
  uint64_t cm3_n;
  uint64_t exps_withq_n;
  uint64_t exps_withoutq_n;
  uint64_t cm1_2ns;
  uint64_t cm2_2ns;
  uint64_t cm3_2ns;
  uint64_t q_2ns;
  uint64_t exps_withq_2ns;
  uint64_t exps_withoutq_2ns;
};

class StarkInfo
{
    const Config &config;
public:
    StarkStruct starkStruct;
    uint64_t mapTotalN;
    uint64_t nConstants;
    uint64_t nCm1;

    PolsSections mapDeg;
    PolsSections mapOffsets;
    PolsSections mapSectionsN;
    PolsSections mapSectionsN1;
    PolsSections mapSectionsN3;

    StarkInfo(const Config &config);
    void load (json j);
};

#endif