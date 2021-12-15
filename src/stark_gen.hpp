#ifndef STARK_GEN_HPP
#define STARK_GEN_HPP

#include <nlohmann/json.hpp>
#include "pols.hpp"

using namespace std;
using json = nlohmann::json;

class StarkGen
{
public:
    void generate (Pols &cmPols, Pols &constPols, Pols &constTree, json &pil);
};

#endif