#ifndef FFLONK_INFO_HPP
#define FFLONK_INFO_HPP

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include "zkassert.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include <alt_bn128.hpp>

using json = nlohmann::json;

/* FflonkShKey class contains the contents of the file zkevm.fflonkshkey.json,
   which is parsed during the constructor */

class FStagePol
{
public:
    std::string name;
    uint64_t degree;
};

class FStage
{
public:
    uint64_t stage;
    std::vector<FStagePol> pols;
};

class F
{
public:
    uint64_t index;
    uint64_t degree;
    std::vector<std::string> pols;
    std::vector<uint64_t> openingPoints;
    std::vector<FStage> stages;
};

class FflonkShKey
{
    using G1Point = typename AltBn128::Engine::G1Point;
    using G2Point = typename AltBn128::Engine::G2Point;
    using FrElement = typename AltBn128::Engine::FrElement;


    AltBn128::Engine &E;
public:
    uint64_t power;
    uint64_t powerW;
    uint64_t nPublics;
    uint64_t maxQDegree;

    std::map<std::string,FrElement> omegas;

    std::map<uint64_t,std::vector<std::string>> polsNamesStage;

    std::vector<F> f;

    /* Constructor */
    FflonkShKey(AltBn128::Engine &_E, std::string file);

    /* Loads data from a json object */
    void load (json j);
};

#endif
