#ifndef STARK_INFO_HPP
#define STARK_INFO_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "config.hpp"
#include "zkassert.hpp"
#include "goldilocks_base_field.hpp"
#include "polinomial.hpp"
#include "merklehash_goldilocks.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

using json = nlohmann::json;
using namespace std;

/* StarkInfo class contains the contents of the file zkevm.starkinfo.json,
   which is parsed during the constructor */

class Boundary
{
public:
    std::string name;
    uint64_t offsetMin;
    uint64_t offsetMax;
};

class StepStruct
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
    vector<StepStruct> steps;
};

typedef enum
{
    cm1_n = 0,
    cm1_2ns = 1,
    cm2_n = 2,
    cm2_2ns = 3,
    cm3_n = 4,
    cm3_2ns = 5,
    cm4_n = 6,
    cm4_2ns = 7,
    tmpExp_n = 8,
    q_2ns = 9,
    f_2ns = 10,
    eSectionMax = 11
} eSection;

eSection string2section (const string s);

typedef enum {
    h1h2 = 0,
    gprod = 1,
} hintType;

class PolsSections
{
public:
    uint64_t section[eSectionMax];
};

class CmPolMap
{
public:
    std::string stage;
    uint64_t stageNum;
    std::string name;
    uint64_t dim;
    bool imPol;
    uint64_t stagePos;
    uint64_t stageId;
};


class VarPolMap
{
public:
    eSection section;
    uint64_t dim;
    uint64_t sectionPos;
    uint64_t deg;
};

class Hint
{
public:
    hintType type;
    vector<string> fields;
    map<string, uint64_t> fieldId;
    vector<string> dests;
    map<string, uint64_t> destId;
    uint64_t index;
};

class EvMap
{
public:
    typedef enum
    {
        cm = 0,
        _const = 1,
        q = 2
    } eType;

    eType type;
    uint64_t id;
    bool prime;

    void setType (string s)
    {
        if (s == "cm") type = cm;
        else if (s == "const") type = _const;
        else if (s == "q") type = q;
        else
        {
            zklog.error("EvMap::setType() found invalid type: " + s);
            exitProcess();
        }
    }
};

class StarkInfo
{
    const Config &config;
public:
    StarkStruct starkStruct;

    bool pil2;
    uint64_t subproofId;
    uint64_t airId;

    uint64_t nPublics;
    uint64_t nConstants;
    uint64_t nCm1;

    uint64_t nStages;

    vector<uint64_t> numChallenges;

    uint64_t nChallenges;

    vector<uint64_t> openingPoints;
    
    vector<Boundary> boundaries;

    uint64_t qDeg;
    uint64_t qDim;
    vector<uint64_t> qs;

    uint64_t mapTotalN;
    PolsSections mapSectionsN;
    PolsSections mapOffsets;

    // pil2-stark-js specific
    vector<CmPolMap> cmPolsMap;

    // pil-stark specific
    vector<VarPolMap> varPolMap;
    vector<uint64_t> cm_n;
    vector<uint64_t> cm_2ns;

    vector<EvMap> evMap;

    std::map<uint64_t, vector<Hint>> hints;

    map<uint64_t,uint64_t> exp2pol;
    
    /* Constructor */
    StarkInfo(const Config &config, string file);

    /* Loads data from a json object */
    void load (json j);

    uint64_t getPolinomialRef(std::string type, uint64_t index);

    /* Returns a polynomial specified by its ID */
    Polinomial getPolinomial(Goldilocks::Element *pAddress, uint64_t idPol, uint64_t deg);
};

#endif