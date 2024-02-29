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
    const_ = 0,
    cm = 1,
    tmp = 2,
    public_ = 3,
    subproofvalue = 4,
    challenge = 5,
    number = 6,
} opType;

opType string2opType (const string s);

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
    publicValue = 2,
} hintType;

hintType string2hintType (const string s);

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

class Symbol 
{
public:
    opType op;
    uint64_t stage;
    uint64_t stageId;
    uint64_t id;
    uint64_t value;

    void setSymbol(json j) {
        op = string2opType(j["op"]);
        if(j.contains("stage")) stage = j["stage"];
        if(j.contains("stageId")) stageId = j["stageId"];
        if(j.contains("id")) id = j["id"];
        if(j.contains("value")) value = j["value"];
    };

    void setSymbol(uint64_t stage_, uint64_t id_) {
        op = string2opType("cm");
        stage = stage_;
        id = id_;
    };
};

class ExpressionCodeSymbol 
{
public:
    uint64_t stage;
    uint64_t expId;
    vector<Symbol> symbolsUsed;
};

class Hint 
{
public:
    hintType type;
    std::vector<string> fields;
    std::map<std::string, Symbol> fieldSymbols;
    std::vector<Symbol> destSymbols;
    std::vector<Symbol> symbols;
    uint64_t index;
};

class VarPolMap
{
public:
    eSection section;
    uint64_t dim;
    uint64_t sectionPos;
    uint64_t deg;
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
    int64_t prime;

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

private:
    Symbol setSymbol(json j);
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
    
    vector<uint64_t> stageChallengeIndex;
    uint64_t qChallengeIndex;
    uint64_t xiChallengeIndex;
    uint64_t fri1ChallengeIndex;
    uint64_t fri2ChallengeIndex;

    uint64_t nChallenges;
    uint64_t nSubProofValues;

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
    vector<vector<Symbol>> symbolsStage;
    vector<vector<Symbol>> stageCodeSymbols;
    vector<vector<ExpressionCodeSymbol>> expressionsCodeSymbols;
    
    std::map<uint64_t, vector<Hint>> hints;

    // pil-stark specific
    vector<VarPolMap> varPolMap;
    vector<uint64_t> cm_n;
    vector<uint64_t> cm_2ns;

    vector<EvMap> evMap;


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