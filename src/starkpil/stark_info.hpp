#ifndef STARK_INFO_HPP
#define STARK_INFO_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "zkassert.hpp"
#include "goldilocks_base_field.hpp"
#include "polinomial.hpp"
#include "merklehash_goldilocks.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"

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
    bool hashCommits;
    string verificationHashType;
    uint64_t merkleTreeArity;
    bool merkleTreeCustom;
    vector<StepStruct> steps;
};

opType string2opType (const string s);

class PolMap
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

class EvMap
{
public:
    typedef enum
    {
        cm = 0,
        _const = 1,
    } eType;

    eType type;
    uint64_t id;
    int64_t prime;
    uint64_t openingPos;

    void setType (string s)
    {
        if (s == "cm") type = cm;
        else if (s == "const") type = _const;
        else
        {
            zklog.error("EvMap::setType() found invalid type: " + s);
            exitProcess();
        }
    }
};

class StarkInfo
{
public:
    // Read from starkInfo file
    StarkStruct starkStruct;

    uint64_t subproofId;
    uint64_t airId;

    uint64_t nPublics;
    uint64_t nSubProofValues;
    uint64_t nCommitmentsStage1;
    uint64_t nConstants;
    
    uint64_t nStages;

    vector<PolMap> cmPolsMap;
    vector<PolMap> challengesMap;
    vector<EvMap> evMap;
    
    vector<int64_t> openingPoints;
    vector<Boundary> boundaries;

    uint64_t qDeg;
    uint64_t qDim;

    std::map<std::string, uint64_t> mapSectionsN;

    // Precomputed
    std::map<std::pair<std::string, bool>, uint64_t> mapOffsets;
    std::map<std::string, std::pair<uint64_t, uint64_t>> mapNTTOffsetsHelpers; // <stage, <offset, size>>
    
    std::vector<std::map<uint64_t, uint64_t>> mapOffsetsPolsHints; // <polId, offset>
    std::vector<uint64_t> offsetsExtraMemoryHints; // <hint>

    uint64_t mapTotalN;
 
    /* Constructor */
    StarkInfo(string file);

    /* Loads data from a json object */
    void load (json j);

    void setMapOffsets(std::vector<Hint> &hints);

    /* Returns a polynomial specified by its ID */
    Polinomial getPolinomial(Goldilocks::Element *pAddress, uint64_t idPol, uint64_t deg);

private: 
    void setMemoryPolsHint(uint64_t stage, Hint &hint, std::vector<string> &fields, uint64_t &memoryOffset, uint64_t limitMemoryOffset, uint64_t additionalMemoryOffset);
    bool isHintStage(uint64_t stage, Hint &hint, std::vector<string> &dstFields);
};

#endif