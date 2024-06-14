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

using json = nlohmann::json;
using namespace std;

/* StarkInfo class contains the contents of the file zkevm.starkinfo.json,
   which is parsed during the constructor */

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
    xDivXSubXi_2ns = 11,
    LEv = 12,
    evals = 13,
    cm1_2ns_tmp = 14,
    cm2_2ns_tmp = 15,
    eSectionMax = 16
} eSection;

eSection string2section (const string s);

class PolsSections
{
public:
    uint64_t section[eSectionMax];
};

class VarPolMap
{
public:
    eSection section;
    uint64_t dim;
    uint64_t sectionPos;
};

class PolInfo
{
public:
    VarPolMap map;
    uint64_t N;
    uint64_t offset;
    uint64_t size;
    Goldilocks::Element * pAddress;
    Goldilocks::Element * get(uint64_t step)
    {
        zkassert(map.dim==1);
        return pAddress + step*size;
    }
    Goldilocks::Element * get1(uint64_t step)
    {
        zkassert(map.dim==3);
        return pAddress + step*size;
    }
    Goldilocks::Element * get2(uint64_t step)
    {
        zkassert(map.dim==3);
        return pAddress + step*size + 1;
    }
    Goldilocks::Element * get3(uint64_t step)
    {
        zkassert(map.dim==3);
        return pAddress + step*size + 2;
    }
};

class PeCtx
{
public:
    uint64_t tExpId;
    uint64_t fExpId;
    uint64_t zId;
    uint64_t c1Id;
    uint64_t numId;
    uint64_t denId;
    uint64_t c2Id;
};

class PuCtx
{
public:
    uint64_t tExpId;
    uint64_t fExpId;
    uint64_t h1Id;
    uint64_t h2Id;
    uint64_t zId;
    uint64_t c1Id;
    uint64_t numId;
    uint64_t denId;
    uint64_t c2Id;
};

class CiCtx
{
public:
    uint64_t zId;
    uint64_t numId;
    uint64_t denId;
    uint64_t c1Id;
    uint64_t c2Id;
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
public:
    bool reduceMemory;

    StarkStruct starkStruct;

    uint64_t nConstants;
    uint64_t nPublics;
    uint64_t nCm1;
    uint64_t nCm2;
    uint64_t nCm3;
    uint64_t nCm4;
    uint64_t qDeg;
    uint64_t qDim;
    uint64_t friExpId;

    uint64_t merkleTreeArity;

    PolsSections mapDeg;
    PolsSections mapSectionsN;
    PolsSections mapOffsets;
    vector<VarPolMap> varPolMap;
    vector<uint64_t> qs;
    vector<uint64_t> cm_n;
    vector<uint64_t> cm_2ns;
    vector<PeCtx> peCtx;
    vector<PuCtx> puCtx;
    vector<CiCtx> ciCtx;
    vector<EvMap> evMap;
    map<string,uint64_t> exp2pol;

    // Precomputed
    std::map<std::string, std::pair<uint64_t, uint64_t>> mapNTTOffsetsHelpers; // <stage, <offset, size>>
    
    std::map<uint64_t, uint64_t> mapOffsetsPolsH1H2;
    std::map<uint64_t, uint64_t> mapOffsetsPolsGrandProduct;

    std::vector<uint64_t> offsetsExtraMemoryH1H2;

    uint64_t mapTotalN;
    
    void setMapOffsets();

    /* Constructor */
    StarkInfo(string file, bool reduceMemory = false);

    /* Loads data from a json object */
    void load (json j);

    /* Returns information about a polynomial specified by its ID */
    void getPol(void * pAddress, uint64_t idPol, PolInfo &polInfo);

    /* Returns the size of a polynomial specified by its ID */
    uint64_t getPolSize(uint64_t polId);

    /* Returns a polynomial specified by its ID */
    Polinomial getPolinomial(Goldilocks::Element *pAddress, uint64_t idPol);

    void setMemoryPol(uint64_t stage, uint64_t polId, uint64_t &memoryOffset, uint64_t limitMemoryOffset, uint64_t additionalMemoryOffset);
    
    /* Returns the size of the constant tree data/file */
    uint64_t getConstTreeSizeInBytes (void) const
    {
        uint64_t NExtended = 1 << starkStruct.nBitsExt;
        uint64_t constTreeSize = nConstants * NExtended + NExtended * HASH_SIZE + (NExtended - 1) * HASH_SIZE + MERKLEHASHGOLDILOCKS_HEADER_SIZE;
        uint64_t constTreeSizeBytes = constTreeSize * sizeof(Goldilocks::Element);
        return constTreeSizeBytes;
    }
};

#endif