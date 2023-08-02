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

namespace FflonkInfo {
    
    /* FflonkInfo class contains the contents of the file zkevm.fflonkinfo.json,
    which is parsed during the constructor */
    class StepStruct
    {
    public:
        uint64_t nBits;
    };

    typedef enum
    {
        cm1_n = 0,
        cm1_2ns = 1,
        cm2_n = 2,
        cm2_2ns = 3,
        cm3_n = 4,
        cm3_2ns = 5,
        tmpExp_n = 6,
        q_2ns = 7,
        eSectionMax = 8
    } eSection;

    eSection string2section (const std::string s);

    class PolsSections
    {
    public:
        uint64_t section[eSectionMax];
    };

    class PolsSectionsVector
    {
    public:
        std::vector<uint64_t> section[eSectionMax];
    };

    class VarPolMap
    {
    public:
        eSection section;
        uint64_t dim;
        uint64_t sectionPos;
    };

    class PolInfo {
    public:
        std::string sectionName;
        u_int64_t id    ;
        u_int64_t nPols;
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

    class Publics
    {
    public:
        std::string polType;
        uint64_t polId;
        uint64_t idx;
        uint64_t id;
        std::string name; 
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

        void setType (std::string s)
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

    class StepType
    {
    public:
        typedef enum
        {
            tmp = 0,
            exp = 1,
            eval = 2,
            challenge = 3,
            number = 4,
            x = 5,
            Z = 6,
            _public = 7,
            cm = 8,
            _const = 9,
            q = 10,
            tmpExp = 11,
        } eType;

        eType type;
        uint64_t id;
        bool prime;
        uint64_t p;
        std::string value;

        void setType (std::string s)
        {
            if (s == "tmp") type = tmp;
            else if (s == "exp") type = exp;
            else if (s == "eval") type = eval;
            else if (s == "challenge") type = challenge;
            else if (s == "number") type = number;
            else if (s == "x") type = x;
            else if (s == "Z") type = Z;
            else if (s == "public") type = _public;
            else if (s == "cm") type = cm;
            else if (s == "const") type = _const;
            else if (s == "q") type = q;
            else if (s == "tmpExp") type = tmpExp;
            else
            {
                zklog.error("StepType::setType() found invalid type: " + s);
                exitProcess();
            }
        }
    };

    class StepOperation
    {
    public:
        typedef enum
        {
            add = 0,
            sub = 1,
            mul = 2,
            copy = 3
        } eOperation;

        eOperation op;
        StepType dest;
        std::vector<StepType> src;

        void setOperation (std::string s)
        {
            if (s == "add") op = add;
            else if (s == "sub") op = sub;
            else if (s == "mul") op = mul;
            else if (s == "copy") op = copy;
            else
            {
                zklog.error("StepOperation::setOperation() found invalid type: " + s);
                exitProcess();
            }
        }
    };

    class Step
    {
    public:
        std::vector<StepOperation> first;
        std::vector<StepOperation> i;
        std::vector<StepOperation> last;
        uint64_t tmpUsed;
    };

    class FflonkInfo
    {
        AltBn128::Engine &E;
    public:
        uint64_t nConstants;
        uint64_t nPublics;
        uint64_t nBitsZK;
        uint64_t nCm1;
        uint64_t nCm2;
        uint64_t nCm3;
        uint64_t qDeg;
        uint64_t qDim;

        uint64_t maxPolsOpenings;

        PolsSections mapDeg;
        PolsSectionsVector mapSections;
        PolsSections mapSectionsN;
        std::vector<VarPolMap> varPolMap;
        std::vector<uint64_t> cm_n;
        std::vector<uint64_t> cm_2ns;
        std::vector<PeCtx> peCtx;
        std::vector<PuCtx> puCtx;
        std::vector<CiCtx> ciCtx;
        std::vector<EvMap> evMap;
        std::vector<Step> publicsCode;
        std::vector<Publics> publics;
        Step step2prev;
        Step step3prev;
        Step step3;
        Step step42ns;
        std::vector<uint64_t> exps_n;
        std::vector<uint64_t> q_2nsVector;
        std::vector<uint64_t> tmpExp_n;
        std::map<std::string,uint64_t> exp2pol;
        
        /* Constructor */
        FflonkInfo(AltBn128::Engine &_E, std::string file);

        /* Loads data from a json object */
        void load (json j);

        std::string getSectionName(eSection section);

        PolInfo getPolInfo(u_int64_t polId);
    };

    #endif
}