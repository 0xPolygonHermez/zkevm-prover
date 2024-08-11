#ifndef STARKS_HPP
#define STARKS_HPP

#include <algorithm>
#include <cmath>
#include "config.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "constant_pols_starks.hpp"
#include "const_pols.hpp"
#include "proof_stark.hpp"
#include "fri.hpp"
#include "transcriptGL.hpp"
#include "steps.hpp"
#include "zklog.hpp"
#include "merkleTreeBN128.hpp"
#include "transcriptBN128.hpp"
#include "exit_process.hpp"
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"
#include "hint/h1h2_hint_handler.hpp"
#include "hint/gprod_hint_handler.hpp"
#include "hint/gsum_hint_handler.hpp"

using namespace Hints;

template <typename ElementType>
class Starks
{
public:
    const Config &config;
    StarkInfo &starkInfo;
    CHelpers &chelpers;
    ConstPols<ElementType> &constPols;

    bool debug = false;
    uint32_t nrowsPack = NROWS_PACK;
    
    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

private:
    uint64_t N;
    uint64_t NExtended;

    NTT_Goldilocks ntt;
    NTT_Goldilocks nttExtended;

    MerkleTreeType **treesGL;
    MerkleTreeType **treesFRI;

    vector<bool> subProofValuesCalculated;
    vector<bool> commitsCalculated;

    uint64_t nFieldElements;
    uint64_t merkleTreeArity;
    bool merkleTreeCustom;

    Goldilocks::Element *mem;
    void *pAddress;

    Goldilocks::Element *S;
    Goldilocks::Element *xis;
    Goldilocks::Element *x;

    Goldilocks::Element *zi;

void merkelizeMemory(); // function for DBG purposes
void printPolRoot(uint64_t polId, StepsParams& params); // function for DBG purposes
void printPol(Goldilocks::Element* pol, uint64_t dim);

public:
    Starks(const Config &config, void *_pAddress, StarkInfo &starkInfo_, CHelpers &chelpers_, ConstPols<ElementType> &constPols_, bool debug_) : config(config),
                                                                           starkInfo(starkInfo_),
                                                                           chelpers(chelpers_),
                                                                           constPols(constPols_),
                                                                           N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           nttExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           pAddress(_pAddress)
                                                                           
    {
        debug = debug_;

        // Avoid unnecessary initialization if we are not going to generate any proof
        if (!config.generateProof())
            return;

        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            nFieldElements = 1;
            merkleTreeArity = starkInfo.starkStruct.merkleTreeArity;
            merkleTreeCustom = starkInfo.starkStruct.merkleTreeCustom;
        } else {
            nFieldElements = HASH_SIZE;
            merkleTreeArity = 2;
            merkleTreeCustom = true;
        }

        treesGL = new MerkleTreeType*[starkInfo.nStages + 2];

        TimerStart(COMPUTE_ZHINV);
        zi = new Goldilocks::Element[starkInfo.boundaries.size() * NExtended];

        for(uint64_t i = 0; i < starkInfo.boundaries.size(); ++i) {
            Boundary boundary = starkInfo.boundaries[i];
            if(boundary.name == "everyRow") {
                buildZHInv();
            } else if(boundary.name == "firstRow") {
                buildOneRowZerofierInv(i, 0);
            } else if(boundary.name == "lastRow") {
                buildOneRowZerofierInv(i, N);
            } else if(boundary.name == "everyRow") {
                buildFrameZerofierInv(i, boundary.offsetMin, boundary.offsetMax);
            }
        }
        TimerStopAndLog(COMPUTE_ZHINV);

        TimerStart(COMPUTE_X);
        mem = (Goldilocks::Element *)pAddress;

        uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
        x = new Goldilocks::Element[N << extendBits];
        x[0] = Goldilocks::shift();
        for (uint64_t k = 1; k < (N << extendBits); k++)
        {
            x[k] = x[k - 1] * Goldilocks::w(starkInfo.starkStruct.nBits + extendBits);
        }

        S = new Goldilocks::Element[starkInfo.qDeg];
        xis = new Goldilocks::Element[starkInfo.openingPoints.size() * FIELD_EXTENSION];
        Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
        S[0] = Goldilocks::one();
        for(uint64_t i = 1; i < starkInfo.qDeg; i++) {
            S[i] = Goldilocks::mul(S[i - 1], shiftIn);
        }
        TimerStopAndLog(COMPUTE_X);

        TimerStart(MERKLE_TREE_ALLOCATION);
        for (uint64_t i = 0; i < starkInfo.nStages + 1; i++)
        {
            std::string section = "cm" + to_string(i + 1);
            uint64_t nCols = starkInfo.mapSectionsN[section];
            Goldilocks::Element *pBuffExtended = &mem[starkInfo.mapOffsets[std::make_pair(section, true)]];
            treesGL[i] = new MerkleTreeType(merkleTreeArity, merkleTreeCustom,  NExtended, nCols, pBuffExtended);
        }

        if(debug) {
            treesGL[starkInfo.nStages + 1] = new MerkleTreeType();
        } else {
            treesGL[starkInfo.nStages + 1] = new MerkleTreeType(merkleTreeArity, merkleTreeCustom, (Goldilocks::Element *)constPols.pConstTreeAddress);
            treesFRI = new MerkleTreeType*[starkInfo.starkStruct.steps.size() - 1];
            for(uint64_t step = 0; step < starkInfo.starkStruct.steps.size() - 1; ++step) {
                uint64_t nGroups = 1 << starkInfo.starkStruct.steps[step + 1].nBits;
                uint64_t groupSize = (1 << starkInfo.starkStruct.steps[step].nBits) / nGroups;

                treesFRI[step] = new MerkleTreeType(merkleTreeArity, merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION, NULL);
            }
        }
        TimerStopAndLog(MERKLE_TREE_ALLOCATION);

        commitsCalculated.resize(starkInfo.cmPolsMap.size(), false);
        subProofValuesCalculated.resize(starkInfo.nSubProofValues, false);
    };
    ~Starks()
    {
        if (!config.generateProof())
            return;
        
        delete S;
        delete xis;
        delete x;

        delete zi;

        for (uint i = 0; i < starkInfo.nStages + 2; i++)
        {
            delete treesGL[i];
        }
        delete[] treesGL;

        if(!debug) {
            for (uint64_t i = 0; i < starkInfo.starkStruct.steps.size() - 1; i++)
            {
                delete treesFRI[i];
            }
            delete[] treesFRI;
        }
    };

    void buildZHInv()
    {
        uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
        uint64_t extend = (1 << extendBits);
        
        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();

        for (uint64_t i = 0; i < starkInfo.starkStruct.nBits; i++) Goldilocks::square(sn, sn);

        for (uint64_t i=0; i<extend; i++) {
            Goldilocks::inv(zi[i], (sn * w) - Goldilocks::one());
            Goldilocks::mul(w, w, Goldilocks::w(extendBits));
        }

        #pragma omp parallel for
        for (uint64_t i=extend; i<NExtended; i++) {
            zi[i] = zi[i % extend];
        }
    };

    void buildOneRowZerofierInv(uint64_t offset, uint64_t rowIndex)
    {
        Goldilocks::Element root = Goldilocks::one();

        for(uint64_t i = 0; i < rowIndex; ++i) {
            root = root * Goldilocks::w(starkInfo.starkStruct.nBits);
        }

        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();

        for(uint64_t i = 0; i < NExtended; ++i) {
            Goldilocks::Element x = sn * w;
            Goldilocks::inv(zi[i + offset * NExtended], (x - root) * zi[i]);
            w = w * Goldilocks::w(starkInfo.starkStruct.nBitsExt);
        }
    }

    void buildFrameZerofierInv(uint64_t offset, uint64_t offsetMin, uint64_t offsetMax)
    {
        uint64_t nRoots = offsetMin + offsetMax;
        Goldilocks::Element roots[nRoots];

        for(uint64_t i = 0; i < offsetMin; ++i) {
            roots[i] = Goldilocks::one();
            for(uint64_t j = 0; j < i; ++j) {
                roots[i] = roots[i] * Goldilocks::w(starkInfo.starkStruct.nBits);
            }
        }

        for(uint64_t i = 0; i < offsetMax; ++i) {
            roots[i + offsetMin] = Goldilocks::one();
            for(uint64_t j = 0; j < (N - i - 1); ++j) {
                roots[i + offsetMin] = roots[i + offsetMin] * Goldilocks::w(starkInfo.starkStruct.nBits);
            }
        }

        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();

        for(uint64_t i = 0; i < NExtended; ++i) {
            zi[i + offset*NExtended] = Goldilocks::one();
            Goldilocks::Element x = sn * w;
            for(uint64_t j = 0; j < nRoots; ++j) {
                zi[i + offset*NExtended] = zi[i + offset*NExtended] * (x - roots[j]);
            }
            w = w * Goldilocks::w(starkInfo.starkStruct.nBitsExt);
        }
    }

    void genProof(FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs, CHelpersSteps *chelpersSteps);
    
    void calculateHints(uint64_t step, StepsParams& params, CHelpersSteps *chelpersSteps);

    void extendAndMerkelize(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof);

    void calculateExpression(Goldilocks::Element* dest, uint64_t id, StepsParams &params, CHelpersSteps *chelpersSteps, bool domainExtended, bool imPol);
    void calculateConstraint(uint64_t constraintId, StepsParams &params, CHelpersSteps *chelpersSteps);
    void calculateImPolsExpressions(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps);

    void commitStage(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof);
    void computeStageExpressions(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof, CHelpersSteps *chelpersSteps);
    void computeQ(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof);
    
    void computeEvals(StepsParams& params, FRIProof<ElementType> &proof);

    void computeFRIPol(uint64_t step, StepsParams& params, CHelpersSteps *chelpersSteps);
    
    void computeFRIFolding(FRIProof<ElementType> &fproof, Goldilocks::Element* pol, uint64_t step, Goldilocks::Element *challenge);
    void computeFRIQueries(FRIProof<ElementType> &fproof, uint64_t* friQueries);

    void calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements);

    void addTranscriptGL(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements);
    void getChallenge(TranscriptType &transcript, Goldilocks::Element& challenge);

private:
    bool canExpressionBeCalculated(ParserParams &parserParams);

    bool isHintResolved(Hint &hint, std::vector<string> dstFields);
    bool canHintBeResolved(Hint &hint, std::vector<string> srcFields);

    void evmap(StepsParams &params, Goldilocks::Element *LEv);

    void setCommitCalculated(uint64_t id);
    void setSubproofValueCalculated(uint64_t id);

    uint64_t isStageCalculated(uint64_t step);
    bool isSymbolCalculated(opType operand, uint64_t id);

    void calculateS(Polinomial &s, Polinomial &den, Goldilocks::Element multiplicity);
public:
    void cleanSymbolsCalculated();

    // Following function are created to be used by the ffi interface
    void *ffi_create_steps_params(Goldilocks::Element *pChallenges, Goldilocks::Element* pSubproofValues, Goldilocks::Element *pEvals, Goldilocks::Element *pPublicInputs);
    void ffi_extend_and_merkelize(uint64_t step, StepsParams *params, FRIProof<ElementType> *proof);
    void ffi_treesGL_get_root(uint64_t index, ElementType *dst);

    void *ffi_get_vector_pointer(char *name);
    void ffi_set_commit_calculated(uint64_t id);
    void ffi_set_subproofvalue_calculated(uint64_t id);
};

template class Starks<Goldilocks::Element>;
template class Starks<RawFr::Element>;

#endif // STARKS_H
