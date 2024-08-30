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
#include "expressions_bin.hpp"
#include "expressions_avx.hpp"
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

    SetupCtx& setupCtx;
    ExpressionsCtx &expressionsCtx;

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

    uint64_t nFieldElements;

    Goldilocks::Element *S;
    Goldilocks::Element *xis;
    Goldilocks::Element *x;

void merkelizeMemory(Goldilocks::Element *pAddress); // function for DBG purposes

public:
    Starks(const Config &config, SetupCtx& setupCtx_, ExpressionsCtx& expressionsCtx_, bool debug_) : config(config),
                                                                           setupCtx(setupCtx_),
                                                                           expressionsCtx(expressionsCtx_),
                                                                           N(config.generateProof() ? 1 << setupCtx.starkInfo->starkStruct.nBits : 0),
                                                                           NExtended(config.generateProof() ? 1 << setupCtx.starkInfo->starkStruct.nBitsExt : 0),
                                                                           ntt(config.generateProof() ? 1 << setupCtx.starkInfo->starkStruct.nBits : 0),
                                                                           nttExtended(config.generateProof() ? 1 << setupCtx.starkInfo->starkStruct.nBitsExt : 0)
    {
        debug = debug_;

        // Avoid unnecessary initialization if we are not going to generate any proof
        if (!config.generateProof())
            return;

        if(setupCtx.starkInfo->starkStruct.verificationHashType == std::string("BN128")) {
            nFieldElements = 1;
        } else {
            nFieldElements = HASH_SIZE;
        }

        treesGL = new MerkleTreeType*[setupCtx.starkInfo->nStages + 2];

        TimerStart(COMPUTE_X);

        uint64_t extendBits = setupCtx.starkInfo->starkStruct.nBitsExt - setupCtx.starkInfo->starkStruct.nBits;
        x = new Goldilocks::Element[N << extendBits];
        x[0] = Goldilocks::shift();
        for (uint64_t k = 1; k < (N << extendBits); k++)
        {
            x[k] = x[k - 1] * Goldilocks::w(setupCtx.starkInfo->starkStruct.nBits + extendBits);
        }

        S = new Goldilocks::Element[setupCtx.starkInfo->qDeg];
        xis = new Goldilocks::Element[setupCtx.starkInfo->openingPoints.size() * FIELD_EXTENSION];
        Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
        S[0] = Goldilocks::one();
        for(uint64_t i = 1; i < setupCtx.starkInfo->qDeg; i++) {
            S[i] = Goldilocks::mul(S[i - 1], shiftIn);
        }
        TimerStopAndLog(COMPUTE_X);

        TimerStart(MERKLE_TREE_ALLOCATION);
        treesGL[setupCtx.starkInfo->nStages + 1] = new MerkleTreeType(setupCtx.starkInfo->starkStruct.merkleTreeArity, setupCtx.starkInfo->starkStruct.merkleTreeCustom, (Goldilocks::Element *)setupCtx.constPols->pConstTreeAddress);
        for (uint64_t i = 0; i < setupCtx.starkInfo->nStages + 1; i++)
        {
            std::string section = "cm" + to_string(i + 1);
            uint64_t nCols = setupCtx.starkInfo->mapSectionsN[section];
            treesGL[i] = new MerkleTreeType(setupCtx.starkInfo->starkStruct.merkleTreeArity, setupCtx.starkInfo->starkStruct.merkleTreeCustom, NExtended, nCols, NULL, false);
        }

        if(!debug) {            
            treesFRI = new MerkleTreeType*[setupCtx.starkInfo->starkStruct.steps.size() - 1];
            for(uint64_t step = 0; step < setupCtx.starkInfo->starkStruct.steps.size() - 1; ++step) {
                uint64_t nGroups = 1 << setupCtx.starkInfo->starkStruct.steps[step + 1].nBits;
                uint64_t groupSize = (1 << setupCtx.starkInfo->starkStruct.steps[step].nBits) / nGroups;

                treesFRI[step] = new MerkleTreeType(setupCtx.starkInfo->starkStruct.merkleTreeArity, setupCtx.starkInfo->starkStruct.merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION, NULL);
            }
        }
        TimerStopAndLog(MERKLE_TREE_ALLOCATION);

    };
    ~Starks()
    {
        if (!config.generateProof())
            return;
        
        delete S;
        delete xis;
        delete x;

        for (uint i = 0; i < setupCtx.starkInfo->nStages + 2; i++)
        {
            delete treesGL[i];
        }
        delete[] treesGL;

        if(!debug) {
            for (uint64_t i = 0; i < setupCtx.starkInfo->starkStruct.steps.size() - 1; i++)
            {
                delete treesFRI[i];
            }
            delete[] treesFRI;
        }
    };

    void genProof(Goldilocks::Element *pAddress, FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs);
    
    void calculateHints(uint64_t step, StepsParams &params);

    void extendAndMerkelize(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof);

    void calculateFRIPolynomial(StepsParams &params);

    void commitStage(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof);
    void computeStageExpressions(uint64_t step,  StepsParams &params, FRIProof<ElementType> &proof);
    void computeQ(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof);
    
    void computeEvals(StepsParams &params, FRIProof<ElementType> &proof);

    void computeFRIPol(uint64_t step, StepsParams &params);
    
    void computeFRIFolding(uint64_t step, StepsParams &params, Goldilocks::Element *challenge, FRIProof<ElementType> &fproof);
    void computeFRIQueries(FRIProof<ElementType> &fproof, uint64_t* friQueries);

    void calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements);

    void addTranscriptGL(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements);
    void getChallenge(TranscriptType &transcript, Goldilocks::Element& challenge);
private:
    void evmap(StepsParams &params, Goldilocks::Element *LEv);
    
    // ALL THIS FUNCTIONS CAN BE REMOVED WHEN WC IS READY
    bool canExpressionBeCalculated(ParserParams &parserParams, StepsParams &params);
    bool isHintResolved(Hint &hint, std::vector<string> dstFields, StepsParams &params);
    bool canHintBeResolved(Hint &hint, std::vector<string> srcFields, StepsParams &params);
    uint64_t isStageCalculated(uint64_t step, StepsParams &params);
    bool isSymbolCalculated(opType operand, uint64_t id, StepsParams &params);
    void calculateS(Polinomial &s, Polinomial &den, Goldilocks::Element multiplicity);

public:

    // Following function are created to be used by the ffi interface
    void ffi_extend_and_merkelize(uint64_t step,  StepsParams &params, FRIProof<ElementType> *proof);
    void ffi_treesGL_get_root(uint64_t index, ElementType *dst);
};

template class Starks<Goldilocks::Element>;
template class Starks<RawFr::Element>;

#endif // STARKS_H
