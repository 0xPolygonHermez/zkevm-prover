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
    uint64_t merkleTreeArity;
    bool merkleTreeCustom;

    Goldilocks::Element *S;
    Goldilocks::Element *xis;
    Goldilocks::Element *x;

void merkelizeMemory(Goldilocks::Element *pAddress); // function for DBG purposes

public:
    Starks(const Config &config, StarkInfo &starkInfo_, CHelpersSteps& cHelpersSteps, bool debug_) : config(config),
                                                                           starkInfo(starkInfo_),
                                                                           N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           nttExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0)
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

        TimerStart(COMPUTE_X);

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
        treesGL[starkInfo.nStages + 1] = new MerkleTreeType(merkleTreeArity, merkleTreeCustom, (Goldilocks::Element *)cHelpersSteps.constPols.pConstTreeAddress);
        for (uint64_t i = 0; i < starkInfo.nStages + 1; i++)
        {
            treesGL[i] = new MerkleTreeType(merkleTreeArity, merkleTreeCustom,  NExtended, starkInfo.mapSectionsN["cm" + to_string(i + 1)], NULL, false);
        }

        if(!debug) {            
            treesFRI = new MerkleTreeType*[starkInfo.starkStruct.steps.size() - 1];
            for(uint64_t step = 0; step < starkInfo.starkStruct.steps.size() - 1; ++step) {
                uint64_t nGroups = 1 << starkInfo.starkStruct.steps[step + 1].nBits;
                uint64_t groupSize = (1 << starkInfo.starkStruct.steps[step].nBits) / nGroups;

                treesFRI[step] = new MerkleTreeType(merkleTreeArity, merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION, NULL);
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

    void genProof(Goldilocks::Element *pAddress, FRIProof<ElementType> &proof, CHelpersSteps& cHelpersSteps, Goldilocks::Element *publicInputs);
    
    void calculateHints(uint64_t step, CHelpersSteps &cHelpersSteps);

    void extendAndMerkelize(uint64_t step, CHelpersSteps &cHelpersSteps, FRIProof<ElementType> &proof);

    void calculateFRIPolynomial(CHelpersSteps &cHelpersSteps);

    void commitStage(uint64_t step, CHelpersSteps &cHelpersSteps, FRIProof<ElementType> &proof);
    void computeStageExpressions(uint64_t step,  CHelpersSteps &cHelpersSteps, FRIProof<ElementType> &proof);
    void computeQ(uint64_t step, CHelpersSteps &cHelpersSteps, FRIProof<ElementType> &proof);
    
    void computeEvals(CHelpersSteps &cHelpersSteps, FRIProof<ElementType> &proof);

    void computeFRIPol(uint64_t step, CHelpersSteps &cHelpersSteps);
    
    void computeFRIFolding(uint64_t step, CHelpersSteps& cHelpersSteps, Goldilocks::Element *challenge, FRIProof<ElementType> &fproof);
    void computeFRIQueries(FRIProof<ElementType> &fproof, uint64_t* friQueries);

    void calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements);

    void addTranscriptGL(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements);
    void getChallenge(TranscriptType &transcript, Goldilocks::Element& challenge);

private:
    void evmap(CHelpersSteps &cHelpersSteps, Goldilocks::Element *LEv);
    
    // ALL THIS FUNCTIONS CAN BE REMOVED WHEN WC IS READY
    bool canExpressionBeCalculated(ParserParams &parserParams, CHelpersSteps& cHelpersSteps);
    bool isHintResolved(Hint &hint, std::vector<string> dstFields, CHelpersSteps& cHelpersSteps);
    bool canHintBeResolved(Hint &hint, std::vector<string> srcFields, CHelpersSteps& cHelpersSteps);
    uint64_t isStageCalculated(uint64_t step, CHelpersSteps &cHelpersSteps);
    bool isSymbolCalculated(opType operand, uint64_t id, CHelpersSteps &cHelpersSteps);
    void calculateS(Polinomial &s, Polinomial &den, Goldilocks::Element multiplicity);

public:

    // Following function are created to be used by the ffi interface
    void ffi_extend_and_merkelize(uint64_t step,  CHelpersSteps &cHelpersSteps, FRIProof<ElementType> *proof);
    void ffi_treesGL_get_root(uint64_t index, ElementType *dst);
};

template class Starks<Goldilocks::Element>;
template class Starks<RawFr::Element>;

#endif // STARKS_H
