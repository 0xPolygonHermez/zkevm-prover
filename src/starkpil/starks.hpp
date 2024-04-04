#ifndef STARKS_HPP
#define STARKS_HPP

#include <algorithm>
#include <cmath>
#include "config.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "constant_pols_starks.hpp"
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
#include "hint/subproof_value_hint_handler.hpp"

using namespace Hints;

struct StarkFiles
{
    std::string zkevmConstPols;
    bool mapConstPolsFile;
    std::string zkevmConstantsTree;
    std::string zkevmStarkInfo;
    std::string zkevmCHelpers;
};

template <typename ElementType>
class Starks
{
public:
    const Config &config;
    StarkInfo starkInfo;
    bool debug = false;
    bool optimizeMemoryNTT = false;
    uint32_t nrowsBatch = NROWS_BATCH;
    
    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

private:
    void *pConstPolsAddress;
    void *pConstPolsAddress2ns;
    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;
    void *pConstTreeAddress;
    StarkFiles starkFiles;
    uint64_t N;
    uint64_t NExtended;
    uint64_t hashSize;
    uint64_t merkleTreeArity;
    bool merkleTreeCustom;
    NTT_Goldilocks ntt;
    NTT_Goldilocks nttExtended;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial zi;
    uint64_t constPolsSize;
    uint64_t constPolsDegree;
    MerkleTreeType **treesGL;
    MerkleTreeType **treesFRI;

    vector<bool> publicsCalculated;
    vector<bool> constsCalculated;
    vector<bool> subProofValuesCalculated;
    vector<bool> witnessCalculated;
    vector<bool> challengesCalculated;

    Goldilocks::Element *mem;
    void *pAddress;

    Polinomial x;

    std::unique_ptr<BinFileUtils::BinFile> cHelpersBinFile;
    CHelpers chelpers;

void merkelizeMemory(); // function for DBG purposes
void printPolRoot(uint64_t polId, StepsParams& params); // function for DBG purposes

public:
    Starks(const Config &config, StarkFiles starkFiles, void *_pAddress, bool debug_) : config(config),
                                                                           starkInfo(starkFiles.zkevmStarkInfo),
                                                                           starkFiles(starkFiles),
                                                                           N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           nttExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0),
                                                                           x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0),
                                                                           zi(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0),
                                                                           pAddress(_pAddress),
                                                                           x(config.generateProof() ? N << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 0, config.generateProof() ? FIELD_EXTENSION : 0)
    {
        debug = debug_;

        // Avoid unnecessary initialization if we are not going to generate any proof
        if (!config.generateProof())
            return;

        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            hashSize = 1;
            merkleTreeArity = starkInfo.starkStruct.merkleTreeArity;
            merkleTreeCustom = starkInfo.starkStruct.merkleTreeCustom;
        } else {
            hashSize = HASH_SIZE;
            merkleTreeArity = 2;
            merkleTreeCustom = true;
        }

        // Allocate an area of memory, mapped to file, to read all the constant polynomials,
        // and create them using the allocated address
        TimerStart(LOAD_CONST_POLS_TO_MEMORY);
        pConstPolsAddress = NULL;
        if (starkFiles.zkevmConstPols.size() == 0)
        {
            zklog.error("Starks::Starks() received an empty config.zkevmConstPols");
            exitProcess();
        }

        constPolsDegree = (1 << starkInfo.starkStruct.nBits);
        constPolsSize = starkInfo.nConstants * sizeof(Goldilocks::Element) * constPolsDegree;

        if (starkFiles.mapConstPolsFile)
        {
            pConstPolsAddress = mapFile(starkFiles.zkevmConstPols, constPolsSize, false);
            zklog.info("Starks::Starks() successfully mapped " + to_string(constPolsSize) + " bytes from constant file " + starkFiles.zkevmConstPols);
        }
        else
        {
        pConstPolsAddress = copyFile(starkFiles.zkevmConstPols, constPolsSize);
            zklog.info("Starks::Starks() successfully copied " + to_string(constPolsSize) + " bytes from constant file " + starkFiles.zkevmConstPols);
        }
        pConstPols = new ConstantPolsStarks(pConstPolsAddress, constPolsSize, starkInfo.nConstants);
        TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

        if(!debug) {
            // Map constants tree file to memory
            TimerStart(LOAD_CONST_TREE_TO_MEMORY);
            pConstTreeAddress = NULL;
            if (starkFiles.zkevmConstantsTree.size() == 0)
            {
                zklog.error("Starks::Starks() received an empty config.zkevmConstantsTree");
                exitProcess();
            }

            uint64_t constTreeSizeBytes = getConstTreeSize();

            if (config.mapConstantsTreeFile)
            {
                pConstTreeAddress = mapFile(starkFiles.zkevmConstantsTree, constTreeSizeBytes, false);
                zklog.info("Starks::Starks() successfully mapped " + to_string(constTreeSizeBytes) + " bytes from constant tree file " + starkFiles.zkevmConstantsTree);
            }
            else
            {
                pConstTreeAddress = copyFile(starkFiles.zkevmConstantsTree, constTreeSizeBytes);
                zklog.info("Starks::Starks() successfully copied " + to_string(constTreeSizeBytes) + " bytes from constant file " + starkFiles.zkevmConstantsTree);
            }
            TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);

            // Initialize and allocate ConstantPols2ns
            TimerStart(LOAD_CONST_POLS_2NS_TO_MEMORY);
            pConstPolsAddress2ns = (void *)calloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt), sizeof(Goldilocks::Element));
            pConstPols2ns = new ConstantPolsStarks(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants);
            std::memcpy(pConstPolsAddress2ns, (uint8_t *)pConstTreeAddress + 2 * sizeof(Goldilocks::Element), starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element));

            TimerStopAndLog(LOAD_CONST_POLS_2NS_TO_MEMORY);
        }

        // TODO x_n and x_2ns could be precomputed
        TimerStart(COMPUTE_X_N_AND_X_2_NS);
        Goldilocks::Element xx = Goldilocks::one();
        for (uint64_t i = 0; i < N; i++)
        {
            *x_n[i] = xx;
            Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBits));
        }
        xx = Goldilocks::shift();
        for (uint64_t i = 0; i < NExtended; i++)
        {
            *x_2ns[i] = xx;
            Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBitsExt));
        }
        TimerStopAndLog(COMPUTE_X_N_AND_X_2_NS);

        TimerStart(COMPUTE_ZHINV);
        Polinomial::buildZHInv(zi, starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt);
        TimerStopAndLog(COMPUTE_ZHINV);

        mem = (Goldilocks::Element *)pAddress;
        
        *x[0] = Goldilocks::shift();

        uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

        for (uint64_t k = 1; k < (N << extendBits); k++)
        {
            x[k][0] = x[k - 1][0] * Goldilocks::w(starkInfo.starkStruct.nBits + extendBits);
        }

        TimerStart(MERKLE_TREE_ALLOCATION);
        treesGL = new MerkleTreeType*[starkInfo.nStages + 2];
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
            treesGL[starkInfo.nStages + 1] = new MerkleTreeType(merkleTreeArity, merkleTreeCustom, (Goldilocks::Element *)pConstTreeAddress);
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

        TimerStart(CHELPERS_ALLOCATION);
        if(!starkFiles.zkevmCHelpers.empty()) {
            cHelpersBinFile = BinFileUtils::openExisting(starkFiles.zkevmCHelpers, "chps", 1);
            chelpers.loadCHelpers(cHelpersBinFile.get());
        }
        TimerStopAndLog(CHELPERS_ALLOCATION);

        constsCalculated.resize(starkInfo.nConstants, true);

        publicsCalculated.resize(starkInfo.nPublics, false);
        subProofValuesCalculated.resize(starkInfo.nSubProofValues, false);
        challengesCalculated.resize(starkInfo.challengesMap.size(), false);
        witnessCalculated.resize(starkInfo.cmPolsMap.size(), false);
        
        uint64_t currentSectionStart = starkInfo.mapOffsets[std::make_pair("cm" + to_string(starkInfo.nStages), false)] * sizeof(Goldilocks::Element);
        uint64_t nttHelperSize = starkInfo.mapSectionsN["cm" + to_string(starkInfo.nStages)] * NExtended * sizeof(Goldilocks::Element);
        if(currentSectionStart > nttHelperSize) {
            optimizeMemoryNTT = true;
        }
    };
    ~Starks()
    {
        if (!config.generateProof())
            return;

        delete pConstPols;
        
        if (config.mapConstPolsFile)
        {
            unmapFile(pConstPolsAddress, constPolsSize);
        }
        else
        {
            free(pConstPolsAddress);
        }

        for (uint i = 0; i < starkInfo.nStages + 2; i++)
        {
            delete treesGL[i];
        }
        delete[] treesGL;

        if(!debug) {
            if (config.mapConstantsTreeFile)
            {
                unmapFile(pConstTreeAddress, constPolsSize);
            }
            else
            {
                free(pConstTreeAddress);
            }

            delete pConstPols2ns;
            free(pConstPolsAddress2ns);

            for (uint64_t i = 0; i < starkInfo.starkStruct.steps.size() - 1; i++)
            {
                delete treesFRI[i];
            }
            delete[] treesFRI;
        }
    };

    uint64_t getConstTreeSize()
    {   
        uint n_tmp = 1 << starkInfo.starkStruct.nBitsExt;
        uint64_t nextN = floor(((double)(n_tmp - 1) / merkleTreeArity) + 1);
        uint64_t acc = nextN * merkleTreeArity;
        while (n_tmp > 1)
        {
            // FIll with zeros if n nodes in the leve is not even
            n_tmp = nextN;
            nextN = floor((n_tmp - 1) / merkleTreeArity) + 1;
            if (n_tmp > 1)
            {
                acc += nextN * merkleTreeArity;
            }
            else
            {
                acc += 1;
            }
        }

        uint64_t numElements = (1 << starkInfo.starkStruct.nBitsExt) * starkInfo.nConstants * sizeof(Goldilocks::Element);
        uint64_t total = numElements + acc * hashSize * sizeof(ElementType);
        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            total += 16; // HEADER
        } else {
            total += merkleTreeArity * sizeof(ElementType);
        }
        return total; 
        
    };

    void genProof(FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs, CHelpersSteps *chelpersSteps);
    
    void calculateHints(uint64_t step, StepsParams& params, vector<Hint> &hints);

    void extendAndMerkelize(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof);

    void calculateExpressions(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps);
    void calculateExpression(uint64_t id, StepsParams &params, CHelpersSteps *chelpersSteps);
    void calculateConstraint(uint64_t constraintId, StepsParams &params, CHelpersSteps *chelpersSteps);
    
    void computeStage(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof, TranscriptType &transcript, CHelpersSteps *chelpersSteps);
    void computeQ(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof);
    
    void computeEvals(StepsParams& params, FRIProof<ElementType> &proof);

    Polinomial *computeFRIPol(uint64_t step, StepsParams& params, CHelpersSteps *chelpersSteps);
    
    void computeFRIFolding(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t step, Polinomial &challenge);
    void computeFRIQueries(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t* friQueries);

    void calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements);
    void calculateHash(ElementType* hash, Polinomial &pol);

    void addTranscriptGL(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, Polinomial &pol);
    void getChallenge(TranscriptType &transcript, Goldilocks::Element& challenge);

private:
    bool canExpressionBeCalculated(ParserParams &parserParams);

    void transposePolsColumns(StepsParams& params, vector<int64_t> cm2Transposed, Polinomial* transPols, Hint hint, Goldilocks::Element *pBuffer);
    void transposePolsRows(StepsParams& params, vector<int64_t> cm2Transposed, Polinomial *transPols, Hint hint);

    bool isHintResolved(Hint &hint, std::vector<string> dstFields);
    bool canHintBeResolved(Hint &hint, std::vector<string> srcFields);

    void evmap(StepsParams &params, Polinomial &LEv);

    uint64_t isStageCalculated(uint64_t step);
    bool isSymbolCalculated(opType operand, uint64_t id);
    void setSymbolCalculated(opType operand, uint64_t id);
    void cleanSymbolsCalculated();

public:
    // Following function are created to be used by the ffi interface
    void *ffi_create_steps_params(Polinomial *pChallenges, Polinomial* pSubproofValues, Polinomial *pEvals, Polinomial *pXDivXSubXi, Goldilocks::Element *pPublicInputs);
    void ffi_extend_and_merkelize(uint64_t step, StepsParams* params, FRIProof<ElementType>* proof);
    void ffi_treesGL_get_root(uint64_t index, ElementType *dst);

    void *ffi_get_vector_pointer(char *name);
};

template class Starks<Goldilocks::Element>;
template class Starks<RawFr::Element>;

#endif // STARKS_H
