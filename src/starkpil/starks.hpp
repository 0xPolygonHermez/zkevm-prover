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
    bool optimizeMemoryNTT = false;
    
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
    NTT_Goldilocks ntt;
    NTT_Goldilocks nttExtended;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial zi;
    uint64_t constPolsSize;
    uint64_t constPolsDegree;
    MerkleTreeType **treesGL;
    MerkleTreeType **treesFRI;

    Goldilocks::Element *mem;
    void *pAddress;

    Polinomial x;

    std::unique_ptr<BinFileUtils::BinFile> cHelpersBinFile;
    CHelpers chelpers;

void merkelizeMemory(); // function for DBG purposes

public:
    Starks(const Config &config, StarkFiles starkFiles, void *_pAddress) : config(config),
                                                                           starkInfo(config, starkFiles.zkevmStarkInfo),
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
        // Avoid unnecessary initialization if we are not going to generate any proof
        if (!config.generateProof())
            return;

        if(starkInfo.starkStruct.verificationHashType == std::string("BN128")) {
            hashSize = 1;
            merkleTreeArity = 16;
        } else {
            hashSize = HASH_SIZE;
            merkleTreeArity = 2;
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
            Polinomial::mulElement(x, k, x, k - 1, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
        }

        TimerStart(MERKLE_TREE_ALLOCATION);
        treesGL = new MerkleTreeType*[starkInfo.nStages + 2];
        for (uint64_t i = 0; i < starkInfo.nStages + 1; i++)
        {
            std::string section = "cm" + to_string(i + 1) + "_n";
            std::string sectionExt = "cm" + to_string(i + 1) + "_2ns";
            uint64_t nCols = starkInfo.mapSectionsN.section[string2section(section)];
            Goldilocks::Element *pBuffExtended = &mem[starkInfo.mapOffsets.section[string2section(sectionExt)]];
            treesGL[i] = new MerkleTreeType(NExtended, nCols, pBuffExtended);
        }
        treesGL[starkInfo.nStages + 1] = new MerkleTreeType((Goldilocks::Element *)pConstTreeAddress);

        treesFRI = new MerkleTreeType*[starkInfo.starkStruct.steps.size() - 1];
        for(uint64_t step = 0; step < starkInfo.starkStruct.steps.size() - 1; ++step) {
            uint64_t nGroups = 1 << starkInfo.starkStruct.steps[step + 1].nBits;
            uint64_t groupSize = (1 << starkInfo.starkStruct.steps[step].nBits) / nGroups;

            treesFRI[step] = new MerkleTreeType(nGroups, groupSize * FIELD_EXTENSION, NULL);
        }
        TimerStopAndLog(MERKLE_TREE_ALLOCATION);

        TimerStart(CHELPERS_ALLOCATION);
        if(!starkFiles.zkevmCHelpers.empty()) {
            cHelpersBinFile = BinFileUtils::openExisting(starkFiles.zkevmCHelpers, "chps", 1);
            chelpers.loadCHelpers(cHelpersBinFile.get());
        }
        TimerStopAndLog(CHELPERS_ALLOCATION);
    };
    ~Starks()
    {
        if (!config.generateProof())
            return;

        delete pConstPols;
        delete pConstPols2ns;
        free(pConstPolsAddress2ns);

        if (config.mapConstPolsFile)
        {
            unmapFile(pConstPolsAddress, constPolsSize);
        }
        else
        {
            free(pConstPolsAddress);
        }
        if (config.mapConstantsTreeFile)
        {
            unmapFile(pConstTreeAddress, constPolsSize);
        }
        else
        {
            free(pConstTreeAddress);
        }

        for (uint i = 0; i < starkInfo.nStages + 2; i++)
        {
            delete treesGL[i];
        }
        delete[] treesGL;

        for (uint64_t i = 0; i < starkInfo.starkStruct.steps.size() - 1; i++)
        {
            delete treesFRI[i];
        }
        delete[] treesFRI;
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
            total += merkleTreeArity;
        } else {
            total += merkleTreeArity * sizeof(ElementType);
        }
        return total; 
        
    };

    void genProof(FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs, CHelpersSteps *chelpersSteps);
    
    void calculateZ(StepsParams& params);
    void calculateH1H2(StepsParams& params);

    void extendAndMerkelize(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof);

    void calculateExpressions(std::string step, StepsParams &params, CHelpersSteps *chelpersSteps);
    
    void computeQ(StepsParams& params, FRIProof<ElementType> &proof);
    void computeEvals(StepsParams& params, FRIProof<ElementType> &proof);

    Polinomial *computeFRIPol(StepsParams& params, CHelpersSteps *chelpersSteps);
    
    void computeFRIFolding(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t step, Polinomial &challenge);
    void computeFRIQueries(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t* friQueries);

    void addTranscriptPublics(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, Polinomial &pol);
public:
    void getChallenges(TranscriptType &transcript, Goldilocks::Element* challenges, uint64_t nChallenges);

private:
    int findIndex(std::vector<uint64_t> openingPoints, int prime);

    Polinomial *transposeH1H2Columns(StepsParams& params);
    void transposeH1H2Rows(StepsParams& params, Polinomial *transPols);
    Polinomial *transposeZColumns(StepsParams& params);
    void transposeZRows(StepsParams& params, Polinomial *transPols);
    void evmap(StepsParams &params, Polinomial &LEv);

public:
    // Following function are created to be used by the ffi interface
    void *ffi_create_steps_params(Polinomial *pChallenges, Polinomial *pEvals, Polinomial *pXDivXSubXi, Goldilocks::Element *pPublicInputs);
    void ffi_extend_and_merkelize(uint64_t step, StepsParams* params, FRIProof<ElementType>* proof);
    void ffi_treesGL_get_root(uint64_t index, ElementType *dst);
};

template class Starks<Goldilocks::Element>;
template class Starks<RawFr::Element>;

#endif // STARKS_H
