#ifndef STARKS_HPP
#define STARKS_HPP

#include <algorithm>
#include "config.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "constant_pols_starks.hpp"
#include "friProof.hpp"
#include "friProofC12.hpp"
#include "friProve.hpp"
#include "transcript.hpp"
#include "steps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "chelpers.hpp"
#include "chelpers_steps.hpp"

#define STARK_C12_A_NUM_TREES 5
#define NUM_CHALLENGES 8

struct StarkFiles
{
    std::string zkevmConstPols;
    bool mapConstPolsFile;
    std::string zkevmConstantsTree;
    std::string zkevmStarkInfo;
    std::string zkevmCHelpers;
};

class Starks
{
public:
    const Config &config;
    StarkInfo starkInfo;
    bool reduceMemory;

private:
    void *pConstPolsAddress;
    void *pConstPolsAddress2ns;
    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;
    void *pConstTreeAddress;
    StarkFiles starkFiles;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    NTT_Goldilocks nttExtended;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial zi;
    uint64_t constPolsSize;
    uint64_t constPolsDegree;
    MerkleTreeGL *treesGL[STARK_C12_A_NUM_TREES];

    Goldilocks::Element *mem;

    Goldilocks::Element *p_cm1_2ns;
    Goldilocks::Element *p_cm1_n;
    Goldilocks::Element *p_cm2_2ns;
    Goldilocks::Element *p_cm2_n;
    Goldilocks::Element *p_cm3_2ns;
    Goldilocks::Element *p_cm3_n;
    Goldilocks::Element *cm4_2ns;
    Goldilocks::Element *p_q_2ns;
    Goldilocks::Element *p_f_2ns;
    Goldilocks::Element *p_tmpExp_n;

    Goldilocks::Element *p_cm1_2ns_tmp;
    Goldilocks::Element *p_cm2_2ns_tmp;

    void *pAddress;

    Polinomial x;

    std::unique_ptr<BinFileUtils::BinFile> cHelpersBinFile;
    CHelpers chelpers;

void printPolRoot(uint64_t polId, StepsParams& params); // function for DBG purposes
void merkelizeMemory(); // function for DBG purposes

public:
    Starks(const Config &config, StarkFiles starkFiles, bool reduceMemory_, void *_pAddress) : config(config),
                                                                           starkInfo(starkFiles.zkevmStarkInfo, reduceMemory_),
                                                                           reduceMemory(reduceMemory_),
                                                                           starkFiles(starkFiles),
                                                                           N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           nttExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0, true),
                                                                           x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0, true),
                                                                           zi(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0, true),
                                                                           pAddress(_pAddress),
                                                                           x(config.generateProof() ? N << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 0, config.generateProof() ? FIELD_EXTENSION : 0)
    {
        TimerStart(STARK_CONSTRUCTOR);
        // Avoid unnecessary initialization if we are not going to generate any proof
        if (!config.generateProof())
            return;

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
            pConstPolsAddress = loadFileParallel(starkFiles.zkevmConstPols, constPolsSize);
            zklog.info("Starks::Starks() successfully copied " + to_string(constPolsSize) + " bytes from constant file " + starkFiles.zkevmConstPols);
        }
        pConstPols = new ConstantPolsStarks(pConstPolsAddress, N, starkInfo.nConstants);
        if(pConstPols == NULL)
        {
            zklog.error("Starks::Starks() failed to allocate pConstPols");
            exitProcess();
        }
        TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

        if (!LOAD_CONST_FILES)
        {
            TimerStart(CALCULATE_CONST_TREE_TO_MEMORY);
            pConstPolsAddress2ns = (void *)malloc_zkevm(NExtended * starkInfo.nConstants * sizeof(Goldilocks::Element));
            if(pConstPolsAddress2ns == NULL)
            {
                zklog.error("Starks::Starks() failed to allocate pConstPolsAddress2ns");
                exitProcess();
            }
            TimerStart(EXTEND_CONST_POLS);
            uint64_t nBlocks = 8;
            uint64_t bufferSize = ((2 * NExtended * starkInfo.nConstants) * sizeof(Goldilocks::Element) / nBlocks);
            Goldilocks::Element* nttHelper = (Goldilocks::Element *)malloc(bufferSize);
            if(nttHelper == NULL)
            {
                zklog.error("Starks::Starks() failed to allocate nttHelper");
                exitProcess();
            }
            ntt.extendPol((Goldilocks::Element *)pConstPolsAddress2ns, (Goldilocks::Element *)pConstPolsAddress, NExtended, N, starkInfo.nConstants, nttHelper, 3, nBlocks);
            free(nttHelper);
            TimerStopAndLog(EXTEND_CONST_POLS);
            TimerStart(MERKELIZE_CONST_TREE);
            treesGL[4] = new MerkleTreeGL(NExtended, starkInfo.nConstants, (Goldilocks::Element *)pConstPolsAddress2ns);
            treesGL[4]->merkelize();
            TimerStopAndLog(MERKELIZE_CONST_TREE);
            Polinomial rootC(HASH_SIZE, 1);
            treesGL[4]->getRoot(rootC.address());
            zklog.info("MerkleTree rootGL C: [ " + rootC.toString(4) + " ]");
            TimerStopAndLog(CALCULATE_CONST_TREE_TO_MEMORY);
        } else {
            // Map constants tree file to memory
            pConstTreeAddress = NULL;
            if (starkFiles.zkevmConstantsTree.size() == 0)
            {
                zklog.error("Starks::Starks() received an empty config.zkevmConstantsTree");
                exitProcess();
            }
            TimerStart(LOAD_CONST_TREE_TO_MEMORY);
            if (config.mapConstantsTreeFile)
            {
                pConstTreeAddress = mapFile(starkFiles.zkevmConstantsTree, starkInfo.getConstTreeSizeInBytes(), false);
                zklog.info("Starks::Starks() successfully mapped " + to_string(starkInfo.getConstTreeSizeInBytes()) + " bytes from constant tree file " + starkFiles.zkevmConstantsTree);
            }
            else
            {
                pConstTreeAddress = loadFileParallel(starkFiles.zkevmConstantsTree, starkInfo.getConstTreeSizeInBytes());
                zklog.info("Starks::Starks() successfully copied " + to_string(starkInfo.getConstTreeSizeInBytes()) + " bytes from constant file " + starkFiles.zkevmConstantsTree);
            }
            treesGL[4] = new MerkleTreeGL((Goldilocks::Element *)pConstTreeAddress);
            pConstPolsAddress2ns = (uint8_t *)pConstTreeAddress + 2 * sizeof(uint64_t);
            TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);
        }

        // Allocate ConstantPols2ns
        pConstPols2ns = new ConstantPolsStarks(pConstPolsAddress2ns, NExtended, starkInfo.nConstants);

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

        p_cm1_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns]];
        p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];
        p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
        p_cm2_n = &mem[starkInfo.mapOffsets.section[eSection::cm2_n]];
        p_cm3_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
        p_cm3_n = &mem[starkInfo.mapOffsets.section[eSection::cm3_n]];
        p_tmpExp_n = &mem[starkInfo.mapOffsets.section[eSection::tmpExp_n]];
        cm4_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm4_2ns]];
        p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];
        p_f_2ns = &mem[starkInfo.mapOffsets.section[eSection::f_2ns]];

        p_cm1_2ns_tmp = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns_tmp]];
        p_cm2_2ns_tmp = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns_tmp]];

        *x[0] = Goldilocks::shift();

        uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

        for (uint64_t k = 1; k < (N << extendBits); k++)
        {
            Polinomial::mulElement(x, k, x, k - 1, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
        }

        TimerStart(MERKLE_TREE_ALLOCATION);
        treesGL[0] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN.section[eSection::cm1_n], p_cm1_2ns);
        treesGL[1] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN.section[eSection::cm2_n], p_cm2_2ns);
        treesGL[2] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN.section[eSection::cm3_n], p_cm3_2ns);
        treesGL[3] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN.section[eSection::cm4_2ns], cm4_2ns);
        TimerStopAndLog(MERKLE_TREE_ALLOCATION);

        TimerStart(CHELPERS_ALLOCATION);
        if(!starkFiles.zkevmCHelpers.empty()) {
            cHelpersBinFile = BinFileUtils::openExisting(starkFiles.zkevmCHelpers, "chps", 1);
            chelpers.loadCHelpers(cHelpersBinFile.get());
        }
        TimerStopAndLog(CHELPERS_ALLOCATION);
        TimerStopAndLog(STARK_CONSTRUCTOR);
    };
    ~Starks()
    {
        if (!config.generateProof())
            return;

        delete pConstPols;
        delete pConstPols2ns;
        
        if (config.mapConstPolsFile)
        {
            unmapFile(pConstPolsAddress, constPolsSize);
        } else {
            free_zkevm(pConstPolsAddress);
        }

        if(LOAD_CONST_FILES) {
            if (config.mapConstantsTreeFile) {
                unmapFile(pConstTreeAddress, constPolsSize);
            } else {
                free(pConstTreeAddress);
            }
        } else {
            free_zkevm(pConstPolsAddress2ns);
        }

        for (uint i = 0; i < 5; i++)
        {
            delete treesGL[i];
        }

        BinFileUtils::BinFile *pCHelpers = cHelpersBinFile.release();
        assert(cHelpersBinFile.get() == nullptr);
        assert(cHelpersBinFile == nullptr);
        delete pCHelpers;

    };

    void genProof(FRIProof &proof, Goldilocks::Element *publicInputs, Goldilocks::Element verkey[4], CHelpersSteps *chelpersSteps);

    Polinomial *transposeH1H2Columns(void *pAddress, uint64_t &numCommited, StepsParams& params);
    void transposeH1H2Rows(void *pAddress, uint64_t &numCommited, Polinomial *transPols);
    Polinomial *transposeZColumns(void *pAddress, uint64_t &numCommited, StepsParams& params);
    void transposeZRows(void *pAddress, uint64_t &numCommited, Polinomial *transPols);
    void evmap(void *pAddress, Polinomial &evals, Polinomial &LEv, Polinomial &LpEv);
};

#endif // STARKS_H
