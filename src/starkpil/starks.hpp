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
#include "zhInv.hpp"
#include "steps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

#define STARK_C12_A_NUM_TREES 5
#define NUM_CHALLENGES 8

struct StarkFiles
{
    std::string zkevmConstPols;
    bool mapConstPolsFile;
    std::string zkevmConstantsTree;
    std::string zkevmStarkInfo;
};

class Starks
{
public:
    const Config &config;
    StarkInfo starkInfo;
    uint64_t nrowsStepBatch;

private:
    void *pConstPolsAddress;
    void *pConstPolsAddress2ns;
    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;
    void *pConstTreeAddress;
    StarkFiles starkFiles;
    ZhInv zi;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    NTT_Goldilocks nttExtended;
    Polinomial x_n;
    Polinomial x_2ns;
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
    Goldilocks::Element *pBuffer;

    void *pAddress;

    Polinomial x;

    void merkelizeMemory(); // function for DBG purposes

public:
    Starks(const Config &config, StarkFiles starkFiles, void *_pAddress) : config(config),
                                                                           starkInfo(config, starkFiles.zkevmStarkInfo),
                                                                           starkFiles(starkFiles),
                                                                           zi(config.generateProof() ? starkInfo.starkStruct.nBits : 0,
                                                                              config.generateProof() ? starkInfo.starkStruct.nBitsExt : 0),
                                                                           N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                           nttExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                           x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0),
                                                                           x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0),
                                                                           pAddress(_pAddress),
                                                                           x(config.generateProof() ? N << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 0, config.generateProof() ? FIELD_EXTENSION : 0)
    {
        nrowsStepBatch = 1;
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

        if (config.mapConstantsTreeFile)
        {
            pConstTreeAddress = mapFile(starkFiles.zkevmConstantsTree, starkInfo.getConstTreeSizeInBytes(), false);
            zklog.info("Starks::Starks() successfully mapped " + to_string(starkInfo.getConstTreeSizeInBytes()) + " bytes from constant tree file " + starkFiles.zkevmConstantsTree);
        }
        else
        {
            pConstTreeAddress = copyFile(starkFiles.zkevmConstantsTree, starkInfo.getConstTreeSizeInBytes());
            zklog.info("Starks::Starks() successfully copied " + to_string(starkInfo.getConstTreeSizeInBytes()) + " bytes from constant file " + starkFiles.zkevmConstantsTree);
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

        mem = (Goldilocks::Element *)pAddress;
        pBuffer = &mem[starkInfo.mapTotalN];

        p_cm1_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns]];
        p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];
        p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
        p_cm2_n = &mem[starkInfo.mapOffsets.section[eSection::cm2_n]];
        p_cm3_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
        p_cm3_n = &mem[starkInfo.mapOffsets.section[eSection::cm3_n]];
        cm4_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm4_2ns]];
        p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];
        p_f_2ns = &mem[starkInfo.mapOffsets.section[eSection::f_2ns]];

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
        treesGL[4] = new MerkleTreeGL((Goldilocks::Element *)pConstTreeAddress);
        TimerStopAndLog(MERKLE_TREE_ALLOCATION);
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

        for (uint i = 0; i < 5; i++)
        {
            delete treesGL[i];
        }
    };

    void genProof(FRIProof &proof, Goldilocks::Element *publicInputs, Goldilocks::Element verkey[4], Steps *steps);

    Polinomial *transposeH1H2Columns(void *pAddress, uint64_t &numCommited, Goldilocks::Element *pBuffer);
    void transposeH1H2Rows(void *pAddress, uint64_t &numCommited, Polinomial *transPols);
    Polinomial *transposeZColumns(void *pAddress, uint64_t &numCommited, Goldilocks::Element *pBuffer);
    void transposeZRows(void *pAddress, uint64_t &numCommited, Polinomial *transPols);
    void evmap(void *pAddress, Polinomial &evals, Polinomial &LEv, Polinomial &LpEv);
};

#endif // STARKS_H
