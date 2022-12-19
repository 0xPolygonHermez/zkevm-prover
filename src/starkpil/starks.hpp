#ifndef STARKS_HPP
#define STARKS_HPP

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

public:
    Starks(const Config &config, StarkFiles starkFiles) : config(config),
                                                          starkInfo(config, starkFiles.zkevmStarkInfo),
                                                          starkFiles(starkFiles),
                                                          zi(config.generateProof() ? starkInfo.starkStruct.nBits : 0,
                                                             config.generateProof() ? starkInfo.starkStruct.nBitsExt : 0),
                                                          N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                          NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                          ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                          nttExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                          x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0),
                                                          x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0)

    {
        // Avoid unnecessary initialization if we are not going to generate any proof
        if (!config.generateProof())
            return;

        // Allocate an area of memory, mapped to file, to read all the constant polynomials,
        // and create them using the allocated address
        TimerStart(LOAD_CONST_POLS_TO_MEMORY);
        pConstPolsAddress = NULL;
        if (starkFiles.zkevmConstPols.size() == 0)
        {
            cerr << "Error: Starks::Starks() received an empty config.zkevmConstPols" << endl;
            exit(-1);
        }
        constPolsDegree = (1 << starkInfo.starkStruct.nBits);
        constPolsSize = starkInfo.nConstants * sizeof(Goldilocks::Element) * constPolsDegree;

        if (starkFiles.mapConstPolsFile)
        {
            pConstPolsAddress = mapFile(starkFiles.zkevmConstPols, constPolsSize, false);
            cout << "Starks::Starks() successfully mapped " << constPolsSize << " bytes from constant file " << starkFiles.zkevmConstPols << endl;
        }
        else
        {
            pConstPolsAddress = copyFile(starkFiles.zkevmConstPols, constPolsSize);
            cout << "Starks::Starks() successfully copied " << constPolsSize << " bytes from constant file " << starkFiles.zkevmConstPols << endl;
        }
        pConstPols = new ConstantPolsStarks(pConstPolsAddress, constPolsSize, starkInfo.nConstants);
        TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

        // Map constants tree file to memory
        TimerStart(LOAD_CONST_TREE_TO_MEMORY);
        pConstTreeAddress = NULL;
        if (starkFiles.zkevmConstantsTree.size() == 0)
        {
            cerr << "Error: Starks::Starks() received an empty config.zkevmConstantsTree" << endl;
            exit(-1);
        }

        if (config.mapConstantsTreeFile)
        {
            pConstTreeAddress = mapFile(starkFiles.zkevmConstantsTree, starkInfo.getConstTreeSizeInBytes(), false);
            cout << "Starks::Starks() successfully mapped " << starkInfo.getConstTreeSizeInBytes() << " bytes from constant tree file " << starkFiles.zkevmConstantsTree << endl;
        }
        else
        {
            pConstTreeAddress = copyFile(starkFiles.zkevmConstantsTree, starkInfo.getConstTreeSizeInBytes());
            cout << "Starks::Starks() successfully copied " << starkInfo.getConstTreeSizeInBytes() << " bytes from constant file " << starkFiles.zkevmConstantsTree << endl;
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
    };
    
    void genProof(void *pAddress, FRIProof &proof, Goldilocks::Element *publicInputs, Steps *steps);

    Polinomial *transposeH1H2Columns(void *pAddress, uint64_t &numCommited, Goldilocks::Element *pBuffer);
    void transposeH1H2Rows(void *pAddress, uint64_t &numCommited, Polinomial *transPols);
    Polinomial *transposeZColumns(void *pAddress, uint64_t &numCommited, Goldilocks::Element *pBuffer);
    void transposeZRows(void *pAddress, uint64_t &numCommited, Polinomial *transPols);
    void evmap(void *pAddress, Polinomial &evals, Polinomial &LEv, Polinomial &LpEv);
};

#endif // STARKS_H
