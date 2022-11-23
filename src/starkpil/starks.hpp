#include "config.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "abstractPols.hpp"

#define STARK_C12_A_NUM_TREES 5
#define NUM_CHALLENGES 8

class Starks
{
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
    uint64_t numCommited;
    uint64_t N;
    uint64_t NExtended;
    NTT_Goldilocks ntt;
    Polinomial x_n;
    Polinomial x_2ns;
    Polinomial challenges;
    Polinomial xDivXSubXi;
    Polinomial xDivXSubWXi;
    Polinomial evals;
    MerkleTreeGL *treesGL[STARK_C12_A_NUM_TREES];
    uint64_t constPolsSize;
    uint64_t constPolsDegree;

public:
    Starks(const Config &config, StarkFiles starkFiles) : config(config),
                                                          starkInfo(config, starkFiles.zkevmStarkInfo),
                                                          zi(config.generateProof() ? starkInfo.starkStruct.nBits : 0,
                                                             config.generateProof() ? starkInfo.starkStruct.nBitsExt : 0),
                                                          numCommited(starkInfo.nCm1),
                                                          N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                          NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                          ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                          x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0),
                                                          x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0),
                                                          challenges(config.generateProof() ? NUM_CHALLENGES : 0, config.generateProof() ? FIELD_EXTENSION : 0),
                                                          xDivXSubXi(config.generateProof() ? NExtended : 0, config.generateProof() ? FIELD_EXTENSION : 0),
                                                          xDivXSubWXi(config.generateProof() ? NExtended : 0, config.generateProof() ? FIELD_EXTENSION : 0),
                                                          evals(config.generateProof() ? N : 0, config.generateProof() ? FIELD_EXTENSION : 0),
                                                          starkFiles(starkFiles)

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
        constPolsSize = ConstantPolsStarks::numPols() * sizeof(Goldilocks::Element) * constPolsDegree;

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
        pConstPols = new ConstantPolsStarks(pConstPolsAddress, constPolsSize);
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
        pConstPols2ns = new ConstantPolsStarks(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt));
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
        for (uint i = 0; i < 5; i++)
        {
            treesGL[i] = new MerkleTreeGL();
        }
        TimerStopAndLog(COMPUTE_X_N_AND_X_2_NS);
    };
};
