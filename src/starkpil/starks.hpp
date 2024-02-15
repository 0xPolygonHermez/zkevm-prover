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
#include "transcript.hpp"
#include "zhInv.hpp"
#include "steps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

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
    MerkleTreeGL **treesGL;
    MerkleTreeGL **treesFRI;

    Transcript transcript;
    Goldilocks::Element *mem;
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
        
        *x[0] = Goldilocks::shift();

        uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

        for (uint64_t k = 1; k < (N << extendBits); k++)
        {
            Polinomial::mulElement(x, k, x, k - 1, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
        }

        TimerStart(MERKLE_TREE_ALLOCATION);
        treesGL = new MerkleTreeGL*[starkInfo.nStages + 2];
        treesGL[starkInfo.nStages + 1] = new MerkleTreeGL((Goldilocks::Element *)pConstTreeAddress);

        treesFRI = new MerkleTreeGL*[starkInfo.starkStruct.steps.size() - 1];
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

    void genProof(FRIProof<Goldilocks::Element, Goldilocks> &proof, Goldilocks::Element *publicInputs, Goldilocks::Element verkey[4], Steps *steps);
    
    void calculateZ(StepsParams& params, StarkInfo starkInfo);
    void calculateH1H2(StepsParams& params, StarkInfo starkInfo);

    void extendAndMerkelize(uint64_t step, StepsParams& params, StarkInfo starkInfo, FRIProof<Goldilocks::Element, Goldilocks> &proof);
    void calculateExpressions(std::string step, uint64_t nrowsStepBatch, Steps *steps, StepsParams &params, uint64_t N);
    void computeQ(StepsParams& params, StarkInfo starkInfo, FRIProof<Goldilocks::Element, Goldilocks> &proof);
    void computeEvals(StepsParams& params, FRIProof<Goldilocks::Element, Goldilocks> &proof, StarkInfo starkInfo);

    Polinomial computeFRIPol(StepsParams& params, StarkInfo starkInfo, Steps *steps, uint64_t nrowsStepBatch);
    void computeFRIFolding(FRIProof<Goldilocks::Element, Goldilocks> &fproof, StarkInfo starkInfo, Polinomial &friPol, uint64_t step, Polinomial &challenge);
    void computeFRIQueries(FRIProof<Goldilocks::Element, Goldilocks> &fproof, StarkInfo starkInfo, Polinomial &friPol, uint64_t* friQueries);

    void addTranscript(Transcript &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(Transcript &transcript, Polinomial& pol);
    void getChallenges(Transcript &transcript, Polinomial &challenges, uint64_t nChallenges, uint64_t index);

    int findIndex(std::vector<uint64_t> openingPoints, int prime);

private:
    Polinomial *transposeH1H2Columns(StepsParams& params, StarkInfo starkInfo);
    void transposeH1H2Rows(StepsParams& params, StarkInfo starkInfo, Polinomial *transPols);
    Polinomial *transposeZColumns(StepsParams& params, StarkInfo starkInfo);
    void transposeZRows(StepsParams& params, StarkInfo starkInfo, Polinomial *transPols);
    void evmap(StepsParams &params, Polinomial *LEv, StarkInfo starkInfo);

public:
    // Following function are created to be used by the ffi interface
    void *createStepsParams(void *pChallenges, void *pEvals, void *pXDivXSubXi, void *pXDivXSubWXi, void *pPublicInputs);
    // void treeMerkelize(uint64_t index);
    // void treeGetRoot(uint64_t index, Goldilocks::Element *root);
    // void extendPol(uint64_t step);
    // void *getPBuffer();
    // void ffi_calculateH1H2(Polinomial *transPols);
    // void ffi_calculateZ(Polinomial *newPols);
    // void ffi_exps_2ns(Polinomial *qq1, Polinomial *qq2);
    // void ffi_lev_lpev(Polinomial *LEv, Polinomial *LpEv, Polinomial *xis, Polinomial *wxis, Polinomial *c_w, Polinomial *challenges);
    // void ffi_xdivxsubxi(uint64_t extendBits, Polinomial *xi, Polinomial *wxi, Polinomial *challenges, Polinomial *xDivXSubXi, Polinomial *xDivXSubWXi);
    // void ffi_finalize_proof(FRIProof<Goldilocks::Element, Goldilocks> *proof, Transcript *transcript, Polinomial *evals, Polinomial *root0, Polinomial *root1, Polinomial *root2, Polinomial *root3);
    void ffi_extend_and_merkelize(uint64_t step, void *pParams, void *pProof);
};

#endif // STARKS_H
