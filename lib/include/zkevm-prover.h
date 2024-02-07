#ifndef ZKEVM_API_H
#define ZKEVM_API_H
    #include <stdint.h>

    int zkevm_main(char *configFile, void* pAddress);

    // FFI functions

    // Steps
    // ========================================================================================
    void *zkevm_steps_new();
    void zkevm_steps_free(void *pZkevmSteps);
    void *c12a_steps_new();
    void c12a_steps_free(void *pC12aSteps);
    void *recursive1_steps_new();
    void recursive1_steps_free(void *pRecursive1Steps);
    void *recursive2_steps_new();
    void recursive2_steps_free(void *Recursive2Steps);


    // FRIProof
    // ========================================================================================
    void *fri_proof_new(uint64_t polN, uint64_t dim, uint64_t numTrees, uint64_t evalSize, uint64_t nPublics);
    void fri_proof_free(void *pFriProof);

    // Config
    // ========================================================================================
    void *config_new(char* filename);
    void config_free(void *pConfig);

    // Starks
    // ========================================================================================
    void *starks_new(void *pConfig, char* constPols, bool mapConstPolsFile, char* constantsTree, char* starkInfo, void *pAddress);
    void starks_gen_proof(void *pStarks, void *pFRIProof, void *pPublicInputs, void *pVerkey, void *pSteps);
    void starks_free(void *pStarks);

    // CommitPolsStarks
    // ========================================================================================
    void *commit_pols_starks_new(void *pAddress, uint64_t degree, uint64_t nCommitedPols);
    void commit_pols_starks_free(void *pCommitPolsStarks);

    // Circom
    // ========================================================================================
    void circom_get_commited_pols(void *pCommitPolsStarks, char* zkevmVerifier, char* execFile, void* zkin, uint64_t N, uint64_t nCols);
    void circom_recursive1_get_commited_pols(void *pCommitPolsStarks, char* zkevmVerifier, char* execFile, void* zkin, uint64_t N, uint64_t nCols);

    // zkin
    // ========================================================================================
    void *zkin_new(void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, unsigned long numRootC, void *pRootC);

    // Save proof
    // ========================================================================================
    void save_proof(void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, char* publicsOutputFile, char* filePrefix);
#endif