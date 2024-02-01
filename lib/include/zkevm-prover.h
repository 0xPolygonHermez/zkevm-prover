#ifndef ZKEVM_API_H
#define ZKEVM_API_H
    #include <stdint.h>

    int zkevm_main(char *configFile, void* pAddress);

    // FFI functions

    // ZkevmSteps
    // ========================================================================================
    void *zkevm_steps_new();
    void zkevm_steps_free(void *pZkevmSteps);


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
#endif