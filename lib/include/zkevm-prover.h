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
#endif