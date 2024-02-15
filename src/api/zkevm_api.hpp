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

    void step2prev_parser_first_avx(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step2prev_parser_first_avx512(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step2prev_first_parallel(void *pSteps, void *pParams, uint64_t nrows);
    
    void step3prev_parser_first_avx(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step3prev_parser_first_avx512(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step3prev_first_parallel(void *pSteps, void *pParams, uint64_t nrows);

    void step3_parser_first_avx(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step3_parser_first_avx512(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step3_first_parallel(void *pSteps, void *pParams, uint64_t nrows);

    void step42ns_parser_first_avx(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step42ns_parser_first_avx512(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step42ns_first_parallel(void *pSteps, void *pParams, uint64_t nrows);

    void step52ns_parser_first_avx(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step52ns_parser_first_avx512(void *pSteps, void *pParams, uint64_t nrows, uint64_t nrowsBatch);
    void step52ns_first_parallel(void *pSteps, void *pParams, uint64_t nrows);

    // FRIProof
    // ========================================================================================
    void *fri_proof_new(void *pStarks);
    void *fri_proof_get_root(void *pFriProof, uint64_t root_index, uint64_t root_subindex);
    void *fri_proof_get_tree_root(void *pFriProof, uint64_t tree_index, uint64_t root_index);
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

    // void *transpose_h1_h2_columns(void *pStarks, void *pAddress, uint64_t *numCommited, void *pBuffer);
    // void transpose_h1_h2_rows(void *pStarks, void *pAddress, uint64_t *numCommited, void *transPols);
    // void *transpose_z_columns(void *pStarks, void *pAddress, uint64_t *numCommited, void *pBuffer);
    // void transpose_z_rows(void *pStarks, void *pAddress, uint64_t *numCommited, void *transPols);
    // void evmap(void *pStarks, void *pAddress, void *evals, void *LEv, void *LpEv);

    void *steps_params_new(void *pStarks, void * pChallenges, void *pEvals, void *pXDivXSubXi, void *pXDivXSubWXi, void *pPublicInputs);
    void steps_params_free(void *pStepsParams);
    void extend_and_merkelize(void *pStarks, uint64_t step, void *pParams, void *proof);
    // void tree_merkelize(void *pStarks, uint64_t index);
    // void tree_get_root(void *pStarks, uint64_t index, void *root);
    // void extend_pol(void *pStarks, uint64_t step);
    // void *get_pbuffer(void *pStarks);

    void calculate_h1_h2(void *pStarks, void *pParams);
    void calculate_z(void *pStarks, void *pParams);
    void calculate_expressions(void *pStarks, char* step, uint64_t nrowsStepBatch, void *pSteps, void *pParams, uint64_t n);
    void compute_q(void *pStarks, void *pParams, void *pProof);
    void compute_evals(void *pStarks, void *pParams, void *pProof);

    void *compute_fri_pol(void *pStarks, void *pParams, void *steps, uint64_t nrowsStepBatch);
    void compute_fri_folding(void *pStarks, void *pProof, void *pFriPol, uint64_t step, void *challenge);
    void compute_fri_queries(void *pStarks, void *pProof, void *pFriPol, uint64_t* friQueries);

    // void calculate_exps_2ns(void *pStarks, void  *pQq1, void *pQq2);
    // void calculate_lev_lpev(void *pStarks, void *pLEv, void *pLpEv, void *pXis, void *pWxis, void *pC_w, void *pChallenges);
    // void calculate_xdivxsubxi(void *pStarks, uint64_t extendBits, void *xi, void *wxi, void *challenges, void *xDivXSubXi, void *xDivXSubWXi);
    // void finalize_proof(void *pStarks, void *proof, void *transcript, void *evals, void *root0, void *root1, void *root2, void *root3);

    uint64_t get_num_rows_step_batch(void *pStarks);

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
    void *zkin_new(void* pStarks, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, unsigned long numRootC, void *pRootC);

    // FRI Proof
    // ========================================================================================
    void save_proof(void* pStarks, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, char* publicsOutputFile, char* filePrefix);

    // Transcript
    // =================================================================================
    void *transcript_new();
    void transcript_add(void *pTranscript, void *pInput, uint64_t size);
    void transcript_add_polinomial(void *pTranscript, void *pPolinomial);
    void transcript_get_field(void *pTranscript, void *pOutput);
    void transcript_free(void *pTranscript);
    void get_challenges(void *pTranscript, void *pPolinomial, uint64_t nChallenges, uint64_t index);
    void get_permutations(void *pTranscript, uint64_t *res, uint64_t n, uint64_t nBits);

    // Polinomial
    // =================================================================================
    void *polinomial_new(uint64_t degree, uint64_t dim, char* name);
    void *polinomial_new_void();
    void *polinomial_get_address(void *pPolinomial);
    void *polinomial_get_p_element(void *pPolinomial, uint64_t index);
    void polinomial_free(void *pPolinomial);

    // CommitPols
    // =================================================================================
    void *commit_pols_new(void * pAddress, uint64_t degree);
    void commit_pols_free(void *pCommitPols);
#endif