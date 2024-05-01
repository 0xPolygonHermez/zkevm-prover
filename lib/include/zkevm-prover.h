#ifndef ZKEVM_API_H
#define ZKEVM_API_H
    #include <stdint.h>

    int zkevm_main(char *configFile, void* pAddress, void** pSMRequests, void* pSMRequestsOut, void *pStarkInfo = nullptr);
    int zkevm_delete_sm_requests(void **pSMRequests);
    int zkevm_arith(void * inputs, int ninputs, void * pAddress);
    int zkevm_arith_req(void* pSMRequests,  void * pAddress);
    int zkevm_binary_req(void* pSMRequests,  void * pAddress);
    int zkevm_memory(void * inputs_, int ninputs, void * pAddress);
    int zkevm_memory_req(void* pSMRequests,  void * pAddress);
    int zkevm_mem_align(void * inputs_, int ninputs, void* pAddress);
    int zkevm_mem_align_req(void* pSMRequests,  void * pAddress);
    
    int zkevm_padding_kk(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut); 
    int zkevm_padding_kk_req(void* pSMRequests,  void * pAddress); 
    int zkevm_padding_kk_bit(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_padding_kk_bit_req(void* pSMRequests,  void * pAddress);
    int zkevm_bits2field_kk(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_bits2field_kk_req(void* pSMRequests,  void * pAddress);
    int zkevm_keccak_f(void * inputs_, int ninputs, void * pAddress);
    int zkevm_keccak_f_req(void* pSMRequests,  void * pAddress);

    int zkevm_padding_sha256(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_padding_sha256_req(void* pSMRequests,  void * pAddress);
    int zkevm_padding_sha256_bit(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_padding_sha256_bit_req(void* pSMRequests,  void * pAddress);
    int zkevm_bits2field_sha256(void * inputs_, int ninputs, void * pAddress, void* pSMRequests, void* pSMRequestsOut);
    int zkevm_bits2field_sha256_req(void* pSMRequests,  void * pAddress);
    int zkevm_sha256_f(void * inputs_, int ninputs, void * pAddress);
    int zkevm_sha256_f_req(void* pSMRequests,  void * pAddress);

    int zkevm_storage_req(void* pSMRequests,  void * pAddress);
    int zkevm_padding_pg_req(void* pSMRequests,  void * pAddress);
    int zkevm_climb_key_req(void* pSMRequests,  void * pAddress);
    int zkevm_poseidon_g_req(void* pSMRequests,  void * pAddress);
    
    // FRI Proof
    // ========================================================================================
    void save_proof(void* pStarkInfo, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, char* publicsOutputFile, char* filePrefix);

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
    void *generic_steps_new();
    void generic_steps_free(void *pGenericSteps);

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

    // Stark Info
    // ========================================================================================
    void *starkinfo_new(char* filename);
    uint64_t get_mapTotalN(void *pStarkInfo);
    void set_mapOffsets(void *pStarkInfo, void *pChelpers);
    uint64_t get_map_offsets(void *pStarkInfo, char *stage, bool flag);
    uint64_t get_map_sections_n(void *pStarkInfo, char *stage);
    void starkinfo_free(void *pStarkInfo);

    // Starks
    // ========================================================================================
    void *starks_new(void *pConfig, char *constPols, bool mapConstPolsFile, char *constantsTree, void *starkInfo, void *cHelpers, void *pAddress);

    void *get_stark_info(void *pStarks);
    void starks_free(void *pStarks);

    void *chelpers_new(char* cHelpers);
    void chelpers_free(void *pChelpers);

    void init_hints();

    void *steps_params_new(void *pStarks, void * pChallenges, void *pSubproofValues, void *pEvals, void *pPublicInputs);
    void *get_steps_params_field(void *pStepsParams, char *name);
    void steps_params_free(void *pStepsParams);
    void extend_and_merkelize(void *pStarks, uint64_t step, void *pParams, void *proof);
    void treesGL_get_root(void *pStarks, uint64_t index, void *root);

    void compute_stage(void *pStarks, uint32_t elementType, uint64_t step, void *pParams, void *pProof, void *pTranscript, void *pChelpersSteps);
    void compute_evals(void *pStarks, void *pParams, void *pProof);

    void *compute_fri_pol(void *pStarks, uint64_t step, void *pParams, void *cHelpersSteps);
    void compute_fri_folding(void *pStarks, void *pProof, void *pFriPol, uint64_t step, void *pChallenge);
    void compute_fri_queries(void *pStarks, void *pProof, uint64_t* friQueries);

    void *get_proof_root(void *pProof, uint64_t stage_id, uint64_t index);

    void *get_vector_pointer(void *pStarks, char *name);
    void resize_vector(void *pVector, uint64_t newSize, bool value);
    void set_bool_vector_value(void *pVector, uint64_t index, bool value);

    void clean_symbols_calculated(void *pStarks);
    void set_symbol_calculated(void *pStarks, uint32_t operand, uint64_t id);

    void calculate_hash(void *pStarks, void *pHhash, void *pBuffer, uint64_t nElements);

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
    void *zkin_new(void* pStarkInfo, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, unsigned long numRootC, void *pRootC);

    // Transcript
    // =================================================================================
    void *transcript_new(uint32_t elementType, uint64_t arity, bool custom);
    void transcript_add(void *pTranscript, void *pInput, uint64_t size);
    void transcript_add_polinomial(void *pTranscript, void *pPolinomial);
    void transcript_free(void *pTranscript, uint32_t elementType);
    void get_challenge(void *pStarks, void *pTranscript, void *pElement);
    void get_permutations(void *pTranscript, uint64_t *res, uint64_t n, uint64_t nBits);

    // Polinomial
    // =================================================================================
    void *polinomial_new(uint64_t degree, uint64_t dim, char* name);
    void *polinomial_get_p_element(void *pPolinomial, uint64_t index);
    void polinomial_free(void *pPolinomial);

    // Poseidon
    // =================================================================================
    void goldilocks_linear_hash(void *pInput, void *pOutput);

#endif