#ifndef LIB_API_H
#define LIB_API_H
    #include <stdint.h>

    // Save Proof
    // ========================================================================================
    void save_challenges(void *pChallenges, char* globalInfoFile, char *fileDir);
    void save_publics(unsigned long numPublicInputs, void *pPublicInputs, char *fileDir);
    void save_proof(uint64_t proof_id, void *pStarkInfo, void *pFriProof, char *fileDir);

    // FRIProof
    // ========================================================================================
    void *fri_proof_new(void *pSetupCtx);
    void *fri_proof_get_tree_root(void *pFriProof, uint64_t tree_index, uint64_t root_index);
    void fri_proof_set_subproofvalues(void *pFriProof, void *subproofValues);
    void *fri_proof_get_zkinproof(uint64_t proof_id, void *pFriProof, void* pPublics, void* pChallenges, void *pStarkInfo, char* globalInfoFile, char *fileDir);
    void fri_proof_free_zkinproof(void *pZkinProof);
    void fri_proof_free(void *pFriProof);

    // SetupCtx
    // ========================================================================================
    void *setup_ctx_new(void* p_stark_info, void* p_expression_bin, void* p_const_pols);
    void *get_hint_ids_by_name(void *pSetupCtx, char* hintName);
    void setup_ctx_free(void *pSetupCtx);

    // Stark Info
    // ========================================================================================
    void *stark_info_new(char* filename);
    uint64_t get_stark_info_n(void *pStarkInfo);
    uint64_t get_stark_info_n_publics(void *pStarkInfo);
    uint64_t get_map_total_n(void *pStarkInfo);
    uint64_t get_map_offsets(void *pStarkInfo, char *stage, bool flag);
    void stark_info_free(void *pStarkInfo);

    // Const Pols
    // ========================================================================================
    void *const_pols_new(char* filename, void *pStarkInfo);
    void *const_pols_with_tree_new(char* filename, char* treeFilename, void *pStarkInfo);
    void const_pols_free(void *pConstPols);

    // Expressions Bin
    // ========================================================================================
    void *expressions_bin_new(char* filename);
    void expressions_bin_free(void *pExpressionsBin);

    // Hints
    // ========================================================================================
    void *get_hint_field(void *pSetupCtx, void* buffer, void* public_inputs, void* challenges, void* subproofValues, void* evals, uint64_t hintId, char* hintFieldName, bool dest, bool inverse, bool print_expression);
    uint64_t set_hint_field(void *pSetupCtx, void* buffer, void* subproofValues, void *values, uint64_t hintId, char* hintFieldName);

    // Starks
    // ========================================================================================
    void *starks_new(void *pSetupCtx);
    void starks_free(void *pStarks);

    void extend_and_merkelize(void *pStarks, uint64_t step, void *buffer, void *proof, void *pBuffHelper);
    void treesGL_get_root(void *pStarks, uint64_t index, void *root);

    void *calculate_xdivxsub(void *pStarks, void* xiChallenge, void *xDivXSub);
    void *get_fri_pol(void *pSetupCtx, void *buffer);

    void calculate_fri_polynomial(void *pStarks, void* buffer, void* public_inputs, void* challenges, void* subproofValues, void* evals, void *xDivXSub);
    void calculate_quotient_polynomial(void *pStarks, void* buffer, void* public_inputs, void* challenges, void* subproofValues, void* evals);
    void calculate_impols_expressions(void *pStarks, uint64_t step, void* buffer, void* public_inputs, void* challenges, void* subproofValues, void* evals);

    void commit_stage(void *pStarks, uint32_t elementType, uint64_t step, void *buffer, void *pProof, void *pBuffHelper);
    
    void compute_lev(void *pStarks, void *xiChallenge, void* LEv);
    void compute_evals(void *pStarks, void *buffer, void *LEv, void *evals, void *pProof);

    void compute_fri_folding(void *pStarks, void *pProof, uint64_t step, void *buffer, void *pChallenge);
    void compute_fri_queries(void *pStarks, void *pProof, uint64_t* friQueries);

    void calculate_hash(void *pStarks, void *pHhash, void *pBuffer, uint64_t nElements);
    
    // Transcript
    // =================================================================================
    void *transcript_new(uint32_t elementType, uint64_t arity, bool custom);
    void transcript_add(void *pTranscript, void *pInput, uint64_t size);
    void transcript_add_polinomial(void *pTranscript, void *pPolinomial);
    void transcript_free(void *pTranscript, uint32_t elementType);
    void get_challenge(void *pStarks, void *pTranscript, void *pElement);
    void get_permutations(void *pTranscript, uint64_t *res, uint64_t n, uint64_t nBits);

    // Verify constraints
    // =================================================================================
    void *verify_constraints(void *pSetupCtx, void* buffer, void* public_inputs, void* challenges, void* subproofValues, void* evals);
    bool verify_global_constraints(char *globalConstraintsBinFile, void *publics, void **airgroupValues);

    // Debug functions
    // =================================================================================
    void *print_by_name(void *pSetupCtx, void* buffer, void* public_inputs, void* challenges, void* subproofValues, char* name, uint64_t *lengths, uint64_t first_value, uint64_t last_value, bool return_values);
    void print_row(void *pSetupCtx, void *buffer, uint64_t stage, uint64_t row);
    void print_expression(void *pSetupCtx, void* pol, uint64_t dim, uint64_t first_value, uint64_t last_value);

    // Recursive proof
    // =================================================================================
    void *gen_recursive_proof(void *pSetupCtx, void* pAddress, void* pPublicInputs, char *proof_file);
    void *get_zkin_ptr(char *zkin_file);
    void *public2zkin(void *pZkin, void* pPublics, char* globalInfoFile, uint64_t airgroupId, bool isAggregated);
    void *add_recursive2_verkey(void *pZkin, char* recursive2VerKeyFilename);
    void *join_zkin_recursive2(char* globalInfoFile, void* pPublics, void* pChallenges, void *zkin1, void *zkin2, void *starkInfoRecursive2);
    void *join_zkin_final(void* pPublics, void* pChallenges, char* globalInfoFile, void **zkinRecursive2, void **starkInfoRecursive2);

#endif