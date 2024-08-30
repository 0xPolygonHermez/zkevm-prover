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
    void fri_proof_set_subproofvalues(void *pFriProof, void *pParams);
    void fri_proof_free(void *pFriProof);

    // SetupCtx
    // ========================================================================================
    void *setup_ctx_new(void* p_stark_info, void* p_expression_bin, void* p_const_pols);
    void setup_ctx_free(void *pSetupCtx);

    // Stark Info
    // ========================================================================================
    void *stark_info_new(char* filename);
    uint64_t get_map_total_n(void *pStarkInfo);
    uint64_t get_map_offsets(void *pStarkInfo, char *stage, bool flag);
    void stark_info_free(void *pStarkInfo);

    // Const Pols
    // ========================================================================================
    void *const_pols_new(char* filename, void *pStarkInfo);
    void const_pols_free(void *pConstPols);

    // Expressions Bin
    // ========================================================================================
    void *expressions_bin_new(char* filename);
    void expressions_bin_free(void *pExpressionsBin);

    // ExpressionsCtx
    // ========================================================================================
    void *expressions_ctx_new(void *pSetupCtx);
    bool verify_constraints(void *pExpressionsCtx, void*pParams, uint64_t step);
    void *get_fri_pol(void *pExpressionsCtx, void *pParams);
    void* get_hint_ids_by_name(void *pExpressionsCtx, char* hintName);
    void *get_hint_field(void *pExpressionsCtx, void*pParams, uint64_t hintId, char* hintFieldName, bool dest);
    void set_hint_field(void *pExpressionsCtx,  void*pParams, void *values, uint64_t hintId, char* hintFieldName);
    void expressions_ctx_free(void *pExpressionsCtx);

    void set_commit_calculated(void *pExpressionsCtx, uint64_t id);
    void can_stage_be_calculated(void *pExpressionsCtx, uint64_t step);
    void can_impols_be_calculated(void *pExpressionsCtx, uint64_t step);

    // StepsParams
    // ========================================================================================
    void *init_params(void* ptr, void* public_inputs, void* challenges, void* evals, void* subproofValues);

    // Starks
    // ========================================================================================
    void *starks_new(void *pConfig, void *pSetupCtx, void *pExpressionsCtx);
    void *starks_new_default(void *pSetupCtx, void *pExpressionsCtx);
    void starks_free(void *pStarks);

    void extend_and_merkelize(void *pStarks, uint64_t step, void *pParams, void *proof);
    void treesGL_get_root(void *pStarks, uint64_t index, void *root);

    void calculate_quotient_polynomial(void *pExpressionsCtx, void* pParams);
    void calculate_impols_expressions(void *pExpressionsCtx, void* pParams, uint64_t step);

    void commit_stage(void *pStarks, uint32_t elementType, uint64_t step, void *pParams, void *pProof);
    void compute_evals(void *pStarks, void *pParams, void *pProof);

    void *compute_fri_pol(void *pStarks, uint64_t step, void *pParams);
    void compute_fri_folding(void *pStarks, uint64_t step, void *pParams, void *pChallenge,  void *pProof);
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

    // Global constraints
    // =================================================================================
    bool verify_global_constraints(char *globalInfoFile, char *globalConstraintsBinFile, void *publics, void *pProofs, uint64_t nProofs);
#endif