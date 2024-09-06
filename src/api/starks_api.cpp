#include "config.hpp"
#include "main.hpp"
#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "verify_constraints.hpp"
#include "hints.hpp"
#include "global_constraints.hpp"
#include <filesystem>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

void save_challenges(void *pChallenges, char* globalInfoFile, char *fileDir) {

    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    uint64_t nStages = globalInfo["numChallenges"].size();

    Goldilocks::Element *challenges = (Goldilocks::Element *)pChallenges;
    
    uint64_t c = 0;

    json challengesJson;
    challengesJson["challenges"] = json::array();
    for(uint64_t i = 0; i < nStages; ++i) {
        challengesJson["challenges"][i] = json::array();
        for(uint64_t j = 0; j < globalInfo["numChallenges"][i]; ++j) {
            challengesJson["challenges"][i][j] = json::array();
            for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                challengesJson["challenges"][i][j][k] = Goldilocks::toString(challenges[c++]);
            }
        }
    }

    challengesJson["challenges"][nStages] = json::array();
    challengesJson["challenges"][nStages][0] = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        challengesJson["challenges"][nStages][0][k] = Goldilocks::toString(challenges[c++]);
    }
    
    challengesJson["challenges"][nStages + 1] = json::array();
    challengesJson["challenges"][nStages + 1][0] = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        challengesJson["challenges"][nStages + 1][0][k] = Goldilocks::toString(challenges[c++]);
    }

    challengesJson["challenges"][nStages + 2] = json::array();
    challengesJson["challenges"][nStages + 2][0] = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        challengesJson["challenges"][nStages + 2][0][k] = Goldilocks::toString(challenges[c++]);
    }
    
    challengesJson["challenges"][nStages + 2][1] = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        challengesJson["challenges"][nStages + 2][1][k] = Goldilocks::toString(challenges[c++]);
    }

    challengesJson["challengesFRISteps"] = json::array();
    for(uint64_t i = 0; i < globalInfo["stepsFRI"].size() + 1; ++i) {
        challengesJson["challengesFRISteps"][i] = json::array();
        for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
            challengesJson["challengesFRISteps"][i][k] = Goldilocks::toString(challenges[c++]);
        }
    }
    
    json2file(challengesJson, string(fileDir) + "/challenges.json");
}


void save_publics(unsigned long numPublicInputs, void *pPublicInputs, char *fileDir) {

    Goldilocks::Element* publicInputs = (Goldilocks::Element *)pPublicInputs;

    // Generate publics
    json publicStarkJson;
    for (uint64_t i = 0; i < numPublicInputs; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    // save publics to filestarks
    json2file(publicStarkJson, string(fileDir) + "/publics.json");
}



void save_proof(uint64_t proof_id, void *pStarkInfo, void *pFriProof, char *fileDir)
{
    auto friProof = (FRIProof<Goldilocks::Element> *)pFriProof;

    nlohmann::ordered_json jProof = friProof->proof.proof2json();

    nlohmann::ordered_json zkin = proof2zkinStark(jProof, *(StarkInfo *)pStarkInfo);

    std::filesystem::create_directory(string(fileDir) + "/zkin");
    std::filesystem::create_directory(string(fileDir) + "/proofs");

    // Save output to file
    json2file(zkin, string(fileDir) + "/zkin/proof_" + to_string(proof_id) + "_zkin.json");
    
    // Save proof to file
    json2file(jProof, string(fileDir) + "/proofs/proof_" + to_string(proof_id) + ".json");
}


void *fri_proof_new(void *pSetupCtx)
{
    SetupCtx setupCtx = *(SetupCtx *)pSetupCtx;
    FRIProof<Goldilocks::Element> *friProof = new FRIProof<Goldilocks::Element>(setupCtx.starkInfo);

    return friProof;
}


void *fri_proof_get_tree_root(void *pFriProof, uint64_t tree_index, uint64_t root_index)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    return &friProof->proof.fri.trees[tree_index].root[root_index];
}

void fri_proof_set_subproofvalues(void *pFriProof, void *pParams)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    auto params = *(StepsParams *)pParams;
    friProof->proof.setSubproofValues(params.subproofValues);
}

void fri_proof_free(void *pFriProof)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    delete friProof;
}

// SetupCtx
// ========================================================================================

void *setup_ctx_new(void* p_stark_info, void* p_expression_bin, void* p_const_pols) {
    SetupCtx *setupCtx = new SetupCtx(*(StarkInfo*)p_stark_info, *(ExpressionsBin*)p_expression_bin, *(ConstPols *)p_const_pols);
    return setupCtx;
}

void* get_hint_ids_by_name(void *pSetupCtx, char* hintName)
{
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx;

    VecU64Result hintIds =  setupCtx->expressionsBin.getHintIdsByName(string(hintName));
    return new VecU64Result(hintIds);
}

void setup_ctx_free(void *pSetupCtx) {
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx;
    delete setupCtx;
}

// StarkInfo
// ========================================================================================
void *stark_info_new(char *filename)
{
    auto starkInfo = new StarkInfo(filename);

    return starkInfo;
}

uint64_t get_map_total_n(void *pStarkInfo)
{
    return ((StarkInfo *)pStarkInfo)->mapTotalN;
}

uint64_t get_map_offsets(void *pStarkInfo, char *stage, bool flag)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    return starkInfo->mapOffsets[std::make_pair(stage, flag)];
}

void stark_info_free(void *pStarkInfo)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    delete starkInfo;
}

// Const Pols
// ========================================================================================
void *const_pols_new(char* filename, void *pStarkInfo) 
{
    auto const_pols = new ConstPols(*(StarkInfo *)pStarkInfo, filename);

    return const_pols;
}

void const_pols_free(void *pConstPols)
{
    auto constPols = (ConstPols *)pConstPols;
    delete constPols;
}

// Expressions Bin
// ========================================================================================
void *expressions_bin_new(char* filename)
{
    auto expressionsBin = new ExpressionsBin(filename);

    return expressionsBin;
};
void expressions_bin_free(void *pExpressionsBin)
{
    auto expressionsBin = (ExpressionsBin *)pExpressionsBin;
    delete expressionsBin;
};

// StepsParams
// ========================================================================================
void *init_params(void* ptr, void* public_inputs, void* challenges, void* evals, void* subproofValues) {
    StepsParams *params = new StepsParams {
        pols : (Goldilocks::Element *)ptr,
        publicInputs : (Goldilocks::Element *)public_inputs,
        challenges : (Goldilocks::Element *)challenges,
        subproofValues : (Goldilocks::Element *)subproofValues,
        evals : (Goldilocks::Element *)evals,
        prover_initialized : true,
    };
    return params;
}

void *get_fri_pol(void *pSetupCtx, void *pParams)
{
    SetupCtx setupCtx = *(SetupCtx *)pSetupCtx;
    auto params = *(StepsParams *)pParams;
    
    return &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
}

void *get_hint_field(void *pSetupCtx, void* pParams, uint64_t hintId, char *hintFieldName, bool dest, bool inverse, bool printExpression) 
{
    HintFieldInfo hintFieldInfo = getHintField(*(SetupCtx *)pSetupCtx, *(StepsParams *)pParams, hintId, string(hintFieldName), dest, inverse, printExpression);
    return new HintFieldInfo(hintFieldInfo);
}

uint64_t set_hint_field(void *pSetupCtx, void* pParams, void *values, uint64_t hintId, char * hintFieldName) 
{
    return setHintField(*(SetupCtx *)pSetupCtx, *(StepsParams *)pParams, (Goldilocks::Element *)values, hintId, string(hintFieldName));
}

void *verify_constraints(void *pSetupCtx, void* pParams)
{
    ConstraintsResults *constraintsInfo = verifyConstraints(*(SetupCtx *)pSetupCtx, *(StepsParams *)pParams);
    return constraintsInfo;
}

void params_free(void* pParams) {
    StepsParams *params = (StepsParams *)pParams;
    delete params;
}


// Starks
// ========================================================================================

void *starks_new(void *pSetupCtx)
{
    return new Starks<Goldilocks::Element>(*(SetupCtx *)pSetupCtx);
}

void starks_free(void *pStarks)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    delete starks;
}

void extend_and_merkelize(void *pStarks, uint64_t step, void *pParams, void *pProof)
{
    auto starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->ffi_extend_and_merkelize(step, *(StepsParams *)pParams, (FRIProof<Goldilocks::Element> *)pProof);
}

void treesGL_get_root(void *pStarks, uint64_t index, void *dst)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;

    starks->ffi_treesGL_get_root(index, (Goldilocks::Element *)dst);
}

void calculate_fri_polynomial(void *pStarks, void* pParams)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateFRIPolynomial(*(StepsParams *)pParams);
}


void calculate_quotient_polynomial(void *pStarks, void* pParams)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateQuotientPolynomial(*(StepsParams *)pParams);
}

void calculate_impols_expressions(void *pStarks, void* pParams, uint64_t step)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateImPolsExpressions(step, *(StepsParams *)pParams);
}

void commit_stage(void *pStarks, uint32_t elementType, uint64_t step, void *pParams, void *pProof) {
    // type == 1 => Goldilocks
    // type == 2 => BN128
    switch (elementType)
    {
    case 1:
        ((Starks<Goldilocks::Element> *)pStarks)->commitStage(step, *(StepsParams *)pParams, *(FRIProof<Goldilocks::Element> *)pProof);
        break;
    default:
        cerr << "Invalid elementType: " << elementType << endl;
        break;
    }
}


void compute_evals(void *pStarks, void *pParams, void *pProof)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeEvals(*(StepsParams *)pParams, *(FRIProof<Goldilocks::Element> *)pProof);
}

void prepare_fri_pol(void *pStarks, void *pParams)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->prepareFRIPolynomial(*(StepsParams *)pParams);
}

void compute_fri_folding(void *pStarks, uint64_t step, void *pParams, void *pChallenge,  void *pProof)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIFolding(step, *(StepsParams *)pParams, (Goldilocks::Element *)pChallenge, *(FRIProof<Goldilocks::Element> *)pProof);
}

void compute_fri_queries(void *pStarks, void *pProof, uint64_t *friQueries)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIQueries(*(FRIProof<Goldilocks::Element> *)pProof, friQueries);
}

void calculate_hash(void *pStarks, void *pHhash, void *pBuffer, uint64_t nElements)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateHash((Goldilocks::Element *)pHhash, (Goldilocks::Element *)pBuffer, nElements);
}

// Transcript
// =================================================================================
void *transcript_new(uint32_t elementType, uint64_t arity, bool custom)
{
    // type == 1 => Goldilocks
    // type == 2 => BN128
    switch (elementType)
    {
    case 1:
        return new TranscriptGL(arity, custom);
    case 2:
        return new TranscriptBN128(arity, custom);
    default:
        return NULL;
    }
}

void transcript_add(void *pTranscript, void *pInput, uint64_t size)
{
    auto transcript = (TranscriptGL *)pTranscript;
    auto input = (Goldilocks::Element *)pInput;

    transcript->put(input, size);
}

void transcript_add_polinomial(void *pTranscript, void *pPolinomial)
{
    auto transcript = (TranscriptGL *)pTranscript;
    auto pol = (Polinomial *)pPolinomial;

    for (uint64_t i = 0; i < pol->degree(); i++)
    {
        transcript->put(pol->operator[](i), pol->dim());
    }
}

void transcript_free(void *pTranscript, uint32_t elementType)
{
    switch (elementType)
    {
    case 1:
        delete (TranscriptGL *)pTranscript;
        break;
    case 2:
        delete (TranscriptBN128 *)pTranscript;
        break;
    }
}

void get_challenge(void *pStarks, void *pTranscript, void *pElement)
{
    TranscriptGL *transcript = (TranscriptGL *)pTranscript;
    ((Starks<Goldilocks::Element> *)pStarks)->getChallenge(*transcript, *(Goldilocks::Element *)pElement);
}

void get_permutations(void *pTranscript, uint64_t *res, uint64_t n, uint64_t nBits)
{
    TranscriptGL *transcript = (TranscriptGL *)pTranscript;
    transcript->getPermutations(res, n, nBits);
}

// Global constraints
// =================================================================================
bool verify_global_constraints(char *globalInfoFile, char *globalConstraintsBinFile, void *publics, void *pProofs, uint64_t nProofs) {
    
    FRIProof<Goldilocks::Element> **proofs = (FRIProof<Goldilocks::Element> **)pProofs;

    return verifyGlobalConstraints(string(globalInfoFile), string(globalConstraintsBinFile), (Goldilocks::Element *)publics, proofs, nProofs);
}

// Debug functions
// =================================================================================  
void *print_by_name(void *pSetupCtx, void *pParams, char* name, uint64_t *lengths, uint64_t first_value, uint64_t last_value, bool return_values) {
    HintFieldInfo hintFieldInfo = printByName(*(SetupCtx *)pSetupCtx, *(StepsParams *)pParams, string(name), lengths, first_value, last_value, return_values);
    return new HintFieldInfo(hintFieldInfo);
}

void print_expression(void *pSetupCtx, void* pol, uint64_t dim, uint64_t first_value, uint64_t last_value) {
    printExpression((Goldilocks::Element *)pol, dim, first_value, last_value);
}