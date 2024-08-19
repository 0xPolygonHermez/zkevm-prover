#include "config.hpp"
#include "circom.hpp"
#include "main.hpp"
#include "main.recursive1.hpp"
#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

void save_proof(void *pStarkInfo, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, char *publicsOutputFile, char *filePrefix)
{
    auto friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    Goldilocks::Element *publicInputs = (Goldilocks::Element *)pPublicInputs;

    // Generate publics
    json publicStarkJson;
    for (uint64_t i = 0; i < numPublicInputs; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    nlohmann::ordered_json jProofRecursive1 = friProof->proofs.proof2json();
    nlohmann::ordered_json zkinRecursive1 = proof2zkinStark(jProofRecursive1, *(StarkInfo *)pStarkInfo);
    zkinRecursive1["publics"] = publicStarkJson;

    // save publics to filestarks
    json2file(publicStarkJson, publicsOutputFile);

    // Save output to file
    if (config.saveOutputToFile)
    {
        json2file(zkinRecursive1, string(filePrefix) + "batch_proof.output.json");
    }
    // Save proof to file
    if (config.saveProofToFile)
    {
        jProofRecursive1["publics"] = publicStarkJson;
        json2file(jProofRecursive1, string(filePrefix) + "batch_proof.proof.json");
    }
}

void *fri_proof_new(void *pStarks)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    FRIProof<Goldilocks::Element> *friProof = new FRIProof<Goldilocks::Element>(starks->starkInfo);

    return friProof;
}

void *fri_proof_get_root(void *pFriProof, uint64_t root_index, uint64_t root_subindex)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    return &friProof->proofs.roots[root_index][root_subindex];
}

void *fri_proof_get_tree_root(void *pFriProof, uint64_t tree_index, uint64_t root_index)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    return &friProof->proofs.fri.trees[tree_index].root[root_index];
}

void fri_proof_free(void *pFriProof)
{
    FRIProof<Goldilocks::Element> *friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    delete friProof;
}

void *config_new(char *filename)
{
    Config *config = new Config();
    json configJson;
    file2json(filename, configJson);
    config->load(configJson);

    return config;
}

void config_free(void *pConfig)
{
    Config *config = (Config *)pConfig;
    delete config;
}

void *starkinfo_new(char *filename)
{
    auto starkInfo = new StarkInfo(filename);

    return starkInfo;
}

uint64_t get_mapTotalN(void *pStarkInfo)
{
    return ((StarkInfo *)pStarkInfo)->mapTotalN;
}

uint64_t get_map_offsets(void *pStarkInfo, char *stage, bool flag)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    return starkInfo->mapOffsets[std::make_pair(stage, flag)];
}

uint64_t get_map_sections_n(void *pStarkInfo, char *stage)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    return starkInfo->mapSectionsN[stage];
}

void starkinfo_free(void *pStarkInfo)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    delete starkInfo;
}

void *starks_new(void *pConfig, void *starkInfo, void *cHelpers, void *constPols, void *pAddress)
{
    return new Starks<Goldilocks::Element>(*(Config *)pConfig, pAddress, *(StarkInfo *)starkInfo, *(CHelpers *)cHelpers, *(ConstPols<Goldilocks::Element> *) constPols, false);
}

void *starks_new_default(void *starkInfo, void *cHelpers, void *constPols, void *pAddress)
{
    Config configLocal;
    configLocal.runFileGenBatchProof = true; //to force function generateProof to return true

    return new Starks<Goldilocks::Element>(configLocal, pAddress, *(StarkInfo *)starkInfo, *(CHelpers *)cHelpers, *(ConstPols<Goldilocks::Element> *) constPols, false);
}


void *get_stark_info(void *pStarks)
{
    return &((Starks<Goldilocks::Element> *)pStarks)->starkInfo;
}

void starks_free(void *pStarks)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    delete starks;
}

void *chelpers_new(char *cHelpers)
{
    return new CHelpers(cHelpers);
}

void chelpers_free(void *pChelpers)
{
    CHelpers *cHelpers = (CHelpers *)pChelpers;
    delete cHelpers;
}

void init_hints()
{
    HintHandlerBuilder::registerBuilder(H1H2HintHandler::getName(), std::make_unique<H1H2HintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GProdHintHandler::getName(), std::make_unique<GProdHintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GSumHintHandler::getName(), std::make_unique<GSumHintHandlerBuilder>());
}

void *steps_params_new(void *pPols, void*pConstPols, void *pChallenges, void *pSubproofValues, void *pPublicInputs)
{
    Goldilocks::Element *pols = (Goldilocks::Element *)pPols;
    Goldilocks::Element *constPols = (Goldilocks::Element *)pConstPols;
    Goldilocks::Element *challenges = (Goldilocks::Element *)pChallenges;
    Goldilocks::Element *subproofValues = (Goldilocks::Element *)pSubproofValues;
    Goldilocks::Element *publicInputs = (Goldilocks::Element *)pPublicInputs;

    StepsParams *params = new StepsParams{
        pols : pols,
        constPols : constPols,
        constPolsExtended : nullptr,
        challenges : challenges,
        subproofValues : subproofValues,
        evals : nullptr,
        zi : nullptr,
        publicInputs : publicInputs,
    };

    return params;
}

void steps_params_free(void *pStepsParams)
{
    StepsParams *stepsParams = (StepsParams *)pStepsParams;

    delete stepsParams;
}

void extend_and_merkelize(void *pStarks, uint64_t step, void *pParams, void *pProof)
{
    auto starks = (Starks<Goldilocks::Element> *)pStarks;
    auto params = (StepsParams *)pParams;
    auto proof = (FRIProof<Goldilocks::Element> *)pProof;

    starks->ffi_extend_and_merkelize(step, params, proof);
}

void treesGL_get_root(void *pStarks, uint64_t index, void *dst)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;

    starks->ffi_treesGL_get_root(index, (Goldilocks::Element *)dst);
}

void calculate_quotient_polynomial(void *pStarks, void *pParams, void *pChelpersSteps)
{
     Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
     starks->calculateQuotientPolynomial(*(StepsParams *)pParams, *(CHelpersSteps *)pChelpersSteps);
}

void calculate_impols_expressions(void* pStarks, uint64_t step, void *pParams, void *pChelpersSteps)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateImPolsExpressions(step, *(StepsParams *)pParams, *(CHelpersSteps *)pChelpersSteps);
}

void compute_stage_expressions(void *pStarks, uint32_t elementType, uint64_t step, void *pParams, void *pProof, void *pChelpersSteps)
{
    // type == 1 => Goldilocks
    // type == 2 => BN128
    switch (elementType)
    {
    case 1:
        ((Starks<Goldilocks::Element> *)pStarks)->computeStageExpressions(step, *(StepsParams *)pParams, *(FRIProof<Goldilocks::Element> *)pProof, *(CHelpersSteps *)pChelpersSteps);
        break;
    default:
        cerr << "Invalid elementType: " << elementType << endl;
        break;
    }
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

void compute_fri_pol(void *pStarks, uint64_t step, void *pParams, void *cHelpersSteps)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIPol(step, *(StepsParams *)pParams, *(CHelpersSteps *)cHelpersSteps);
}

void *get_fri_pol(void *pStarks, void *pParams)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    StepsParams *params = (StepsParams *)pParams;
    return &params->pols[starks->starkInfo.mapOffsets[std::make_pair("f", true)]];
}

void compute_fri_folding(void *pStarks, void *pProof, void *pFriPol, uint64_t step, void *pChallenge)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIFolding(*(FRIProof<Goldilocks::Element> *)pProof, (Goldilocks::Element *)pFriPol, step, (Goldilocks::Element *)pChallenge);
}

void compute_fri_queries(void *pStarks, void *pProof, uint64_t *friQueries)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->computeFRIQueries(*(FRIProof<Goldilocks::Element> *)pProof, friQueries);
}

void *get_proof_root(void *pProof, uint64_t stage_id, uint64_t index)
{
    FRIProof<Goldilocks::Element> *proof = (FRIProof<Goldilocks::Element> *)pProof;

    return &proof->proofs.roots[stage_id][index];
}

void resize_vector(void *pVector, uint64_t newSize, bool value)
{
    std::vector<bool> *vector = (std::vector<bool> *)pVector;
    vector->resize(newSize, value);
}

void set_bool_vector_value(void *pVector, uint64_t index, bool value)
{
    std::vector<bool> *vector = (std::vector<bool> *)pVector;
    vector->at(index) = value;
}

void calculate_hash(void *pStarks, void *pHhash, void *pBuffer, uint64_t nElements)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateHash((Goldilocks::Element *)pHhash, (Goldilocks::Element *)pBuffer, nElements);
}

void *const_pols_new(void *pStarkInfo, char* constPolsFile) {
    return new ConstPols<Goldilocks::Element>(*(StarkInfo *)pStarkInfo, constPolsFile);
}

void *const_pols_new(void *pStarkInfo, char* constPolsFile, char* constTreeFile) {
    return new ConstPols<Goldilocks::Element>(*(StarkInfo *)pStarkInfo, constPolsFile, constTreeFile);
}

void const_pols_free(void *pConstPols) {
    ConstPols<Goldilocks::Element> *constPols = (ConstPols<Goldilocks::Element> *)pConstPols;
    delete constPols;
}


void *commit_pols_starks_new(void *pAddress, uint64_t degree, uint64_t nCommitedPols)
{
    return new CommitPolsStarks(pAddress, degree, nCommitedPols);
}

void commit_pols_starks_free(void *pCommitPolsStarks)
{
    CommitPolsStarks *commitPolsStarks = (CommitPolsStarks *)pCommitPolsStarks;
    delete commitPolsStarks;
}

void circom_get_commited_pols(void *pCommitPolsStarks, char *zkevmVerifier, char *execFile, void *zkin, uint64_t N, uint64_t nCols)
{
    nlohmann::json *zkinJson = (nlohmann::json *)zkin;
    Circom::getCommitedPols((CommitPolsStarks *)pCommitPolsStarks, zkevmVerifier, execFile, *zkinJson, N, nCols);
}

void circom_recursive1_get_commited_pols(void *pCommitPolsStarks, char *zkevmVerifier, char *execFile, void *zkin, uint64_t N, uint64_t nCols)
{
    nlohmann::json *zkinJson = (nlohmann::json *)zkin;
    CircomRecursive1::getCommitedPols((CommitPolsStarks *)pCommitPolsStarks, zkevmVerifier, execFile, *zkinJson, N, nCols);
}

void *zkin_new(void *pStarkInfo, void *pFriProof, unsigned long numPublicInputs, void *pPublicInputs, unsigned long numRootC, void *pRootC)
{
    auto friProof = (FRIProof<Goldilocks::Element> *)pFriProof;
    auto publicInputs = (Goldilocks::Element *)pPublicInputs;
    auto rootC = (Goldilocks::Element *)pRootC;

    // Generate publics
    json publicStarkJson;
    for (uint64_t i = 0; i < numPublicInputs; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    json xrootC;
    for (uint64_t i = 0; i < numRootC; i++)
    {
        xrootC[i] = Goldilocks::toString(rootC[i]);
    }

    nlohmann::ordered_json *jProof = new nlohmann::ordered_json();
    nlohmann::json *zkin = new nlohmann::json();
    *jProof = friProof->proofs.proof2json();

    *zkin = proof2zkinStark(*jProof, *(StarkInfo *)pStarkInfo);
    (*zkin)["publics"] = publicStarkJson;
    if (numRootC != 0)
        (*zkin)["rootC"] = xrootC;

    return zkin;
}

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

void *polinomial_new(uint64_t degree, uint64_t dim, char *name)
{
    auto pol = new Polinomial(degree, dim, string(name));
    return (void *)pol;
}

void *polinomial_get_p_element(void *pPolinomial, uint64_t index)
{
    Polinomial *polinomial = (Polinomial *)pPolinomial;
    return polinomial->operator[](index);
}

void polinomial_free(void *pPolinomial)
{
    Polinomial *polinomial = (Polinomial *)pPolinomial;
    delete polinomial;
}

void goldilocks_linear_hash(void *pInput, void *pOutput)
{
    Goldilocks::Element input[12];

    memcpy(input, pInput, 8 * sizeof(Goldilocks::Element));
    memset(&input[8], 0, 4 * sizeof(Goldilocks::Element));

    PoseidonGoldilocks::hash(*(Goldilocks::Element(*)[4])pOutput, input);
}

// CHelpersSteps

void *chelpers_steps_new(void *pStarkInfo, void *pChelpers, void* pParams)
{
    CHelpersSteps *genericSteps = new CHelpersSteps(*(StarkInfo *)pStarkInfo, *(CHelpers *)pChelpers, *(StepsParams *)pParams);
    return genericSteps;
}

void set_commit_calculated(void *pCHelpersSteps, uint64_t id)
{
    CHelpersSteps *cHelpersSteps = (CHelpersSteps *)pCHelpersSteps;
    cHelpersSteps->setCommitCalculated(id);
}

void can_stage_be_calculated(void *pCHelpersSteps, uint64_t step)
{
    CHelpersSteps *cHelpersSteps = (CHelpersSteps *)pCHelpersSteps;
    cHelpersSteps->canStageBeCalculated(step);
}

void *get_hint_field(void *pChelpersSteps, uint64_t hintId, char * hintFieldName) 
{
    CHelpersSteps *cHelpersSteps = (CHelpersSteps *)pChelpersSteps;
    auto [elementPtr, hintFieldTypeValue] = cHelpersSteps->getHintField(hintId, string(hintFieldName));

    void** resultTuple = new void*[2];
    resultTuple[0] = (void *)elementPtr;
    resultTuple[1] = reinterpret_cast<void*>(static_cast<uintptr_t>(hintFieldTypeValue));

    return resultTuple;
}

void set_hint_field(void *pChelpersSteps, void *values, uint64_t hintId, char * hintFieldName) 
{
    CHelpersSteps *cHelpersSteps = (CHelpersSteps *)pChelpersSteps;
    cHelpersSteps->setHintField((Goldilocks::Element *)values, hintId, string(hintFieldName));
}

void chelpers_steps_free(void *pCHelpersSteps)
{
    CHelpersSteps *cHelpersSteps = (CHelpersSteps *)pCHelpersSteps;
    delete cHelpersSteps;
}