#include "config.hpp"
#include "circom.hpp"
#include "main.hpp"
#include "main.recursive1.hpp"
#include "zkglobals.hpp"
#include "ZkevmSteps.hpp"
#include "C12aSteps.hpp"
#include "Recursive1Steps.hpp"
#include "Recursive2Steps.hpp"
#include "RecursiveFSteps.hpp"
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

void *zkevm_steps_new()
{
    ZkevmSteps *zkevmSteps = new ZkevmSteps();
    return zkevmSteps;
}

void zkevm_steps_free(void *pZkevmSteps)
{
    ZkevmSteps *zkevmSteps = (ZkevmSteps *)pZkevmSteps;
    delete zkevmSteps;
}

void *c12a_steps_new()
{
    C12aSteps *c12aSteps = new C12aSteps();
    return c12aSteps;
}
void c12a_steps_free(void *pC12aSteps)
{
    C12aSteps *c12aSteps = (C12aSteps *)pC12aSteps;
    delete c12aSteps;
}
void *recursive1_steps_new()
{
    Recursive1Steps *recursive1Steps = new Recursive1Steps();
    return recursive1Steps;
}
void recursive1_steps_free(void *pRecursive1Steps)
{
    Recursive1Steps *recursive1Steps = (Recursive1Steps *)pRecursive1Steps;
    delete recursive1Steps;
}
void *recursive2_steps_new()
{
    Recursive2Steps *recursive2Steps = new Recursive2Steps();
    return recursive2Steps;
}

void recursive2_steps_free(void *pRecursive2Steps)
{
    Recursive2Steps *recursive2Steps = (Recursive2Steps *)pRecursive2Steps;
    delete recursive2Steps;
}

void *generic_steps_new()
{
    CHelpersSteps *genericSteps = new CHelpersSteps();
    return genericSteps;
}
void generic_steps_free(void *pGenericSteps)
{
    CHelpersSteps *genericSteps = (CHelpersSteps *)pGenericSteps;
    delete genericSteps;
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

void set_mapOffsets(void *pStarkInfo, void *pChelpers)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    auto cHelpers = (CHelpers *)pChelpers;
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

void *starks_new(void *pConfig, char *constPols, bool mapConstPolsFile, char *constantsTree, void *starkInfo, void *cHelpers, void *pAddress)
{
    return new Starks<Goldilocks::Element>(*(Config *)pConfig, {constPols, mapConstPolsFile, constantsTree}, pAddress, *(StarkInfo *)starkInfo, *(CHelpers *)cHelpers, false);
}

void *starks_new_default(char *constPols, bool mapConstPolsFile, char *constantsTree, void *starkInfo, void *cHelpers, void *pAddress)
{
    Config configLocal;
    configLocal.runFileGenBatchProof = true; //to force function generateProof to return true

    Goldilocks::Element* addressElements = (Goldilocks::Element *)pAddress;

    return new Starks<Goldilocks::Element>(configLocal, {constPols, mapConstPolsFile, constantsTree}, pAddress, *(StarkInfo *)starkInfo, *(CHelpers *)cHelpers, false);
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
    HintHandlerBuilder::registerBuilder(SubproofValueHintHandler::getName(), std::make_unique<SubproofValueHintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GProdColHintHandler::getName(), std::make_unique<GProdColHintHandlerBuilder>());
}

void *steps_params_new(void *pStarks, void *pChallenges, void *pSubproofValues, void *pEvals, void *pPublicInputs)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    Goldilocks::Element *challenges = (Goldilocks::Element *)pChallenges;
    Goldilocks::Element *subproofValues = (Goldilocks::Element *)pSubproofValues;
    Goldilocks::Element *evals = (Goldilocks::Element *)pEvals;
    Goldilocks::Element *publicInputs = (Goldilocks::Element *)pPublicInputs;

    return starks->ffi_create_steps_params(challenges, subproofValues, evals, publicInputs);
}

void *get_steps_params_field(void *pStepsParams, char *name)
{
    StepsParams *stepsParams = (StepsParams *)pStepsParams;

    if (strcmp(name, "q_2ns") == 0)
    {
        return stepsParams->q_2ns;
    }
    else if (strcmp(name, "f_2ns") == 0)
    {
        return stepsParams->f_2ns;
    }
    else
    {
        return NULL;
    }
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

void compute_stage_expressions(void *pStarks, uint32_t elementType, uint64_t step, void *pParams, void *pProof, void *pChelpersSteps)
{
    // type == 1 => Goldilocks
    // type == 2 => BN128
    switch (elementType)
    {
    case 1:
        ((Starks<Goldilocks::Element> *)pStarks)->computeStageExpressions(step, *(StepsParams *)pParams, *(FRIProof<Goldilocks::Element> *)pProof, (CHelpersSteps *)pChelpersSteps);
        break;
    default:
        cerr << "Invalid elementType: " << elementType << endl;
        break;
    }
}

void calculate_expression(void *pStarks, void* dest, uint64_t id, void * params, void * chelpersSteps, bool domainExtended)
{
    ((Starks<Goldilocks::Element> *)pStarks)->calculateExpression((Goldilocks::Element *)dest, id, *(StepsParams *)params, (CHelpersSteps *)chelpersSteps, domainExtended);
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
    starks->computeFRIPol(step, *(StepsParams *)pParams, (CHelpersSteps *)cHelpersSteps);
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

void *get_vector_pointer(void *pStarks, char *name)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    return starks->ffi_get_vector_pointer(name);
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

void clean_symbols_calculated(void *pStarks)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->cleanSymbolsCalculated();
}

void set_symbol_calculated(void *pStarks, uint32_t operand, uint64_t id)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->ffi_set_symbol_calculated(operand, id);
}

void calculate_hash(void *pStarks, void *pHhash, void *pBuffer, uint64_t nElements)
{
    Starks<Goldilocks::Element> *starks = (Starks<Goldilocks::Element> *)pStarks;
    starks->calculateHash((Goldilocks::Element *)pHhash, (Goldilocks::Element *)pBuffer, nElements);
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
