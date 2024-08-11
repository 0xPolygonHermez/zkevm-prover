#include "definitions.hpp"
#include "starks.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

USING_PROVER_FORK_NAMESPACE;

template <typename ElementType>
void Starks<ElementType>::genProof(FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs, CHelpersSteps *chelpersSteps)
{
    TimerStart(STARK_PROOF);

    // Initialize vars
    TimerStart(STARK_INITIALIZATION);

    cleanSymbolsCalculated();

    TranscriptType transcript(merkleTreeArity, merkleTreeCustom);

    Goldilocks::Element* evals = new Goldilocks::Element[starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* challenges = new Goldilocks::Element[starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* subproofValues = new Goldilocks::Element[starkInfo.nSubProofValues * FIELD_EXTENSION];

    StepsParams params = {
        pols : mem,
        constPols : constPols.pConstPolsAddress,
        constPolsExtended : constPols.pConstPolsAddressExtended,
        challenges : challenges,
        subproofValues : subproofValues,
        evals : evals,
        zi : zi,
        publicInputs : publicInputs,
    };

    for (uint64_t i = 0; i < starkInfo.mapSectionsN["cm1"]; ++i)
    {
        setCommitCalculated(i);
    }

    TimerStopAndLog(STARK_INITIALIZATION);

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    TimerStart(STARK_STEP_0);
    if(!debug) {
        ElementType verkey[nFieldElements];
        treesGL[starkInfo.nStages + 1]->getRoot(verkey);
        addTranscript(transcript, &verkey[0], nFieldElements);
    }
    
    if(starkInfo.starkStruct.hashCommits) {
        ElementType hash[nFieldElements];
        calculateHash(hash, &publicInputs[0], starkInfo.nPublics);
        addTranscript(transcript, hash, nFieldElements);
    } else {
        addTranscriptGL(transcript, &publicInputs[0], starkInfo.nPublics);
    }

    TimerStopAndLog(STARK_STEP_0);

    for (uint64_t step = 1; step <= starkInfo.nStages; step++)
    {
        TimerStartExpr(STARK_STEP, step);
        for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
        {
            if(starkInfo.challengesMap[i].stage == step) {
                getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
            }
        }

        computeStageExpressions(step, params, proof, chelpersSteps);

        if(step == starkInfo.nStages) {
            calculateImPolsExpressions(step, params, chelpersSteps);
        }

        commitStage(step, params, proof);

        if (debug)
        {
            for (uint64_t i = 0; i < chelpers.constraintsInfoDebug.size(); i++)
            {
                if(chelpers.constraintsInfoDebug[i].stage == step) {
                    calculateConstraint(i, params, chelpersSteps);
                }
            }
            Goldilocks::Element randomValues[4] = {Goldilocks::fromU64(0), Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(3)};
            addTranscriptGL(transcript, randomValues, 4);
        }
        else
        {
            addTranscript(transcript, &proof.proofs.roots[step - 1][0], nFieldElements);
        }

        TimerStopAndLogExpr(STARK_STEP, step);
    }

    if (debug) return;

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stage == starkInfo.nStages + 1) {
            getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }
    
    calculateExpression(&params.pols[starkInfo.mapOffsets[std::make_pair("q", true)]], starkInfo.cExpId, params, chelpersSteps, true, false);
    
    commitStage(starkInfo.nStages + 1, params, proof);

    if (debug)
    {
        Goldilocks::Element randomValues[4] = {Goldilocks::fromU64(0), Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(3)};
        addTranscriptGL(transcript, randomValues, 4);
    }
    else
    {
        addTranscript(transcript, &proof.proofs.roots[starkInfo.nStages][0], nFieldElements);
    }
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stage == starkInfo.nStages + 2) {
            getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }

    computeEvals(params, proof);

    if(starkInfo.starkStruct.hashCommits) {
        ElementType hash[nFieldElements];
        calculateHash(hash, params.evals, starkInfo.evMap.size() * FIELD_EXTENSION);
        addTranscript(transcript, hash, nFieldElements);
    } else {
        addTranscriptGL(transcript, params.evals, starkInfo.evMap.size() * FIELD_EXTENSION);
    }    

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stage == starkInfo.nStages + 3) {
            getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }

    computeFRIPol(starkInfo.nStages + 2, params, chelpersSteps);

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    Goldilocks::Element *friPol = &params.pols[starkInfo.mapOffsets[std::make_pair("f", true)]];
    
    for (uint64_t step = 0; step < starkInfo.starkStruct.steps.size(); step++)
    {
        computeFRIFolding(proof, friPol, step, challenge);
        if (step < starkInfo.starkStruct.steps.size() - 1)
        {
            addTranscript(transcript, &proof.proofs.fri.trees[step + 1].root[0], nFieldElements);
        }
        else
        {
            if(starkInfo.starkStruct.hashCommits) {
                ElementType hash[nFieldElements];
                calculateHash(hash, friPol, (1 << starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                addTranscript(transcript, hash, nFieldElements);
            } else {
                addTranscriptGL(transcript, friPol, (1 << starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            }
        }
        getChallenge(transcript, *challenge);
    }

    uint64_t friQueries[starkInfo.starkStruct.nQueries];

    TranscriptType transcriptPermutation(merkleTreeArity, merkleTreeCustom);
    addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.steps[0].nBits);

    computeFRIQueries(proof, friQueries);

    TimerStopAndLog(STARK_STEP_FRI);

    delete challenges;
    delete evals;
    delete subproofValues;
        
    TimerStopAndLog(STARK_PROOF);
}

template <typename ElementType>
void Starks<ElementType>::calculateImPolsExpressions(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps)
{
    TimerStart(STARK_CALCULATE_IMPOLS_EXPS);
    if (chelpers.imPolsInfo[step - 1].nOps > 0)
    {
        chelpersSteps->calculateExpressions(nullptr, starkInfo, params, chelpers.cHelpersArgsImPols, chelpers.imPolsInfo[step - 1], false, true);
        for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); i++) {
            if(starkInfo.cmPolsMap[i].imPol && starkInfo.cmPolsMap[i].stage == step) {
                setCommitCalculated(i);
            }
        }
    }
    TimerStopAndLog(STARK_CALCULATE_IMPOLS_EXPS);
}

template <typename ElementType>
void Starks<ElementType>::calculateExpression(Goldilocks::Element* dest, uint64_t id, StepsParams &params, CHelpersSteps *chelpersSteps, bool domainExtended, bool imPol)
{
    TimerStartExpr(STARK_CALCULATE_EXPRESSION, id);
    chelpersSteps->calculateExpressions(dest, starkInfo, params, chelpers.cHelpersArgsExpressions, chelpers.expressionsInfo[id], domainExtended, imPol);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPRESSION, id);
}

template <typename ElementType>
void Starks<ElementType>::calculateConstraint(uint64_t constraintId, StepsParams &params, CHelpersSteps *chelpersSteps)
{
    TimerStartExpr(STARK_CALCULATE_CONSTRAINT, constraintId);
    chelpersSteps->calculateExpressions(nullptr, starkInfo, params, chelpers.cHelpersArgsDebug, chelpers.constraintsInfoDebug[constraintId], false, false);
    TimerStopAndLogExpr(STARK_CALCULATE_CONSTRAINT, constraintId);
}

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelize(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof)
{
    TimerStartExpr(STARK_LDE_AND_MERKLETREE_STEP, step);
    TimerStartExpr(STARK_LDE_STEP, step);

    std::string section = "cm" + to_string(step);  
    uint64_t nCols = starkInfo.mapSectionsN["cm" + to_string(step)];

    Goldilocks::Element *pBuff = &params.pols[starkInfo.mapOffsets[make_pair(section, false)]];
    Goldilocks::Element *pBuffExtended = &params.pols[starkInfo.mapOffsets[make_pair(section, true)]];

    std::pair<uint64_t, uint64_t> nttOffsetHelper = starkInfo.mapNTTOffsetsHelpers[section];
    Goldilocks::Element *pBuffHelper = &params.pols[nttOffsetHelper.first];

    uint64_t buffHelperElements = NExtended * nCols;

    uint64_t nBlocks = 1;
    while((nttOffsetHelper.second * nBlocks < buffHelperElements + 8) ||  (nCols > 256*nBlocks) ) {
        nBlocks++;
    }

    ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_LDE_STEP, step);
    TimerStartExpr(STARK_MERKLETREE_STEP, step);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proofs.roots[step - 1][0]);
    TimerStopAndLogExpr(STARK_MERKLETREE_STEP, step);
    TimerStopAndLogExpr(STARK_LDE_AND_MERKLETREE_STEP, step);
}

template <typename ElementType>
void Starks<ElementType>::commitStage(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof)
{   
    if(step == starkInfo.nStages) {
        proof.proofs.setSubproofValues(params.subproofValues);
    }

    if(!debug) {
        if (step <= starkInfo.nStages)
        {
            extendAndMerkelize(step, params, proof);
        }
        else
        {
            computeQ(step, params, proof);
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::computeStageExpressions(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof, CHelpersSteps *chelpersSteps)
{
    TimerStartExpr(STARK_TRY_CALCULATE_EXPS_STEP, step);
    uint64_t symbolsToBeCalculated = isStageCalculated(step);
    while (symbolsToBeCalculated > 0)
    {
        calculateHints(step, params, chelpersSteps);
        uint64_t newSymbolsToBeCalculated = isStageCalculated(step);
        if (newSymbolsToBeCalculated == symbolsToBeCalculated)
        {
            zklog.info("Something went wrong when calculating stage " + to_string(step));
            exitProcess();
            exit(-1);
        }
        symbolsToBeCalculated = newSymbolsToBeCalculated;
    }
    TimerStopAndLogExpr(STARK_TRY_CALCULATE_EXPS_STEP, step);
}

template <typename ElementType>
void Starks<ElementType>::computeQ(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof)
{
    std::string section = "cm" + to_string(starkInfo.nStages + 1);
    uint64_t nCols = starkInfo.mapSectionsN["cm" + to_string(starkInfo.nStages + 1)];
    Goldilocks::Element *cmQ = &params.pols[starkInfo.mapOffsets[make_pair(section, true)]];

    std::pair<uint64_t, uint64_t> nttOffsetHelper = starkInfo.mapNTTOffsetsHelpers[section];
    Goldilocks::Element *pBuffHelper = &params.pols[nttOffsetHelper.first];

    uint64_t buffHelperElements = NExtended * nCols;
    
    uint64_t nBlocks = 1;
    while((nttOffsetHelper.second * nBlocks < buffHelperElements) || (nCols > 256*nBlocks) ) {
        nBlocks++;
    }

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);
    nttExtended.INTT(&params.pols[starkInfo.mapOffsets[std::make_pair("q", true)]], &params.pols[starkInfo.mapOffsets[std::make_pair("q", true)]], NExtended, starkInfo.qDim, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);

    for (uint64_t p = 0; p < starkInfo.qDeg; p++)
    {   
        __m256i sigma = _mm256_set1_epi64x(S[p].fe);
        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i += nrowsPack)
        {
            Goldilocks3::Element_avx tmp_; 
            Goldilocks3::load_avx(tmp_, &params.pols[starkInfo.mapOffsets[std::make_pair("q", true)] + (p * N + i) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION));
            Goldilocks3::op_31_avx(2, tmp_, tmp_, sigma);
            Goldilocks3::store_avx(&cmQ[(i * starkInfo.qDeg + p) * FIELD_EXTENSION],starkInfo.qDeg * FIELD_EXTENSION, tmp_);
        }
    }

    memset(&cmQ[N * starkInfo.qDeg * starkInfo.qDim], 0, (NExtended - N) * starkInfo.qDeg * starkInfo.qDim * sizeof(Goldilocks::Element));

    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_NTT_STEP, step);
    nttExtended.NTT(cmQ, cmQ, NExtended, nCols, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_NTT_STEP, step);

    TimerStartExpr(STARK_MERKLETREE_STEP, step);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proofs.roots[step - 1][0]);

    TimerStopAndLogExpr(STARK_MERKLETREE_STEP, step);

    for (uint64_t i = 0; i < starkInfo.cmPolsMap.size(); ++i)
    {
        if(starkInfo.cmPolsMap[i].stage == step) {
            setCommitCalculated(i);
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::computeEvals(StepsParams &params, FRIProof<ElementType> &proof)
{
    auto evalsStage = starkInfo.nStages + 2;
    auto xiChallenge = std::find_if(starkInfo.challengesMap.begin(), starkInfo.challengesMap.end(), [evalsStage](const PolMap& c) {
        return c.stage == evalsStage && c.stageId == 0;
    });

    uint64_t xiChallengeIndex = std::distance(starkInfo.challengesMap.begin(), xiChallenge);

    TimerStart(STARK_CALCULATE_LEv);
    
    Goldilocks::Element* LEv = &params.pols[starkInfo.mapOffsets[make_pair("LEv", true)]];
    
    Goldilocks::Element xisShifted[starkInfo.openingPoints.size() * FIELD_EXTENSION];

    Goldilocks::Element shift_inv = Goldilocks::inv(Goldilocks::shift());
    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
        Goldilocks::Element w = Goldilocks::one();
        uint64_t openingAbs = starkInfo.openingPoints[i] < 0 ? -starkInfo.openingPoints[i] : starkInfo.openingPoints[i];
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w = w * Goldilocks::w(starkInfo.starkStruct.nBits);
        }

        if (starkInfo.openingPoints[i] < 0)
        {
            w = Goldilocks::inv(w);
        }

        Goldilocks3::mul((Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(params.challenges[xiChallengeIndex * FIELD_EXTENSION]), w);
        Goldilocks3::mul((Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), shift_inv);

        Goldilocks3::one((Goldilocks3::Element &)LEv[i * FIELD_EXTENSION]);
    }


#pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
        for (uint64_t k = 1; k < N; k++)
        {
            Goldilocks3::mul((Goldilocks3::Element &)(LEv[(k*starkInfo.openingPoints.size() + i)*FIELD_EXTENSION]), (Goldilocks3::Element &)(LEv[((k-1)*starkInfo.openingPoints.size() + i)*FIELD_EXTENSION]), (Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]));
        }
    }

    std::pair<uint64_t, uint64_t> nttOffsetHelper = starkInfo.mapNTTOffsetsHelpers["LEv"];
    Goldilocks::Element *pBuffHelper = &params.pols[nttOffsetHelper.first];
    
    ntt.INTT(&LEv[0], &LEv[0], N, FIELD_EXTENSION * starkInfo.openingPoints.size(), pBuffHelper);

    TimerStopAndLog(STARK_CALCULATE_LEv);

    TimerStart(STARK_CALCULATE_EVALS);
    evmap(params, LEv);
    proof.proofs.setEvals(params.evals);
    TimerStopAndLog(STARK_CALCULATE_EVALS);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIPol(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps)
{

    TimerStart(STARK_CALCULATE_XDIVXSUB);

    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
        Goldilocks3::Element_avx xis_;
        xis_[0] = _mm256_set1_epi64x(xis[i * FIELD_EXTENSION].fe);
        xis_[1] = _mm256_set1_epi64x(xis[i * FIELD_EXTENSION + 1].fe);
        xis_[2] = _mm256_set1_epi64x(xis[i * FIELD_EXTENSION + 2].fe);
#pragma omp parallel for
        for (uint64_t k = 0; k < NExtended; k += nrowsPack)
        {
            __m256i x_k;
            Goldilocks::load_avx(x_k, &x[k], uint64_t(1));
            Goldilocks3::Element_avx tmp_; 
            Goldilocks3::op_31_avx(3, tmp_, xis_, x_k);
            Goldilocks3::store_avx(&params.pols[starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)] + (k + i * NExtended) * FIELD_EXTENSION], FIELD_EXTENSION, tmp_);
        }
        // for (uint64_t k = 0; k < NExtended; k++)
        // {
        //     Goldilocks3::sub((Goldilocks3::Element &)(&params.pols[starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)  + (k + i * NExtended) * FIELD_EXTENSION]), x[k], (Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]));
        // }
    }

    Polinomial xDivXSubXi_(&params.pols[starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)]], NExtended * starkInfo.openingPoints.size(), FIELD_EXTENSION, FIELD_EXTENSION);
    Polinomial::batchInverseParallel(xDivXSubXi_, xDivXSubXi_);

    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
#pragma omp parallel for
        for (uint64_t k = 0; k < NExtended; k += nrowsPack)
        {
            __m256i x_k;
            Goldilocks::load_avx(x_k, &x[k], uint64_t(1));
            Goldilocks3::Element_avx tmp_; 
            Goldilocks3::load_avx(tmp_, &params.pols[starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)] + (k + i * NExtended) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION));
            Goldilocks3::op_31_avx(2, tmp_, tmp_, x_k);
            Goldilocks3::store_avx(&params.pols[starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)] + (k + i * NExtended) * FIELD_EXTENSION], FIELD_EXTENSION, tmp_);
        }
        // for (uint64_t k = 0; k < NExtended; k++)
        // {
        //     Goldilocks3::mul((Goldilocks3::Element &)(params.pols[starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true) + (k + i * NExtended) * FIELD_EXTENSION]), (Goldilocks3::Element &)(&params.pols[starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)] + (k + i * NExtended) * FIELD_EXTENSION]), x[k]);
        // }
    }
    TimerStopAndLog(STARK_CALCULATE_XDIVXSUB);

    calculateExpression(&params.pols[starkInfo.mapOffsets[std::make_pair("f", true)]], starkInfo.friExpId, params, chelpersSteps, true, false);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIFolding(FRIProof<ElementType> &fproof, Goldilocks::Element* pol, uint64_t step, Goldilocks::Element *challenge)
{
    FRI<ElementType>::fold(step, fproof, pol, challenge, starkInfo, treesFRI);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIQueries(FRIProof<ElementType> &fproof, uint64_t *friQueries)
{
    FRI<ElementType>::proveQueries(friQueries, fproof, treesGL, treesFRI, starkInfo);
}


template <typename ElementType>
bool Starks<ElementType>::canExpressionBeCalculated(ParserParams &parserParams) {
    for(uint64_t i = 0; i < parserParams.nCmPolsUsed; i++) {
        uint64_t cmPolUsedId = chelpers.cHelpersArgsExpressions.cmPolsIds[parserParams.cmPolsOffset + i];
        if (!isSymbolCalculated(opType::cm, cmPolUsedId)) {
            return false;
        }
    }

    for(uint64_t i = 0; i < parserParams.nChallengesUsed; i++) {
        uint64_t challengeUsedId = chelpers.cHelpersArgsExpressions.challengesIds[parserParams.challengesOffset + i];
        if (!isSymbolCalculated(opType::challenge, challengeUsedId)) {
            return false;
        }
    }

    for(uint64_t i = 0; i < parserParams.nPublicsUsed; i++) {
        uint64_t publicUsedId = chelpers.cHelpersArgsExpressions.publicsIds[parserParams.publicsOffset + i];
        if (!isSymbolCalculated(opType::public_, publicUsedId)) {
            return false;
        }
    }

    for(uint64_t i = 0; i < parserParams.nConstPolsUsed; i++) {
        uint64_t constPolUsedId = chelpers.cHelpersArgsExpressions.constPolsIds[parserParams.constPolsOffset + i];
        if (!isSymbolCalculated(opType::const_, constPolUsedId)) {
            return false;
        }
    }

    for(uint64_t i = 0; i < parserParams.nSubproofValuesUsed; i++) {
        uint64_t subproofValueUsedId = chelpers.cHelpersArgsExpressions.subproofValuesIds[parserParams.subproofValuesOffset + i];
        if (!isSymbolCalculated(opType::subproofvalue, subproofValueUsedId)) {
            return false;
        }
    }
    return true;
}

template <typename ElementType>
bool Starks<ElementType>::isHintResolved(Hint &hint, vector<string> dstFields)
{
    for (uint64_t i = 0; i < dstFields.size(); i++)
    {
        if (!isSymbolCalculated(hint.fields[dstFields[i]].operand, hint.fields[dstFields[i]].id)) {
            return false;
        }
    }

    return true;
}

template <typename ElementType>
bool Starks<ElementType>::canHintBeResolved(Hint &hint, vector<string> srcFields)
{
    for (uint64_t i = 0; i < srcFields.size(); i++)
    {
        HintField field = hint.fields[srcFields[i]];
        if (field.operand == opType::number) continue;
        if(field.operand == opType::tmp) {
            ParserParams expInfo = chelpers.expressionsInfo[field.id];
            
            for(uint64_t i = 0; i < expInfo.nCmPolsUsed; ++i) {
                if(!isSymbolCalculated(opType::cm, chelpers.cHelpersArgsExpressions.cmPolsIds[expInfo.cmPolsOffset + i])) return false;
            }

            for(uint64_t i = 0; i < expInfo.nChallengesUsed; ++i) {
                if(!isSymbolCalculated(opType::challenge, chelpers.cHelpersArgsExpressions.challengesIds[expInfo.challengesOffset + i])) return false;
            }

            for(uint64_t i = 0; i < expInfo.nSubproofValuesUsed; ++i) {
                if(!isSymbolCalculated(opType::subproofvalue, chelpers.cHelpersArgsExpressions.subproofValuesIds[expInfo.subproofValuesOffset + i])) return false;
            }

        } else if (!isSymbolCalculated(hint.fields[srcFields[i]].operand, hint.fields[srcFields[i]].id)) {
            return false;
        }
    }

    return true;
}

template <typename ElementType>
void Starks<ElementType>::calculateHints(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps)
{
    Polinomial* polynomials = new Polinomial[starkInfo.cmPolsMap.size()];

    Polinomial* polynomialsExps = new Polinomial[starkInfo.friExpId + 1];

    vector<bool> srcPolsExpsNames(starkInfo.friExpId + 1, false);

    vector<uint64_t> srcPolsNames;
    vector<uint64_t> dstPolsNames;    

    vector<uint64_t> hintsToCalculate;
    
    TimerStartExpr(STARK_PREPARE_HINTS_STEP, step);
    for (uint64_t i = 0; i < chelpers.hints.size(); i++)
    {
        Hint hint = chelpers.hints[i];
        auto hintHandler = HintHandlerBuilder::create(hint.name)->build();
        vector<string> srcFields = hintHandler->getSources();
        vector<string> dstFields = hintHandler->getDestinations();
        if (!isHintResolved(hint, dstFields) && canHintBeResolved(hint, srcFields))
        {
            hintsToCalculate.push_back(i);

            for (uint64_t i = 0; i < srcFields.size(); i++)
            {
                HintField hintField = hint.fields[srcFields[i]];
                if(hintField.operand != opType::tmp && hintField.operand != opType::cm) continue;
                if (hintField.operand != opType::tmp) {
                    PolMap polInfo = starkInfo.cmPolsMap[hintField.id];
                    srcPolsNames.push_back(hintField.id);
                    polynomials[hintField.id].potConstruct(N, polInfo.dim);
                } else {
                    srcPolsExpsNames[hintField.id] = true;
                    polynomialsExps[hintField.id].potConstruct(N, hintField.dim);                    
                }
            }

            for (uint64_t i = 0; i < dstFields.size(); i++)
            {
                HintField hintField = hint.fields[dstFields[i]];
                if(hintField.operand == opType::tmp) exitProcess();
                if(hintField.operand != opType::cm) continue;
                PolMap polInfo = starkInfo.cmPolsMap[hintField.id];
                dstPolsNames.push_back(hintField.id);
                polynomials[hintField.id].potConstruct(N, polInfo.dim);
            }
        }
    }
    TimerStopAndLogExpr(STARK_PREPARE_HINTS_STEP, step);

    if (hintsToCalculate.size() == 0)
        return;

    
    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);
    Polinomial *srcTransposedPols = new Polinomial[srcPolsNames.size()];
    for(uint64_t i = 0; i < srcPolsNames.size(); i++) {
        starkInfo.getPolynomial(srcTransposedPols[i], params.pols, true, srcPolsNames[i], N);
    }
#pragma omp parallel for
    for(uint64_t j = 0; j < N; ++j) {
        for (uint64_t i = 0; i < srcPolsNames.size(); ++i)
        {
            std::memcpy(polynomials[srcPolsNames[i]][j], srcTransposedPols[i][j], srcTransposedPols[i].dim() * sizeof(Goldilocks::Element));
        }
    }
    delete[] srcTransposedPols;
    TimerStopAndLogExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);

    TimerStart(STARK_CALCULATE_EXPRESSIONS);
    
    for(uint64_t i = 0; i < srcPolsExpsNames.size(); i++) {
        if(srcPolsExpsNames[i]) {
            Goldilocks::Element* dst = polynomialsExps[i].address();
            calculateExpression(dst, i, params, chelpersSteps, false, false);
        }    
    }

    TimerStopAndLog(STARK_CALCULATE_EXPRESSIONS);

    TimerStartExpr(STARK_CALCULATE_HINTS_STEP, step);
    uint64_t maxThreads = omp_get_max_threads();
    uint64_t nThreads = hintsToCalculate.size() > maxThreads ? maxThreads : hintsToCalculate.size();

#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < hintsToCalculate.size(); i++)
    {
        Hint hint = chelpers.hints[hintsToCalculate[i]];
        
        // Build the Hint object
        auto hintHandler = HintHandlerBuilder::create(hint.name)->build();

        // Get the polynomials names involved
        auto srcPolsNames = hintHandler->getSources();
        auto dstPolsNames = hintHandler->getDestinations();

        vector<string> polsNames(srcPolsNames.size() + dstPolsNames.size());
        for(uint64_t i = 0; i < srcPolsNames.size(); ++i) {
            polsNames[i] = srcPolsNames[i];
        }
        for(uint64_t i = 0; i < dstPolsNames.size(); ++i) {
            polsNames[i + srcPolsNames.size()] = dstPolsNames[i];
        }

        // Prepare polynomials map to be sent to the hint
        std::map<std::string, Polinomial *> polynomialsHint;
        for (const auto &polName : polsNames)
        {
            const auto &hintField = hint.fields[polName];
            if (hintField.operand == opType::cm) {
                polynomialsHint[polName] = &polynomials[hintField.id];
            } else if(hintField.operand == opType::tmp) {
                polynomialsHint[polName] = &polynomialsExps[hintField.id];
            }
        }

        if(hint.name == "gsum") {
            calculateS(polynomials[hint.fields["reference"].id], polynomialsExps[hint.fields["denominator"].id], Goldilocks::fromU64(hint.fields["numerator"].value));
                
            params.subproofValues[hint.fields["result"].id * FIELD_EXTENSION] = polynomials[hint.fields["reference"].id][N - 1][0];
            params.subproofValues[hint.fields["result"].id * FIELD_EXTENSION + 1] = polynomials[hint.fields["reference"].id][N - 1][1];
            params.subproofValues[hint.fields["result"].id * FIELD_EXTENSION + 2] = polynomials[hint.fields["reference"].id][N - 1][2];
        } else {
            hintHandler->resolveHint(N, params, hint, polynomialsHint);
        }
    }

    TimerStopAndLogExpr(STARK_CALCULATE_HINTS_STEP, step);

    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_2_STEP, step);
    Polinomial *dstTransposedPols = new Polinomial[dstPolsNames.size()];
    for(uint64_t i = 0; i < dstPolsNames.size(); i++) {
        starkInfo.getPolynomial(dstTransposedPols[i], params.pols, true, dstPolsNames[i], N);
    }
#pragma omp parallel for
    for(uint64_t j = 0; j < N; ++j) {
        for (uint64_t i = 0; i < dstPolsNames.size(); ++i)
        {
            std::memcpy(dstTransposedPols[i][j], polynomials[dstPolsNames[i]][j], dstTransposedPols[i].dim() * sizeof(Goldilocks::Element));
        }
    }
    delete[] dstTransposedPols;
    TimerStopAndLogExpr(STARK_CALCULATE_TRANSPOSE_2_STEP, step);
    
    for (uint64_t i = 0; i < hintsToCalculate.size(); i++)
    {
        Hint hint = chelpers.hints[hintsToCalculate[i]];

        // Build the Hint object
        auto hintHandler = HintHandlerBuilder::create(hint.name)->build();
        auto dstFields = hintHandler->getDestinations();

        for (uint64_t i = 0; i < dstFields.size(); i++)
        {
            HintField hintField = hint.fields[dstFields[i]];
            if(hintField.operand == opType::cm) {
                setCommitCalculated(hintField.id);
            } else if(hintField.operand == opType::subproofvalue) {
                setSubproofValueCalculated(hintField.id);
            }
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::evmap(StepsParams &params, Goldilocks::Element *LEv)
{
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    u_int64_t size_eval = starkInfo.evMap.size();


    int num_threads = omp_get_max_threads();
    int size_thread = size_eval * FIELD_EXTENSION;
    Goldilocks::Element *evals_acc = &params.pols[starkInfo.mapOffsets[std::make_pair("evals", true)]];
    memset(&evals_acc[0], 0, num_threads * size_thread * sizeof(Goldilocks::Element));

//     vector<uint64_t> cmPolsPos(starkInfo.nCommitments, 0);

// #pragma omp parallel
//     {
//         int thread_idx = omp_get_thread_num();
//         Goldilocks::Element *evals_acc_thread = &evals_acc[thread_idx * size_thread];

// #pragma omp for
//         for (uint64_t i = 0; i < N; i+= nrowsPack) {
//             __m256i bufferT_[starkInfo.nConstants + starkInfo.nCommitmentsCols];
            
//             Goldilocks::Element bufferT[nrowsPack];

//             for(uint64_t k = 0; k < starkInfo.nConstants; ++k) {
//                 for(uint64_t j = 0; j < nrowsPack; ++j) {
//                     bufferT[j] = (params.constPolsExtended[k + ((i + j) << extendBits) * starkInfo.nConstants];
//                 }
//                 Goldilocks::load_avx(bufferT_[k], bufferT);
//             }

//             uint64_t nColsAccumulated = starkInfo.nConstants;
//             for(uint64_t k = 0; k < starkInfo.nCommitments; ++k) {
//                 PolMap polInfo = starkInfo.cmPolsMap[k];
//                 for(uint64_t d = 0; d < polInfo.dim; ++d) {
//                     for(uint64_t j = 0; j < nrowsPack; ++j) {
//                         bufferT[j] = params.pols[(starkInfo.mapOffsets[std::make_pair(polInfo.stage, true)] + polInfo.stagePos + d) + ((i + j) << extendBits) * starkInfo.mapSectionsN[polInfo.stage]];
//                     }
//                     Goldilocks::load_avx(bufferT_[nColsAccumulated + d], bufferT);
//                 }
//                 cmPolsPos[k] = nColsAccumulated;
//                 nColsAccumulated += polInfo.dim;
//             }

//             Goldilocks3::Element_avx LEv_[starkInfo.openingPoints.size()];
//             for (uint64_t k = 0; k < starkInfo.openingPoints.size(); k++)
//             {
//             Goldilocks3::load_avx(LEv_[k], &LEv[(k + i*starkInfo.openingPoints.size()) * FIELD_EXTENSION], starkInfo.openingPoints.size() * FIELD_EXTENSION);
//             }

//             for(uint64_t e = 0; e < size_eval; e++) {
//                 Goldilocks3::Element_avx tmp_;
//                 uint64_t openingPos = starkInfo.evMap[e].openingPos;
//                 uint64_t id = starkInfo.evMap[e].id;
//                 if (starkInfo.evMap[e].type == EvMap::eType::_const) {
//                     Goldilocks3::op_31_avx(2, tmp_, LEv_[openingPos], bufferT_[id]);
//                 } else if(starkInfo.cmPolsMap[id].dim == 1) {
//                     Goldilocks3::op_31_avx(2, tmp_, LEv_[openingPos], bufferT_[cmPolsPos[id]]);
//                 } else {
//                     Goldilocks3::mul_avx(tmp_, LEv_[openingPos], (Goldilocks3::Element_avx &)bufferT_[cmPolsPos[id]]);
//                 }

//                 Goldilocks::Element evals_[FIELD_EXTENSION * nrowsPack];
//                 Goldilocks3::store_avx(evals_, FIELD_EXTENSION, tmp_);
//                 for(uint64_t j = 0; j < nrowsPack; ++j) {
//                     Goldilocks3::add((Goldilocks3::Element &)(evals_acc_thread[e * FIELD_EXTENSION]), (Goldilocks3::Element &)(evals_acc_thread[e * FIELD_EXTENSION]), (Goldilocks3::Element &)(evals_[j*FIELD_EXTENSION]));
//                 }
//             }
//         }
// #pragma omp for
//         for (uint64_t i = 0; i < size_eval; ++i)
//         {
//             Goldilocks3::Element sum = { Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero() };
//             for (int k = 0; k < num_threads; ++k)
//             {
//                 Goldilocks3::add(sum, sum, (Goldilocks3::Element &)(evals_acc[k * size_thread + i * FIELD_EXTENSION]));
//             }
//             std::memcpy((Goldilocks3::Element &)(params.evals[i * FIELD_EXTENSION]), sum, FIELD_EXTENSION * sizeof(Goldilocks::Element));
//         }
//     }
    
    Polinomial *ordPols = new Polinomial[size_eval];

    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = starkInfo.evMap[i];
        bool committed = ev.type == EvMap::eType::cm ? true : false;
        Goldilocks::Element* pols = committed ? params.pols : params.constPolsExtended;
        starkInfo.getPolynomial(ordPols[i], pols, committed, ev.id, NExtended);
    }

#pragma omp parallel
    {
        int thread_idx = omp_get_thread_num();
        Goldilocks::Element *evals_acc_thread = &evals_acc[thread_idx * size_thread];
#pragma omp for
        for (uint64_t k = 0; k < N; k++)
        {
            Goldilocks3::Element LEv_[starkInfo.openingPoints.size()];
            for(uint64_t o = 0; o < starkInfo.openingPoints.size(); o++) {
                uint64_t pos = (o + k*starkInfo.openingPoints.size()) * FIELD_EXTENSION;
                LEv_[o][0] = LEv[pos];
                LEv_[o][1] = LEv[pos + 1];
                LEv_[o][2] = LEv[pos + 2];
            }
            uint64_t row = (k << extendBits);
            for (uint64_t i = 0; i < size_eval; i++)
            {
                EvMap ev = starkInfo.evMap[i];
                Goldilocks3::Element res;
                if (ordPols[i].dim() == 1) {
                    Goldilocks3::mul(res, LEv_[ev.openingPos], *ordPols[i][row]);
                } else {
                    Goldilocks3::mul(res, LEv_[ev.openingPos], (Goldilocks3::Element &)(*ordPols[i][row]));
                }
                Goldilocks3::add((Goldilocks3::Element &)(evals_acc_thread[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(evals_acc_thread[i * FIELD_EXTENSION]), res);
            }
        }
#pragma omp for
        for (uint64_t i = 0; i < size_eval; ++i)
        {
            Goldilocks3::Element sum = { Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero() };
            for (int k = 0; k < num_threads; ++k)
            {
                Goldilocks3::add(sum, sum, (Goldilocks3::Element &)(evals_acc[k * size_thread + i * FIELD_EXTENSION]));
            }
            std::memcpy((Goldilocks3::Element &)(params.evals[i * FIELD_EXTENSION]), sum, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
    delete[] ordPols;
}

template <typename ElementType>
void Starks<ElementType>::cleanSymbolsCalculated() {
    std::fill(subProofValuesCalculated.begin(), subProofValuesCalculated.end(), false);
    std::fill(commitsCalculated.begin(), commitsCalculated.end(), false);
}

template <typename ElementType>
void Starks<ElementType>::getChallenge(TranscriptType &transcript, Goldilocks::Element &challenge)
{
    transcript.getField((uint64_t *)&challenge);
}

template <typename ElementType>
void Starks<ElementType>::calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements) {
    TranscriptType transcriptHash(merkleTreeArity, merkleTreeCustom);
    transcriptHash.put(buffer, nElements);
    transcriptHash.getState(hash);

};

template <typename ElementType>
void Starks<ElementType>::addTranscriptGL(TranscriptType &transcript, Goldilocks::Element *buffer, uint64_t nElements)
{
    transcript.put(buffer, nElements);
};

template <typename ElementType>
void Starks<ElementType>::addTranscript(TranscriptType &transcript, ElementType *buffer, uint64_t nElements)
{
    transcript.put(buffer, nElements);
};

template <typename ElementType>
uint64_t Starks<ElementType>::isStageCalculated(uint64_t step) {

    uint64_t symbolsToBeCalculated = 0;
    for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); i++) {
        if(starkInfo.cmPolsMap[i].stage != step || starkInfo.cmPolsMap[i].imPol) continue;
        if(!isSymbolCalculated(opType::cm, i)) symbolsToBeCalculated++;
    }

    for(uint64_t i = 0; i < starkInfo.challengesMap.size(); i++) {
        if(starkInfo.challengesMap[i].stage != step) continue;
        if(!isSymbolCalculated(opType::challenge, i)) symbolsToBeCalculated++;
    }

    if(step == 1) {
        for(uint64_t i = 0; i < starkInfo.nPublics; i++) {
            if(!isSymbolCalculated(opType::public_, i)) symbolsToBeCalculated++;
        }
    }

    if(step == starkInfo.nStages) {
        for(uint64_t i = 0; i < starkInfo.nSubProofValues; i++) {
            if(!isSymbolCalculated(opType::subproofvalue, i)) symbolsToBeCalculated++;
        }
    }

    return symbolsToBeCalculated;
}

template <typename ElementType>
bool Starks<ElementType>::isSymbolCalculated(opType operand, uint64_t id)
{
    bool isCalculated = false;
    if (operand == opType::cm)
    {
        if (commitsCalculated[id])
            isCalculated = true;
    }
    else if (operand == opType::subproofvalue)
    {
        if (subProofValuesCalculated[id])
            isCalculated = true;
    }
    else
    {
        return true;
    }

    return isCalculated;
}

template <typename ElementType>
void Starks<ElementType>::setCommitCalculated(uint64_t id) {
    commitsCalculated[id] = true;
}

template <typename ElementType>
void Starks<ElementType>::setSubproofValueCalculated(uint64_t id) {
    subProofValuesCalculated[id] = true;
}


template <typename ElementType>
void Starks<ElementType>::merkelizeMemory()
{
    uint64_t polsSize = starkInfo.mapTotalN + starkInfo.mapSectionsN["cm3"] * NExtended;
    uint64_t nrowsDGB = 2;
    for (uint64_t k = 0; k < polsSize; ++k)
    {
        if (polsSize % (nrowsDGB * 2) == 0)
        {
            nrowsDGB *= 2;
        }
    }
    uint64_t ncolsDGB = polsSize / nrowsDGB;
    assert(nrowsDGB * ncolsDGB == polsSize);
    uint64_t numElementsTreeDBG = MerklehashGoldilocks::getTreeNumElements(nrowsDGB);
    Goldilocks::Element *treeDBG = new Goldilocks::Element[numElementsTreeDBG];
    Goldilocks::Element rootDBG[4];
#ifdef __AVX512__
    PoseidonGoldilocks::merkletree_avx512(treeDBG, (Goldilocks::Element *)pAddress, ncolsDGB,
                                          nrowsDGB);
#else
    PoseidonGoldilocks::merkletree_avx(treeDBG, (Goldilocks::Element *)pAddress, ncolsDGB,
                                       nrowsDGB);
#endif
    MerklehashGoldilocks::root(&(rootDBG[0]), treeDBG, numElementsTreeDBG);
    std::cout << "rootDBG[0]: [ " << Goldilocks::toU64(rootDBG[0]) << " ]" << std::endl;
    std::cout << "rootDBG[1]: [ " << Goldilocks::toU64(rootDBG[1]) << " ]" << std::endl;
    std::cout << "rootDBG[2]: [ " << Goldilocks::toU64(rootDBG[2]) << " ]" << std::endl;
    std::cout << "rootDBG[3]: [ " << Goldilocks::toU64(rootDBG[3]) << " ]" << std::endl;
    delete[] treeDBG;
}

template <typename ElementType>
void Starks<ElementType>::printPol(Goldilocks::Element* pol, uint64_t dim)
{
    Polinomial p = Polinomial(pol, N, dim, dim);
    MerkleTreeGL *mt_ = new MerkleTreeGL(merkleTreeArity, true, N, dim, pol);
    mt_->merkelize();

    Goldilocks::Element root[4];
    cout << "--------------------" << endl;
    cout << "HELLO: " << endl;
    mt_->getRoot(&root[0]);
    cout << "--------------------" << endl;

    for(uint64_t i = 0; i < N; ++i) {
        if(dim == 3) {
            cout << i << " [" << Goldilocks::toString(p[i][0]) << ", " << Goldilocks::toString(p[i][1]) << ", " << Goldilocks::toString(p[i][2]) << " ]" << endl; 
        } else {
            cout << i << " " << Goldilocks::toString(p[i][0]) << endl;
        }
    }

    delete mt_;
}

template <typename ElementType>
void Starks<ElementType>::printPolRoot(uint64_t polId, StepsParams &params)
{
    PolMap polInfo = starkInfo.cmPolsMap[polId];
    Polinomial p;
    starkInfo.getPolynomial(p, params.pols, true, polId, N);

    Polinomial pCol;
    Goldilocks::Element *pBuffCol = new Goldilocks::Element[polInfo.dim * N];
    pCol.potConstruct(pBuffCol, N, polInfo.dim, polInfo.dim);
    Polinomial::copy(pCol, p);

    MerkleTreeGL *mt_ = new MerkleTreeGL(merkleTreeArity, true, N, polInfo.dim, pBuffCol);
    mt_->merkelize();

    Goldilocks::Element root[4];
    cout << "--------------------" << endl;
    cout << "NAME: " << polInfo.name << endl;
    mt_->getRoot(&root[0]);
    cout << "--------------------" << endl;

    delete mt_;
    delete pBuffCol;
}

template <typename ElementType>
void *Starks<ElementType>::ffi_create_steps_params(Goldilocks::Element *pChallenges, Goldilocks::Element *pSubproofValues, Goldilocks::Element *pEvals, Goldilocks::Element *pPublicInputs)
{
    StepsParams *params = new StepsParams{
        pols : mem,
        constPols : constPols.pConstPolsAddress,
        constPolsExtended : constPols.pConstPolsAddressExtended,
        challenges : pChallenges,
        subproofValues : pSubproofValues,
        evals : pEvals,
        zi : zi,
        publicInputs : pPublicInputs,
    };

    return params;
}

template <typename ElementType>
void Starks<ElementType>::ffi_extend_and_merkelize(uint64_t step, StepsParams *params, FRIProof<ElementType> *proof)
{
    extendAndMerkelize(step, *params, *proof);
}

template <typename ElementType>
void Starks<ElementType>::ffi_treesGL_get_root(uint64_t index, ElementType *dst)
{
    treesGL[index]->getRoot(dst);
}

template <typename ElementType>
void *Starks<ElementType>::ffi_get_vector_pointer(char *name)
{
    if (strcmp(name, "commitsCalculated") == 0)
    {
        return &this->commitsCalculated;
    }
    else if (strcmp(name, "subProofValuesCalculated") == 0)
    {
        return &this->subProofValuesCalculated;
    }
    else
    {
        return NULL;
    }
}

template <typename ElementType>
void Starks<ElementType>::ffi_set_commit_calculated(uint64_t id) {
    setCommitCalculated(id);
}

template <typename ElementType>
void Starks<ElementType>::ffi_set_subproofvalue_calculated(uint64_t id) {
    setSubproofValueCalculated(id);
}

template <typename ElementType>
void Starks<ElementType>::calculateS(Polinomial &s, Polinomial &den, Goldilocks::Element multiplicity)
    {
        uint64_t size = den.degree();

        Polinomial denI(size, 3);
        Polinomial checkVal(1, 3);

        Polinomial::batchInverse(denI, den);
        
        Polinomial::mulElement(s, 0, denI, 0, multiplicity);
        
        for (uint64_t i = 1; i < size; i++)
        {
            Polinomial tmp(1, 3);
            Polinomial::mulElement(tmp, 0, denI, i, multiplicity);
            Polinomial::addElement(s, i, s, i - 1, tmp, 0);
        }

        Polinomial tmp(1, 3);
        Polinomial::mulElement(tmp, 0, denI, size - 1, multiplicity);
        Polinomial::addElement(checkVal, 0, s, size - 1, tmp, 0);
        
        zkassert(Goldilocks3::isZero((Goldilocks3::Element &)*checkVal[0]));
    }