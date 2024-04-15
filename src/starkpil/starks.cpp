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

    cleanSymbolsCalculated();

    // Initialize vars
    TimerStart(STARK_INITIALIZATION);

    TranscriptType transcript(merkleTreeArity, merkleTreeCustom);

    Goldilocks::Element* evals = new Goldilocks::Element[starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* challenges = new Goldilocks::Element[starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* subproofValues = new Goldilocks::Element[starkInfo.nSubProofValues * FIELD_EXTENSION];

    StepsParams params = {
        pols : mem,
        pConstPols : pConstPols,
        pConstPols2ns : pConstPols2ns,
        challenges : challenges,
        subproofValues : subproofValues,
        evals : evals,
        x_n : x_n,
        x_2ns : x_2ns,
        zi : zi,
        xDivXSubXi : xDivXSubXi,
        publicInputs : publicInputs,
        q_2ns : q_2ns,
        f_2ns : f_2ns,
    };

    for (uint64_t i = 0; i < starkInfo.mapSectionsN["cm1"]; ++i)
    {
        setSymbolCalculated(opType::cm, i);
    }

    for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
        setSymbolCalculated(opType::public_, i);
    }

    TimerStopAndLog(STARK_INITIALIZATION);

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

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

    for (uint64_t step = 1; step <= starkInfo.nStages; step++)
    {
        TimerStartExpr(STARK_STEP, step);
        for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
        {
            if(starkInfo.challengesMap[i].stageNum == step) {
                getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
                setSymbolCalculated(opType::challenge, i);
            }
        }

        computeStage(step, params, proof, transcript, chelpersSteps);

        if (debug)
        {
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
        if(starkInfo.challengesMap[i].stageNum == starkInfo.nStages + 1) {
            getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
            setSymbolCalculated(opType::challenge, i);
        }
    }
    
    computeStage(starkInfo.nStages + 1, params, proof, transcript, chelpersSteps);

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
        if(starkInfo.challengesMap[i].stageNum == starkInfo.nStages + 2) {
            getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
            setSymbolCalculated(opType::challenge, i);
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
        if(starkInfo.challengesMap[i].stageNum == starkInfo.nStages + 3) {
            getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
            setSymbolCalculated(opType::challenge, i);
        }
    }

    computeFRIPol(starkInfo.nStages + 2, params, chelpersSteps);

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    for (uint64_t step = 0; step < starkInfo.starkStruct.steps.size(); step++)
    {
        Goldilocks::Element challenge[FIELD_EXTENSION];
        getChallenge(transcript, *challenge);
        computeFRIFolding(proof, params.f_2ns, step, challenge);
        if (step < starkInfo.starkStruct.steps.size() - 1)
        {
            addTranscript(transcript, &proof.proofs.fri.trees[step + 1].root[0], nFieldElements);
        }
        else
        {
            if(starkInfo.starkStruct.hashCommits) {
                ElementType hash[nFieldElements];
                calculateHash(hash, params.f_2ns, (1 << starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                addTranscript(transcript, hash, nFieldElements);
            } else {
                addTranscriptGL(transcript, params.f_2ns, (1 << starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            }    
        }
    }

    uint64_t friQueries[starkInfo.starkStruct.nQueries];

    Goldilocks::Element challenge[FIELD_EXTENSION];
    getChallenge(transcript, *challenge);
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
void Starks<ElementType>::calculateExpressions(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps)
{
    TimerStartExpr(STARK_CALCULATE_EXPS_STEP, step);
    if (chelpers.stagesInfo[step - 1].nOps > 0)
    {
        bool domainExtended = step > starkInfo.nStages ? true : false;
        chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgs, chelpers.stagesInfo[step - 1], nrowsBatch, domainExtended);
        for(uint64_t i = 0; i < chelpers.stagesInfo[step - 1].nCmPolsCalculated; i++) {
            uint64_t cmPolCalculatedId = chelpers.cHelpersArgs.cmPolsCalculatedIds[chelpers.stagesInfo[step - 1].cmPolsCalculatedOffset + i];
            setSymbolCalculated(opType::cm, cmPolCalculatedId);
        }
    }
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_STEP, step);
}

template <typename ElementType>
void Starks<ElementType>::calculateExpression(uint64_t id, StepsParams &params, CHelpersSteps *chelpersSteps)
{
    uint64_t expId = chelpers.expressionsInfo[id].expId;
    TimerStartExpr(STARK_CALCULATE_EXPRESSION, expId);
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgsExpressions, chelpers.expressionsInfo[id], nrowsBatch, false);
    for(uint64_t i = 0; i < chelpers.expressionsInfo[id].nCmPolsCalculated; i++) {
        uint64_t cmPolCalculatedId = chelpers.cHelpersArgsExpressions.cmPolsCalculatedIds[chelpers.expressionsInfo[id].cmPolsCalculatedOffset + i];
        setSymbolCalculated(opType::cm, cmPolCalculatedId);
    }
    TimerStopAndLogExpr(STARK_CALCULATE_EXPRESSION, expId);
}

template <typename ElementType>
void Starks<ElementType>::calculateConstraint(uint64_t constraintId, StepsParams &params, CHelpersSteps *chelpersSteps)
{
    TimerStartExpr(STARK_CALCULATE_CONSTRAINT, constraintId);
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgsDebug, chelpers.constraintsInfoDebug[constraintId], nrowsBatch, false);
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
    while(nttOffsetHelper.second * nBlocks < buffHelperElements ||  nCols / nBlocks > 256 ) {
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
void Starks<ElementType>::computeStage(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof, TranscriptType &transcript, CHelpersSteps *chelpersSteps)
{
    calculateExpressions(step, params, chelpersSteps);

    calculateHints(step, params);

    if (step <= starkInfo.nStages)
    {
        TimerStartExpr(STARK_TRY_CALCULATE_EXPS_STEP, step);
        uint64_t symbolsToBeCalculated = isStageCalculated(step);
        while (symbolsToBeCalculated > 0)
        {
            for (uint64_t i = 0; i < chelpers.expressionsInfo.size(); i++) {
                if(chelpers.expressionsInfo[i].stage == step) {
                    bool isCalculated = true;
                    for(uint64_t j = 0; j < chelpers.expressionsInfo[i].nCmPolsCalculated; j++) {
                        uint64_t cmPolCalculatedId = chelpers.cHelpersArgsExpressions.cmPolsCalculatedIds[chelpers.expressionsInfo[i].cmPolsCalculatedOffset + j];
                        if (!isSymbolCalculated(opType::cm, cmPolCalculatedId)) {
                            isCalculated = false;
                            break;
                        }
                    }
                    if (!isCalculated && canExpressionBeCalculated(chelpers.expressionsInfo[i])) {
                        calculateExpression(i, params, chelpersSteps);
                    }
                }
            }
            calculateHints(step, params);
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
    

    if (debug && step <= starkInfo.nStages)
    {
        for (uint64_t i = 0; i < chelpers.constraintsInfoDebug.size(); i++)
        {
            if(chelpers.constraintsInfoDebug[i].stage == step) {
                calculateConstraint(i, params, chelpersSteps);
            }
        }
    }
    else
    {
        if (step <= starkInfo.nStages)
        {
            extendAndMerkelize(step, params, proof);
        }
        else
        {
            computeQ(step, params, proof);
        }
    }

    if(step == starkInfo.nStages) {
        proof.proofs.setSubAirValues(params.subproofValues);
    }
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
    while(nttOffsetHelper.second * nBlocks < buffHelperElements || nCols / nBlocks > 256 ) {
        nBlocks++;
    }

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);
    nttExtended.INTT(params.q_2ns, params.q_2ns, NExtended, starkInfo.qDim, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);

    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);

    uint64_t stride = 2048;
#pragma omp parallel for
    for (uint64_t ii = 0; ii < N; ii += stride)
    {
        Goldilocks::Element curS = Goldilocks::one();
        for (uint64_t p = 0; p < starkInfo.qDeg; p++)
        {
            for (uint64_t k = ii; k < min(N, ii + stride); ++k)
            {
                Goldilocks3::mul((Goldilocks3::Element &)cmQ[(k * starkInfo.qDeg + p) * FIELD_EXTENSION], (Goldilocks3::Element &)params.q_2ns[(p * N + k) * FIELD_EXTENSION], curS);
            }
            curS = Goldilocks::mul(curS, shiftIn);
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
        if(starkInfo.cmPolsMap[i].stageNum == step) {
            setSymbolCalculated(opType::cm, i);
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::computeEvals(StepsParams &params, FRIProof<ElementType> &proof)
{
    auto evalsStage = starkInfo.nStages + 2;
    auto xiChallenge = std::find_if(starkInfo.challengesMap.begin(), starkInfo.challengesMap.end(), [evalsStage](const PolMap& c) {
        return c.stageNum == evalsStage && c.stageId == 0;
    });

    uint64_t xiChallengeIndex = std::distance(starkInfo.challengesMap.begin(), xiChallenge);

    TimerStart(STARK_CALCULATE_LEv);
    
    Goldilocks::Element* LEv = &params.xDivXSubXi[0];

    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
        Goldilocks::Element w = Goldilocks::one();
        uint64_t offset = i * FIELD_EXTENSION * N;
        Goldilocks3::one((Goldilocks3::Element &)LEv[offset]);
        uint64_t openingAbs = starkInfo.openingPoints[i] < 0 ? -starkInfo.openingPoints[i] : starkInfo.openingPoints[i];
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w = w * Goldilocks::w(starkInfo.starkStruct.nBits);
        }

        if (starkInfo.openingPoints[i] < 0)
        {
            w = Goldilocks::one() / w;
        }

        Goldilocks3::Element xi;
        int offset_xi = xiChallengeIndex * FIELD_EXTENSION;
        xi[0] = params.challenges[offset_xi] * w * Goldilocks::inv(Goldilocks::shift());
        xi[1] = params.challenges[offset_xi + 1] * w * Goldilocks::inv(Goldilocks::shift());
        xi[2] = params.challenges[offset_xi + 2] * w * Goldilocks::inv(Goldilocks::shift());

        for (uint64_t k = 1; k < N; k++)
        {
            Goldilocks3::mul((Goldilocks3::Element &)(LEv[k * FIELD_EXTENSION + offset]), (Goldilocks3::Element &)(LEv[(k - 1) * FIELD_EXTENSION + offset]), xi);
        }

        ntt.INTT(&LEv[offset], &LEv[offset], N, FIELD_EXTENSION);
    }

    TimerStopAndLog(STARK_CALCULATE_LEv);

    TimerStart(STARK_CALCULATE_EVALS);
    evmap(params, LEv);
    proof.proofs.setEvals(params.evals);
    TimerStopAndLog(STARK_CALCULATE_EVALS);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIPol(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps)
{

    auto evalsStage = starkInfo.nStages + 2;
    auto xiChallenge = std::find_if(starkInfo.challengesMap.begin(), starkInfo.challengesMap.end(), [evalsStage](const PolMap& c) {
        return c.stageNum == evalsStage && c.stageId == 0;
    });

    uint64_t xiChallengeIndex = std::distance(starkInfo.challengesMap.begin(), xiChallenge);

    TimerStart(STARK_CALCULATE_XDIVXSUB);

    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

#pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
        uint64_t openingAbs = starkInfo.openingPoints[i] < 0 ? -starkInfo.openingPoints[i] : starkInfo.openingPoints[i];
        Goldilocks::Element w = Goldilocks::one();
        
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w = w * Goldilocks::w(starkInfo.starkStruct.nBits);
        }

        if (starkInfo.openingPoints[i] < 0)
        {
            w = Goldilocks::one() / w;
        }

        Goldilocks3::Element xi;
        int offset_xi = xiChallengeIndex * FIELD_EXTENSION;
        xi[0] = params.challenges[offset_xi] * w;
        xi[1] = params.challenges[offset_xi + 1] * w;
        xi[2] = params.challenges[offset_xi + 2] * w;

#pragma omp parallel for
        for (uint64_t k = 0; k < (N << extendBits); k++)
        {
            params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION] = x[k] - xi[0];
            params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION + 1] = -xi[1];
            params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION + 2] = -xi[2];
        }
    }

    Polinomial xDivXSubXi_(params.xDivXSubXi, NExtended * starkInfo.openingPoints.size(), FIELD_EXTENSION, FIELD_EXTENSION);
    Polinomial::batchInverseParallel(xDivXSubXi_, xDivXSubXi_);
    // Goldilocks3::batchInverse((Goldilocks3::Element *)(&params.xDivXSubXi[0]), (Goldilocks3::Element *)(&params.xDivXSubXi[0]), NExtended * starkInfo.openingPoints.size());

#pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
        for (uint64_t k = 0; k < (N << extendBits); k++)
        {
            params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION] = params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION] * x[k];
            params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION + 1] = params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION + 1] * x[k];
            params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION + 2] = params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION + 2] * x[k];
        }
    }
    TimerStopAndLog(STARK_CALCULATE_XDIVXSUB);

    calculateExpressions(step, params, chelpersSteps);
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
        if (hint.fields[srcFields[i]].operand == opType::number) continue;
        if (!isSymbolCalculated(hint.fields[srcFields[i]].operand, hint.fields[srcFields[i]].id)) {
            return false;
        }
    }

    return true;
}

template <typename ElementType>
void Starks<ElementType>::calculateHints(uint64_t step, StepsParams &params)
{
    Polinomial* polynomials = new Polinomial[starkInfo.cmPolsMap.size()];
    vector<bool> isSrcPolinomial(starkInfo.cmPolsMap.size(), false);
    
    vector<bool> isDstPolinomialCalculated(starkInfo.cmPolsMap.size(), false);
    vector<bool> isSubProofValueCalculated(starkInfo.nSubProofValues, false);

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
                if (hintField.operand == opType::cm || hintField.operand == opType::tmp)
                {
                    isSrcPolinomial[hintField.id] = true;
                    Polinomial p = starkInfo.getPolinomial(params.pols, hintField.id, N);
                    polynomials[hintField.id].potConstruct(&params.pols[starkInfo.mapOffsetsPolsHints[step - 1][hintField.id]], p.degree(), p.dim(), p.dim());
                }
            }

            for (uint64_t i = 0; i < dstFields.size(); i++)
            {
                HintField hintField = hint.fields[dstFields[i]];
                if (hintField.operand == opType::cm || hintField.operand == opType::tmp)
                {
                    isDstPolinomialCalculated[hintField.id] = true;
                    Polinomial p = starkInfo.getPolinomial(params.pols, hintField.id, N);
                    polynomials[hintField.id].potConstruct(&params.pols[starkInfo.mapOffsetsPolsHints[step - 1][hintField.id]], p.degree(), p.dim(), p.dim());
                    
                } else if(hintField.operand == opType::subproofvalue) {
                    isSubProofValueCalculated[hintField.id] = true;
                }
            }
        }
    }
    TimerStopAndLogExpr(STARK_PREPARE_HINTS_STEP, step);

    if (hintsToCalculate.size() == 0)
        return;


    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);
    for (uint64_t i = 0; i < isSrcPolinomial.size(); ++i)
    {
        if(isSrcPolinomial[i]) {
            Polinomial p = starkInfo.getPolinomial(params.pols, i, N);
            Polinomial::copy(polynomials[i], p);
        }
    }
    TimerStopAndLogExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);

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
        polsNames.insert(polsNames.end(), srcPolsNames.begin(), srcPolsNames.end());
        polsNames.insert(polsNames.end(), dstPolsNames.begin(), dstPolsNames.end());

        // Prepare polynomials map to be sent to the hint
        std::map<std::string, Polinomial *> polynomialsHint;
        for (const auto &polName : polsNames)
        {
            const auto &hintField = hint.fields[polName];
            if (hintField.operand == opType::cm || hintField.operand == opType::tmp)
            {
                polynomialsHint[polName] = &polynomials[hintField.id];
            }
        }

        // Resolve hint
        void *extra_mem_ptr = nullptr;

        // At the time being the only hint that requires extra memory is h1h2
        // so we have already allocated the memory for it.
        // This must be changed in the future to be more generic.
        if(hintHandler->getMemoryNeeded(N) > 0) {
            extra_mem_ptr = &params.pols[starkInfo.offsetsExtraMemoryHints[hintsToCalculate[i]]];
        }

        hintHandler->resolveHint(N, params, hint, polynomialsHint, extra_mem_ptr);
    }

    TimerStopAndLogExpr(STARK_CALCULATE_HINTS_STEP, step);

    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_2_STEP, step);
    for (uint64_t i = 0; i < isDstPolinomialCalculated.size(); ++i)
    {
        if(isDstPolinomialCalculated[i]) {
            setSymbolCalculated(opType::cm, i);
            Polinomial p = starkInfo.getPolinomial(params.pols, i, N);
            Polinomial::copy(p, polynomials[i]);
        }
    }
    TimerStopAndLogExpr(STARK_CALCULATE_TRANSPOSE_2_STEP, step);

    for (uint64_t i = 0; i < isSubProofValueCalculated.size(); ++i)
    {
        if(isSubProofValueCalculated[i]) {
            setSymbolCalculated(opType::subproofvalue, i);
        }
    }

    delete[] polynomials;
}

template <typename ElementType>
void Starks<ElementType>::evmap(StepsParams &params, Goldilocks::Element *LEv)
{
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    u_int64_t size_eval = starkInfo.evMap.size();

    // Order polinomials by address, note that there are collisions!
    map<uintptr_t, vector<uint>> map_offsets;
    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = starkInfo.evMap[i];
        if (ev.type == EvMap::eType::_const)
        {
            map_offsets[reinterpret_cast<std::uintptr_t>(&((Goldilocks::Element *)params.pConstPols2ns->address())[ev.id])].push_back(i);
        }
        else if (ev.type == EvMap::eType::cm || ev.type == EvMap::eType::q)
        {
            Polinomial pol = starkInfo.getPolinomial(params.pols,ev.id, NExtended);
            map_offsets[reinterpret_cast<std::uintptr_t>(pol.address())].push_back(i);
        }
        else
        {
            throw std::invalid_argument("Invalid ev type: " + ev.type);
        }
    }

    Polinomial *ordPols = new Polinomial[size_eval];
    vector<uint> openingPos(size_eval);
    vector<uint> indx(size_eval);
    //   build and store ordered polinomials that need to be computed
    uint kk = 0;
    for (std::map<uintptr_t, std::vector<uint>>::const_iterator it = map_offsets.begin(); it != map_offsets.end(); ++it)
    {
        for (std::vector<uint>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
        {
            EvMap ev = starkInfo.evMap[*it2];
            if (ev.type == EvMap::eType::_const)
            {
                ordPols[kk].potConstruct(&((Goldilocks::Element *)params.pConstPols2ns->address())[ev.id], params.pConstPols2ns->degree(), 1, params.pConstPols2ns->numPols());
            }
            else if (ev.type == EvMap::eType::cm || ev.type == EvMap::eType::q)
            {
               ordPols[kk] = starkInfo.getPolinomial(params.pols, ev.id, NExtended);
            }
            int64_t prime = ev.prime;
            auto openingPoint = std::find_if(starkInfo.openingPoints.begin(), starkInfo.openingPoints.end(), [prime](int p) { return p == prime; });
            if(openingPoint == starkInfo.openingPoints.end()) {
                zklog.error("Opening point not found");
                exitProcess();
                exit(-1);
            }
            openingPos[kk] = std::distance(starkInfo.openingPoints.begin(), openingPoint);
            indx[kk] = *it2;
            ++kk;
        }
    }
    assert(kk == size_eval);
    // Build buffer for partial results of the matrix-vector product (columns distribution)  .
    int num_threads = omp_get_max_threads();
    Goldilocks::Element **evals_acc = (Goldilocks::Element **)malloc(num_threads * sizeof(Goldilocks::Element *));
    for (int i = 0; i < num_threads; ++i)
    {
        evals_acc[i] = (Goldilocks::Element *)malloc(size_eval * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }
#pragma omp parallel
    {
        int thread_idx = omp_get_thread_num();
        for (uint64_t i = 0; i < size_eval * FIELD_EXTENSION; ++i)
        {
            evals_acc[thread_idx][i] = Goldilocks::zero();
        }
#pragma omp for
        for (uint64_t k = 0; k < N; k++)
        {
            for (uint64_t i = 0; i < size_eval; i++)
            {
                int index = openingPos[i];
                Goldilocks3::Element res;
                if (ordPols[i].dim() == 1) {
                    res[0] = LEv[(index * N + k) * FIELD_EXTENSION] * ordPols[i][k << extendBits][0];
                    res[1] = LEv[(index * N + k) * FIELD_EXTENSION + 1] * ordPols[i][k << extendBits][0];
                    res[2] = LEv[(index * N + k) * FIELD_EXTENSION + 2] * ordPols[i][k << extendBits][0];
                } else {
                    Goldilocks3::mul(res, (Goldilocks3::Element &)(LEv[(index * N + k) * FIELD_EXTENSION]), (Goldilocks3::Element &)(ordPols[i][k << extendBits][0]));
                }
                evals_acc[thread_idx][i * FIELD_EXTENSION] = evals_acc[thread_idx][i * FIELD_EXTENSION] + res[0];
                evals_acc[thread_idx][i * FIELD_EXTENSION + 1] = evals_acc[thread_idx][i * FIELD_EXTENSION + 1] + res[1];
                evals_acc[thread_idx][i * FIELD_EXTENSION + 2] = evals_acc[thread_idx][i * FIELD_EXTENSION + 2] + res[2];
            }
        }
#pragma omp for
        for (uint64_t i = 0; i < size_eval; ++i)
        {
            Goldilocks::Element sum0 = Goldilocks::zero();
            Goldilocks::Element sum1 = Goldilocks::zero();
            Goldilocks::Element sum2 = Goldilocks::zero();
            int offset = i * FIELD_EXTENSION;
            for (int k = 0; k < num_threads; ++k)
            {
                sum0 = sum0 + evals_acc[k][offset];
                sum1 = sum1 + evals_acc[k][offset + 1];
                sum2 = sum2 + evals_acc[k][offset + 2];
            }
            evals_acc[0][offset] = sum0;
            evals_acc[0][offset + 1] = sum1;
            evals_acc[0][offset + 2] = sum2;
        }
#pragma omp single
        for (uint64_t i = 0; i < size_eval; ++i)
        {
            int offset = i * FIELD_EXTENSION;
            int offset_ = indx[i] * FIELD_EXTENSION;
            params.evals[offset_] = evals_acc[0][offset];
            params.evals[offset_ + 1] = evals_acc[0][offset + 1];
            params.evals[offset_ + 2] = evals_acc[0][offset + 2];
        }
    }
    delete[] ordPols;
    for (int i = 0; i < num_threads; ++i)
    {
        free(evals_acc[i]);
    }
    free(evals_acc);
}

template <typename ElementType>
void Starks<ElementType>::cleanSymbolsCalculated() {
    for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
        publicsCalculated[i] = false;
    }

    for(uint64_t i = 0; i < starkInfo.nSubProofValues; ++i) {
        subProofValuesCalculated[i] = false;
    }

    for(uint64_t i = 0; i < starkInfo.challengesMap.size(); ++i) {
        challengesCalculated[i] = false;
    }

    for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); ++i) {
        witnessCalculated[i] = false;
    }
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
        if(starkInfo.cmPolsMap[i].stageNum != step) continue;
        if(!isSymbolCalculated(opType::cm, i)) symbolsToBeCalculated++;
    }

    for(uint64_t i = 0; i < starkInfo.challengesMap.size(); i++) {
        if(starkInfo.challengesMap[i].stageNum != step) continue;
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
    if (operand == opType::const_)
    {
        if (constsCalculated[id])
            isCalculated = true;
    }
    else if (operand == opType::cm)
    {
        if (witnessCalculated[id])
            isCalculated = true;
    }
    else if (operand == opType::tmp)
    {
        if (witnessCalculated[id])
            isCalculated = true;
    }
    else if (operand == opType::public_)
    {
        if (publicsCalculated[id])
            isCalculated = true;
    }
    else if (operand == opType::subproofvalue)
    {
        if (subProofValuesCalculated[id])
            isCalculated = true;
    }
    else if (operand == opType::challenge)
    {
        if (challengesCalculated[id])
            isCalculated = true;
    }
    else
    {
        zklog.error("Invalid symbol type=" + operand);
        exitProcess();
        exit(-1);
    }

    return isCalculated;
}

template <typename ElementType>
void Starks<ElementType>::setSymbolCalculated(opType operand, uint64_t id)
{
    if (operand == opType::const_)
    {
        if (!constsCalculated[id])
            constsCalculated[id] = true;
    }
    else if (operand == opType::cm || operand == opType::tmp)
    {
        if (!witnessCalculated[id])
            witnessCalculated[id] = true;
    }
    else if (operand == opType::public_)
    {
        if (!publicsCalculated[id])
            publicsCalculated[id] = true;
    }
    else if (operand == opType::subproofvalue)
    {
        if (!subProofValuesCalculated[id])
            subProofValuesCalculated[id] = true;
    }
    else if (operand == opType::challenge)
    {
        if (!challengesCalculated[id])
            challengesCalculated[id] = true;
    }
    else
    {
        zklog.error("Invalid symbol type=" + operand);
        exitProcess();
        exit(-1);
    }
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
void Starks<ElementType>::printPolRoot(uint64_t polId, StepsParams &params)
{
    PolMap polInfo = starkInfo.cmPolsMap[polId];
    Polinomial p = starkInfo.getPolinomial(params.pols, polId, N);

    Polinomial pCol;
    Goldilocks::Element *pBuffCol = new Goldilocks::Element[p.dim() * N];
    pCol.potConstruct(pBuffCol, p.degree(), p.dim(), p.dim());
    Polinomial::copy(pCol, p);

    MerkleTreeGL *mt_ = new MerkleTreeGL(merkleTreeArity, true, N, p.dim(), pBuffCol);
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
        pConstPols : pConstPols,
        pConstPols2ns : pConstPols2ns,
        challenges : pChallenges,
        subproofValues : pSubproofValues,
        evals : pEvals,
        x_n : x_n,
        x_2ns : x_2ns,
        zi : zi,
        xDivXSubXi : xDivXSubXi,
        publicInputs : pPublicInputs,
        q_2ns : q_2ns,
        f_2ns : f_2ns,
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
    if (strcmp(name, "publicsCalculated") == 0)
    {
        return &this->publicsCalculated;
    }
    else if (strcmp(name, "constsCalculated") == 0)
    {
        return &this->constsCalculated;
    }
    else if (strcmp(name, "witnessCalculated") == 0)
    {
        return &this->witnessCalculated;
    }
    else if (strcmp(name, "subProofValuesCalculated") == 0)
    {
        return &this->subProofValuesCalculated;
    }
    else if (strcmp(name, "challengesCalculated") == 0)
    {
        return &this->challengesCalculated;
    }
    else
    {
        return NULL;
    }
}

template <typename ElementType>
void Starks<ElementType>::ffi_set_symbol_calculated(uint32_t operand, uint64_t id) {
    setSymbolCalculated((opType)operand, id);
}