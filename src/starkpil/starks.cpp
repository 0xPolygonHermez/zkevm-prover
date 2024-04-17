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
    while((nttOffsetHelper.second * nBlocks < buffHelperElements) ||  (nCols > 256*nBlocks) ) {
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
    while((nttOffsetHelper.second * nBlocks < buffHelperElements) || (nCols > 256*nBlocks) ) {
        nBlocks++;
    }

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);
    nttExtended.INTT(params.q_2ns, params.q_2ns, NExtended, starkInfo.qDim, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);

    for (uint64_t p = 0; p < starkInfo.qDeg; p++)
    {   
        __m256i sigma = _mm256_set1_epi64x(S[p].fe);
        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i += nrowsBatch)
        {
            Goldilocks3::Element_avx tmp_; 
            Goldilocks3::load_avx(tmp_, &params.q_2ns[(p * N + i) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION));
            Goldilocks3::op_31_avx(2, tmp_, tmp_, sigma);
            Goldilocks3::store_avx(&cmQ[(i * starkInfo.qDeg + p) * FIELD_EXTENSION],starkInfo.qDeg * FIELD_EXTENSION, tmp_);
        }
        // for(uint64_t i = 0; i < N; i++)
        // { 
        //     Goldilocks3::mul((Goldilocks3::Element &)cmQ[(i * starkInfo.qDeg + p) * FIELD_EXTENSION], (Goldilocks3::Element &)params.q_2ns[(p * N + i) * FIELD_EXTENSION], S[p]);
        // }
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
        for (uint64_t k = 0; k < NExtended; k += nrowsBatch)
        {
            __m256i x_k;
            Goldilocks::load_avx(x_k, &x[k], uint64_t(1));
            Goldilocks3::Element_avx tmp_; 
            Goldilocks3::op_31_avx(3, tmp_, xis_, x_k);
            Goldilocks3::store_avx(&params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION], FIELD_EXTENSION, tmp_);
        }
        // for (uint64_t k = 0; k < NExtended; k++)
        // {
        //     Goldilocks3::sub((Goldilocks3::Element &)(params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION]), x[k], (Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]));
        // }
    }

    Polinomial xDivXSubXi_(params.xDivXSubXi, NExtended * starkInfo.openingPoints.size(), FIELD_EXTENSION, FIELD_EXTENSION);
    Polinomial::batchInverseParallel(xDivXSubXi_, xDivXSubXi_);

    for (uint64_t i = 0; i < starkInfo.openingPoints.size(); ++i)
    {
#pragma omp parallel for
        for (uint64_t k = 0; k < NExtended; k += nrowsBatch)
        {
            __m256i x_k;
            Goldilocks::load_avx(x_k, &x[k], uint64_t(1));
            Goldilocks3::Element_avx tmp_; 
            Goldilocks3::load_avx(tmp_, &params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION], uint64_t(FIELD_EXTENSION));
            Goldilocks3::op_31_avx(2, tmp_, tmp_, x_k);
            Goldilocks3::store_avx(&params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION], FIELD_EXTENSION, tmp_);
        }
        // for (uint64_t k = 0; k < NExtended; k++)
        // {
        //     Goldilocks3::mul((Goldilocks3::Element &)(params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION]), (Goldilocks3::Element &)(params.xDivXSubXi[(k + i * NExtended) * FIELD_EXTENSION]), x[k]);
        // }
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
    vector<bool> transposeSrcPolinomial(starkInfo.cmPolsMap.size(), false);
    
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
                if(hintField.operand != opType::tmp && hintField.operand != opType::cm) continue;
                PolMap polInfo = starkInfo.cmPolsMap[hintField.id];
                if (polInfo.stage != "tmpExp") {
                    transposeSrcPolinomial[hintField.id] = true;
                    Polinomial p = starkInfo.getPolinomial(params.pols, hintField.id, N);
                    polynomials[hintField.id].potConstruct(&params.pols[starkInfo.mapOffsetsPolsHints[step - 1][hintField.id]], p.degree(), p.dim(), p.dim());
                } else {
                    polynomials[hintField.id] = starkInfo.getPolinomial(params.pols, hintField.id, N); 
                }
            }

            for (uint64_t i = 0; i < dstFields.size(); i++)
            {
                HintField hintField = hint.fields[dstFields[i]];
                if(hintField.operand == opType::tmp || hintField.operand == opType::cm) {
                    PolMap polInfo = starkInfo.cmPolsMap[hintField.id];
                    if (polInfo.stage != "tmpExp") {
                        isDstPolinomialCalculated[hintField.id] = true;
                        Polinomial p = starkInfo.getPolinomial(params.pols, hintField.id, N);
                        polynomials[hintField.id].potConstruct(&params.pols[starkInfo.mapOffsetsPolsHints[step - 1][hintField.id]], p.degree(), p.dim(), p.dim());
                    } else {
                        isDstPolinomialCalculated[hintField.id] = true;
                        polynomials[hintField.id] = starkInfo.getPolinomial(params.pols, hintField.id, N);
                    } 
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
    for (uint64_t i = 0; i < transposeSrcPolinomial.size(); ++i)
    {
        if(transposeSrcPolinomial[i]) {
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
            if(starkInfo.cmPolsMap[i].stage != "tmpExp") {
                Polinomial p = starkInfo.getPolinomial(params.pols, i, N);
                Polinomial::copy(p, polynomials[i]);
            }
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
    int size_thread = size_eval * FIELD_EXTENSION;
    Goldilocks::Element *evals_acc = &params.pols[starkInfo.mapOffsets[std::make_pair("evals", true)]];
    memset(&evals_acc[0], 0, num_threads * size_thread * sizeof(Goldilocks::Element));
#pragma omp parallel
    {
        int thread_idx = omp_get_thread_num();
#pragma omp for
        for (uint64_t k = 0; k < N; k++)
        {
            for (uint64_t i = 0; i < size_eval; i++)
            {
                int index = openingPos[i];
                Goldilocks3::Element res;
                if (ordPols[i].dim() == 1) {
                    Goldilocks3::mul(res, (Goldilocks3::Element &)(LEv[(index + k * starkInfo.openingPoints.size()) * FIELD_EXTENSION]), *ordPols[i][k * (1 << extendBits)]);
                } else {
                    Goldilocks3::mul(res, (Goldilocks3::Element &)(LEv[(index + k * starkInfo.openingPoints.size()) * FIELD_EXTENSION]), (Goldilocks3::Element &)(*ordPols[i][k * (1 << extendBits)]));
                }
                Goldilocks3::add((Goldilocks3::Element &)(evals_acc[thread_idx * size_thread + i * FIELD_EXTENSION]), (Goldilocks3::Element &)(evals_acc[thread_idx * size_thread + i * FIELD_EXTENSION]), res);
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
            std::memcpy((Goldilocks3::Element &)(evals_acc[i * FIELD_EXTENSION]), sum, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
#pragma omp single
        for (uint64_t i = 0; i < size_eval; ++i)
        {
            std::memcpy((Goldilocks3::Element &)(params.evals[indx[i] * FIELD_EXTENSION]), (Goldilocks3::Element &)(evals_acc[i * FIELD_EXTENSION]), FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
    delete[] ordPols;
}

template <typename ElementType>
void Starks<ElementType>::cleanSymbolsCalculated() {
    std::fill(publicsCalculated.begin(), publicsCalculated.end(), false);
    std::fill(subProofValuesCalculated.begin(), subProofValuesCalculated.end(), false);
    std::fill(challengesCalculated.begin(), challengesCalculated.end(), false);
    std::fill(witnessCalculated.begin(), witnessCalculated.end(), false);
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