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

    TranscriptType transcript(merkleTreeArity, merkleTreeCustom);

    Polinomial evals(starkInfo.evMap.size(), FIELD_EXTENSION);
    Polinomial challenges(starkInfo.challengesMap.size(), FIELD_EXTENSION);
    Polinomial subproofValues(starkInfo.nSubProofValues, FIELD_EXTENSION);

    Polinomial xDivXSubXi(starkInfo.openingPoints.size() * NExtended, FIELD_EXTENSION);

    StepsParams params = {
        pols : mem,
        pConstPols : pConstPols,
        pConstPols2ns : pConstPols2ns,
        challenges : challenges,
        subproofValues : subproofValues,
        x_n : x_n,
        x_2ns : x_2ns,
        zi : zi,
        evals : evals,
        xDivXSubXi : xDivXSubXi,
        publicInputs : publicInputs,
        q_2ns : &mem[starkInfo.mapOffsets[std::make_pair("q", true)]],
        f_2ns : &mem[starkInfo.mapOffsets[std::make_pair("f", true)]]
    };

    cm2Transposed.resize(starkInfo.cmPolsMap.size(), -1);
    publicsCalculated.resize(starkInfo.nPublics, true);

    subProofValuesCalculated.resize(starkInfo.nSubProofValues, false);
    challengesCalculated.resize(starkInfo.challengesMap.size(), false);

    witnessCalculated.resize(starkInfo.cmPolsMap.size(), false);
    for (uint64_t i = 0; i < starkInfo.mapSectionsN["cm1"]; ++i)
    {
        setSymbolCalculated(opType::cm, i);
    }

    TimerStopAndLog(STARK_INITIALIZATION);

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    if(!debug) {
        ElementType verkey[hashSize];
        treesGL[starkInfo.nStages + 1]->getRoot(verkey);
        addTranscript(transcript, &verkey[0], hashSize);
    }
    
    if(starkInfo.starkStruct.hashCommits) {
        ElementType hash[hashSize];
        calculateHash(hash, &publicInputs[0], starkInfo.nPublics);
        addTranscript(transcript, hash, hashSize);
    } else {
        addTranscriptGL(transcript, &publicInputs[0], starkInfo.nPublics);
    }

    for (uint64_t step = 1; step <= starkInfo.nStages; step++)
    {
        TimerStartExpr(STARK_STEP, step);
        computeStage(step, params, proof, transcript, chelpersSteps);
        TimerStopAndLogExpr(STARK_STEP, step);
    }

    if (debug) return;

    TimerStart(STARK_STEP_Q);
    computeStage(starkInfo.nStages + 1, params, proof, transcript, chelpersSteps);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stageNum == starkInfo.nStages + 2) {
            getChallenge(transcript, *params.challenges[i]);
            setSymbolCalculated(opType::challenge, i);
        }
    }

    computeEvals(params, proof);

    if(starkInfo.starkStruct.hashCommits) {
        ElementType hash[hashSize];
        calculateHash(hash, evals);
        addTranscript(transcript, hash, hashSize);
    } else {
        addTranscript(transcript, evals);
    }    

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stageNum == starkInfo.nStages + 3) {
            getChallenge(transcript, *params.challenges[i]);
            setSymbolCalculated(opType::challenge, i);
        }
    }

    Polinomial *friPol = computeFRIPol(starkInfo.nStages + 2, params, chelpersSteps);

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    for (uint64_t step = 0; step < starkInfo.starkStruct.steps.size(); step++)
    {
        Polinomial challenge(1, FIELD_EXTENSION);
        getChallenge(transcript, *challenge[0]);
        computeFRIFolding(proof, friPol[0], step, challenge);
        if (step < starkInfo.starkStruct.steps.size() - 1)
        {
            addTranscript(transcript, &proof.proofs.fri.trees[step + 1].root[0], hashSize);
        }
        else
        {
            if(starkInfo.starkStruct.hashCommits) {
                ElementType hash[hashSize];
                calculateHash(hash, *friPol);
                addTranscript(transcript, hash, hashSize);
            } else {
                addTranscript(transcript, *friPol);
            }    
        }
    }

    uint64_t friQueries[starkInfo.starkStruct.nQueries];

    Polinomial challenge(1, FIELD_EXTENSION);
    getChallenge(transcript, *challenge[0]);
    TranscriptType transcriptPermutation(merkleTreeArity, merkleTreeCustom);
    addTranscript(transcriptPermutation, challenge);
    transcriptPermutation.getPermutations(friQueries, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.steps[0].nBits);

    computeFRIQueries(proof, *friPol, friQueries);

    delete friPol;

    TimerStopAndLog(STARK_STEP_FRI);

    TimerStopAndLog(STARK_PROOF);
}

template <typename ElementType>
void Starks<ElementType>::calculateExpressions(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps)
{
    TimerStartExpr(STARK_CALCULATE_EXPS_STEP, step);
    if (chelpers.stagesInfo[step - 1].nOps > 0)
    {
        chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgs, chelpers.stagesInfo[step - 1]);
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
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgsExpressions, chelpers.expressionsInfo[id]);
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
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgsDebug, chelpers.constraintsInfoDebug[constraintId]);
    TimerStopAndLogExpr(STARK_CALCULATE_CONSTRAINT, constraintId);
}

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelize(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof)
{
    TimerStartExpr(STARK_LDE_AND_MERKLETREE_STEP, step);
    TimerStartExpr(STARK_LDE_STEP, step);

    std::string section = "cm" + to_string(step);

    std::pair<string, bool> nttBufferHelperSectionStart;
    if (step == 1 && starkInfo.mapOffsets[std::make_pair("tmpExp", false)] > starkInfo.mapOffsets[std::make_pair("cm1", true)])
    {
        nttBufferHelperSectionStart = std::make_pair("tmpExp", false);
    }
    else if (step == starkInfo.nStages && optimizeMemoryNTT)
    {
        nttBufferHelperSectionStart = std::make_pair("cm1", false);
    }
    else
    {
        nttBufferHelperSectionStart = std::make_pair("cm" + to_string(step + 1), true);
    }

    uint64_t nCols = starkInfo.mapSectionsN["cm" + to_string(step)];

    Goldilocks::Element *pBuff = &params.pols[starkInfo.mapOffsets[make_pair(section, false)]];
    Goldilocks::Element *pBuffExtended = &params.pols[starkInfo.mapOffsets[make_pair(section, true)]];
    Goldilocks::Element *pBuffHelper = &params.pols[starkInfo.mapOffsets[nttBufferHelperSectionStart]];

    ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols, pBuffHelper);
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
    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stageNum == step) {
            getChallenge(transcript, *params.challenges[i]);
            setSymbolCalculated(opType::challenge, i);
        }
    }

    calculateExpressions(step, params, chelpersSteps);

    calculateHints(step, params, chelpers.hints);

    if (step <= starkInfo.nStages)
    {
        TimerStartExpr(STARK_TRY_CALCULATE_EXPS_STEP, step);
        uint64_t symbolsToBeCalculated = isStageCalculated(step);
        while (symbolsToBeCalculated > 0)
        {
            for (uint64_t i = 0; i < chelpers.expressionsInfo.size(); i++) {
                if(chelpers.expressionsInfo[i].stage == step) {
                    bool isCalculated = true;
                    for(uint64_t i = 0; i < chelpers.expressionsInfo[i].nCmPolsCalculated; i++) {
                        uint64_t cmPolCalculatedId = chelpers.cHelpersArgsExpressions.cmPolsCalculatedIds[chelpers.stagesInfo[step - 1].cmPolsCalculatedOffset + i];
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

    if (debug)
    {
        Goldilocks::Element randomValues[hashSize] = {Goldilocks::fromU64(0), Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(3)};
        addTranscriptGL(transcript, randomValues, hashSize);
    }
    else
    {
        addTranscript(transcript, &proof.proofs.roots[step - 1][0], hashSize);
    }
}

template <typename ElementType>
void Starks<ElementType>::computeQ(uint64_t step, StepsParams &params, FRIProof<ElementType> &proof)
{

    uint64_t qDeg = 0;
    uint64_t qDim = 0;
    for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); ++i) {
        if(starkInfo.cmPolsMap[i].stageNum == step) {
            qDeg += 1;
            if(qDim == 0) qDim = starkInfo.cmPolsMap[i].dim;
        }       
    }

    Goldilocks::Element *pBuffQ = &params.pols[starkInfo.mapOffsets[std::make_pair("q", true)]];
    Polinomial qq1 = Polinomial(pBuffQ, NExtended, qDim, qDim);

    std::string section = "cm" + to_string(starkInfo.nStages + 1);
    Goldilocks::Element *pBuffExtended = &params.pols[starkInfo.mapOffsets[std::make_pair(section, true)]];
    uint64_t nCols = starkInfo.mapSectionsN["cm" + to_string(starkInfo.nStages + 1)];

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);
    nttExtended.INTT(pBuffQ, pBuffQ, NExtended, qDim);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);
    Polinomial qq2 = Polinomial(NExtended * qDeg, qDim, "qq2");

    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);

    uint64_t stride = 2048;
#pragma omp parallel for
    for (uint64_t ii = 0; ii < N; ii += stride)
    {
        Goldilocks::Element curS = Goldilocks::one();
        for (uint64_t p = 0; p < qDeg; p++)
        {
            for (uint64_t k = ii; k < min(N, ii + stride); ++k)
            {
                Goldilocks3::mul((Goldilocks3::Element &)*qq2[k * qDeg + p], (Goldilocks3::Element &)*qq1[p * N + k], curS);
            }
            curS = Goldilocks::mul(curS, shiftIn);
        }
    }
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_NTT_STEP, step);
    nttExtended.NTT(pBuffExtended, qq2.address(), NExtended, nCols);
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

    vector<uint64_t> openingPoints = starkInfo.openingPoints;

    Polinomial LEv(openingPoints.size() * N, FIELD_EXTENSION);

    Polinomial w(openingPoints.size(), FIELD_EXTENSION);
    Polinomial c_w(openingPoints.size(), FIELD_EXTENSION);
    Polinomial xi(openingPoints.size(), FIELD_EXTENSION);

    for (uint64_t i = 0; i < openingPoints.size(); ++i)
    {
        uint64_t offset = i * N;
        Goldilocks3::one((Goldilocks3::Element &)*LEv[offset]);
        uint64_t openingAbs = openingPoints[i] < 0 ? -openingPoints[i] : openingPoints[i];
        Goldilocks3::one((Goldilocks3::Element &)*w[i]);
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            Polinomial::mulElement(w, i, w, i, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));
        }

        if (openingPoints[i] < 0)
        {
            Polinomial::divElement(w, i, (Goldilocks::Element &)Goldilocks::one(), w, i);
        }

        Polinomial::mulElement(c_w, i, params.challenges, xiChallengeIndex, w, i);

        Polinomial::divElement(xi, i, c_w, i, (Goldilocks::Element &)Goldilocks::shift());

        for (uint64_t k = 1; k < N; k++)
        {
            Polinomial::mulElement(LEv, k + offset, LEv, k + offset - 1, xi, i);
        }

        ntt.INTT(LEv[offset], LEv[offset], N, 3);
    }

    TimerStopAndLog(STARK_CALCULATE_LEv);

    TimerStart(STARK_CALCULATE_EVALS);
    evmap(params, LEv);
    proof.proofs.setEvals(params.evals.address());
    TimerStopAndLog(STARK_CALCULATE_EVALS);
}

template <typename ElementType>
Polinomial *Starks<ElementType>::computeFRIPol(uint64_t step, StepsParams &params, CHelpersSteps *chelpersSteps)
{

    auto evalsStage = starkInfo.nStages + 2;
    auto xiChallenge = std::find_if(starkInfo.challengesMap.begin(), starkInfo.challengesMap.end(), [evalsStage](const PolMap& c) {
        return c.stageNum == evalsStage && c.stageId == 0;
    });

    uint64_t xiChallengeIndex = std::distance(starkInfo.challengesMap.begin(), xiChallenge);

    TimerStart(STARK_CALCULATE_XDIVXSUB);

    vector<uint64_t> openingPoints = starkInfo.openingPoints;

    Polinomial xi(openingPoints.size(), FIELD_EXTENSION);
    Polinomial w(openingPoints.size(), FIELD_EXTENSION);

    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

#pragma omp parallel for
    for (uint64_t i = 0; i < openingPoints.size(); ++i)
    {
        uint64_t opening = openingPoints[i] < 0 ? -openingPoints[i] : openingPoints[i];
        Goldilocks3::one((Goldilocks3::Element &)*w[i]);
        for (uint64_t j = 0; j < opening; ++j)
        {
            Polinomial::mulElement(w, i, w, i, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));
        }

        if (openingPoints[i] < 0)
        {
            Polinomial::divElement(w, i, (Goldilocks::Element &)Goldilocks::one(), w, i);
        }

        Polinomial::mulElement(xi, i, params.challenges, xiChallengeIndex, w, i);

#pragma omp parallel for
        for (uint64_t k = 0; k < (N << extendBits); k++)
        {
            Polinomial::subElement(params.xDivXSubXi, k + i * NExtended, x, k, xi, i);
        }
    }

    Polinomial::batchInverseParallel(params.xDivXSubXi, params.xDivXSubXi);

#pragma omp parallel for
    for (uint64_t i = 0; i < openingPoints.size(); ++i)
    {
        for (uint64_t k = 0; k < (N << extendBits); k++)
        {
            Polinomial::mulElement(params.xDivXSubXi, k + i * NExtended, params.xDivXSubXi, k + i * NExtended, x, k);
        }
    }
    TimerStopAndLog(STARK_CALCULATE_XDIVXSUB);

    calculateExpressions(step, params, chelpersSteps);

    Polinomial *friPol = new Polinomial(params.f_2ns, NExtended, FIELD_EXTENSION, FIELD_EXTENSION, "friPol");

    return friPol;
}

template <typename ElementType>
void Starks<ElementType>::computeFRIFolding(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t step, Polinomial &challenge)
{
    FRI<ElementType>::fold(step, fproof, friPol, challenge, starkInfo, treesFRI);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIQueries(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t *friQueries)
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
void Starks<ElementType>::transposePolsColumns(StepsParams &params, Polinomial *transPols, Hint hint, Goldilocks::Element *pBuffer)
{
    u_int64_t stride_pol_ = N * FIELD_EXTENSION + 8;

    vector<string> srcFields = getSrcFields(hint.name);
    vector<string> dstFields = getDstFields(hint.name);

    for (uint64_t i = 0; i < srcFields.size(); i++)
    {
        auto it = hint.fields.find(srcFields[i]);
        if (it == hint.fields.end())
        {
            zklog.error("Unknown src field name=" + srcFields[i]);
            exitProcess();
            exit(-1);
        }
        HintField hintField = hint.fields[srcFields[i]];
        if (hintField.operand == opType::cm || hintField.operand == opType::tmp)
        {
            uint64_t id = hintField.id;
            Polinomial p = starkInfo.getPolinomial(params.pols, id, N);
            uint64_t indx = cm2Transposed[id];
            transPols[indx].potConstruct(&(pBuffer[indx * stride_pol_]), p.degree(), p.dim(), p.dim());
            Polinomial::copy(transPols[indx], p);
        }
    }

    for (uint64_t i = 0; i < dstFields.size(); i++)
    {
        auto it = hint.fields.find(dstFields[i]);
        if (it == hint.fields.end())
        {
            zklog.error("Unknown src field name=" + dstFields[i]);
            exitProcess();
            exit(-1);
        }
        HintField hintField = hint.fields[dstFields[i]];
        if (hintField.operand == opType::cm || hintField.operand == opType::tmp)
        {
            uint64_t id = hintField.id;
            Polinomial p = starkInfo.getPolinomial(params.pols, id, N);
            uint64_t indx = cm2Transposed[id];
            transPols[indx].potConstruct(&(pBuffer[indx * stride_pol_]), p.degree(), p.dim(), p.dim());
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::transposePolsRows(StepsParams &params, Polinomial *transPols, Hint hint)
{
    vector<string> dstFields = getDstFields(hint.name);

    for (uint64_t i = 0; i < dstFields.size(); i++)
    {
        auto it = hint.fields.find(dstFields[i]);
        if (it == hint.fields.end())
        {
            zklog.error("Unknown dest field name=" + dstFields[i]);
            exitProcess();
            exit(-1);
        }
        HintField hintField = hint.fields[dstFields[i]];
        setSymbolCalculated(hintField.operand, hintField.id);
        if (hintField.operand == opType::cm)
        {
            uint64_t id = hintField.id;
            uint64_t transposedId = cm2Transposed[id];
            Polinomial cmPol = starkInfo.getPolinomial(params.pols, id, N);
            Polinomial::copy(cmPol, transPols[transposedId]);
        }
    }
}

template <typename ElementType>
bool Starks<ElementType>::isHintResolved(Hint &hint, vector<string> dstFields)
{
    for (uint64_t i = 0; i < dstFields.size(); i++)
    {
        auto it = hint.fields.find(dstFields[i]);
        if (it == hint.fields.end())
        {
            zklog.error("Unknown dest field name=" + dstFields[i]);
            exitProcess();
            exit(-1);
        }
        if (!isSymbolCalculated(hint.fields[dstFields[i]].operand, hint.fields[dstFields[i]].id))
        {
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
        auto it = hint.fields.find(srcFields[i]);
        if (it == hint.fields.end())
        {
            zklog.error("Unknown dest field name=" + srcFields[i]);
            exitProcess();
            exit(-1);
        }
        if (hint.fields[srcFields[i]].operand == opType::number)
            continue;
        if (!isSymbolCalculated(hint.fields[srcFields[i]].operand, hint.fields[srcFields[i]].id))
        {
            return false;
        }
    }

    return true;
}

template <typename ElementType>
std::vector<string> Starks<ElementType>::getSrcFields(std::string hintName)
{
    if (hintName == "public" || hintName == "subproofvalue")
    {
        return {"expression"};
    }
    else if (hintName == "gsum" || hintName == "gprod")
    {
        return {"numerator", "denominator"};
    }
    else if (hintName == "h1h2")
    {
        return {"f", "t"};
    }
    else
    {
        zklog.error("Invalid hint name=" + hintName);
        exitProcess();
        exit(-1);
    }
}

template <typename ElementType>
std::vector<string> Starks<ElementType>::getDstFields(std::string hintName)
{
    if (hintName == "public" || hintName == "subproofvalue" || hintName == "gsum" || hintName == "gprod")
    {
        return {"reference"};
    }
    else if (hintName == "h1h2")
    {
        return {"referenceH1", "referenceH2"};
    }
    else
    {
        zklog.error("Invalid hint name=" + hintName);
        exitProcess();
        exit(-1);
    }
}

template <typename ElementType>
void Starks<ElementType>::calculateHints(uint64_t step, StepsParams &params, vector<Hint> &hints)
{

    vector<Hint> hintsToCalculate;

    for (uint64_t i = 0; i < hints.size(); i++)
    {
        Hint hint = hints[i];

        vector<string> srcFields = getSrcFields(hint.name);
        vector<string> dstFields = getDstFields(hint.name);

        if (!isHintResolved(hint, dstFields) && canHintBeResolved(hint, srcFields))
        {
            hintsToCalculate.push_back(hint);
        }
    }

    if (hintsToCalculate.size() == 0)
        return;

    uint64_t sectionExtendedOffset = starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), true)];
    Goldilocks::Element *pBuffer = &params.pols[sectionExtendedOffset];

    uint64_t numPols = 0;
    for (uint64_t i = 0; i < hintsToCalculate.size(); ++i)
    {
        Hint hint = hintsToCalculate[i];
        vector<string> srcFields = getSrcFields(hint.name);
        vector<string> dstFields = getDstFields(hint.name);

        vector<string> fields(srcFields.begin(), srcFields.end());
        fields.insert(fields.end(), dstFields.begin(), dstFields.end());

        for (uint64_t i = 0; i < fields.size(); i++)
        {
            auto it = hint.fields.find(fields[i]);
            if (it == hint.fields.end())
            {
                zklog.error("Unknown field name=" + fields[i]);
                exitProcess();
                exit(-1);
            }
            HintField hintField = hint.fields[fields[i]];
            if (hintField.operand == opType::cm || hintField.operand == opType::tmp)
            {
                cm2Transposed[hintField.id] = numPols++;
            }
        }
    }

    Polinomial *transPols = new Polinomial[numPols];

    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);
    for (uint64_t i = 0; i < hintsToCalculate.size(); ++i)
    {
        transposePolsColumns(params, transPols, hintsToCalculate[i], pBuffer);
    }
    TimerStopAndLogExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);

    TimerStartExpr(STARK_CALCULATE_HINTS_STEP, step);
    uint64_t *mem_ = (uint64_t *)pAddress;
    uint64_t *pbufferH = &mem_[sectionExtendedOffset + numPols * (N * FIELD_EXTENSION + 8)];

    uint64_t maxThreads = omp_get_max_threads();
    uint64_t nThreads = hintsToCalculate.size() > maxThreads ? maxThreads : hintsToCalculate.size();

#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < hintsToCalculate.size(); i++)
    {
        Hint hint = hintsToCalculate[i];

        if (1 == 0)
        {
            // Build the Hint object
            auto hintHandler = HintHandlerBuilder::create(hint.name)->build();

            // Get the polynomials names involved
            auto srcPolsNames = hintHandler->getSources();
            auto dstPolsNames = hintHandler->getDestinations();

            vector<string> polsNames(srcPolsNames.size() + dstPolsNames.size());
            polsNames.insert(polsNames.end(), srcPolsNames.begin(), srcPolsNames.end());
            polsNames.insert(polsNames.end(), dstPolsNames.begin(), dstPolsNames.end());

            // Prepare polynomials map to be sent to the hint
            std::map<std::string, Polinomial *> polynomials;
            for (const auto &polName : polsNames)
            {
                const auto &hintField = hint.fields[polName];
                if (hintField.operand == opType::cm || hintField.operand == opType::tmp)
                {
                    polynomials[polName] = &transPols[cm2Transposed[hintField.id]];
                }
            }

            // Resolve hint
            hintHandler->resolveHint(N, hint, polynomials);
        }
        else
        {
            if (hint.name == "h1h2")
            {
                uint64_t h1Id = hint.fields["referenceH1"].id;
                uint64_t h2Id = hint.fields["referenceH2"].id;
                uint64_t fId = hint.fields["f"].id;
                uint64_t tId = hint.fields["t"].id;

                if (transPols[cm2Transposed[h1Id]].dim() == 1)
                {
                    Polinomial::calculateH1H2_opt1(transPols[cm2Transposed[h1Id]], transPols[cm2Transposed[h2Id]], transPols[cm2Transposed[fId]], transPols[cm2Transposed[tId]], i, &pbufferH[omp_get_thread_num() * sizeof(Goldilocks::Element) * N], (sizeof(Goldilocks::Element) - 3) * N);
                }
                else if (transPols[cm2Transposed[h1Id]].dim() == 3)
                {
                    Polinomial::calculateH1H2_opt3(transPols[cm2Transposed[h1Id]], transPols[cm2Transposed[h2Id]], transPols[cm2Transposed[fId]], transPols[cm2Transposed[tId]], i, &pbufferH[omp_get_thread_num() * sizeof(Goldilocks::Element) * N], (sizeof(Goldilocks::Element) - 5) * N);
                }
                else
                {
                    std::cerr << "Error: calculateH1H2_ invalid" << std::endl;
                    exit(-1);
                }
            }
            else if (hint.name == "gprod")
            {
                uint64_t zId = hint.fields["reference"].id;
                uint64_t numeratorId = hint.fields["numerator"].id;
                uint64_t denominatorId = hint.fields["denominator"].id;
                Polinomial::calculateZ(transPols[cm2Transposed[zId]], transPols[cm2Transposed[numeratorId]], transPols[cm2Transposed[denominatorId]]);
            }
            else if (hint.name == "gsum")
            {
            }
            else if (hint.name == "subproofValue")
            {
            }
            else
            {
                zklog.error("Invalid hint type=" + hint.name);
                exitProcess();
                exit(-1);
            }
        }
    }
    TimerStopAndLogExpr(STARK_CALCULATE_HINTS_STEP, step);

    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_2_STEP, step);
    for (uint64_t i = 0; i < hintsToCalculate.size(); ++i)
    {
        Hint hint = hintsToCalculate[i];
        transposePolsRows(params, transPols, hint);
    }
    TimerStopAndLogExpr(STARK_CALCULATE_TRANSPOSE_2_STEP, step);

    delete[] transPols;
}

template <typename ElementType>
void Starks<ElementType>::evmap(StepsParams &params, Polinomial &LEv)
{
    vector<uint64_t> openingPoints = starkInfo.openingPoints;
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    u_int64_t size_eval = starkInfo.evMap.size();

    Polinomial LEv_Helpers(openingPoints.size() * N, FIELD_EXTENSION);

#pragma omp parallel for
    for (uint64_t k = 0; k < N; ++k)
    {
        for (uint64_t i = 0; i < openingPoints.size(); ++i)
        {
            Goldilocks::Element *LEv_ = &LEv[i * N + k][0];
            LEv_Helpers[i * N + k][0] = LEv_[0] + LEv_[1];
            LEv_Helpers[i * N + k][1] = LEv_[0] + LEv_[2];
            LEv_Helpers[i * N + k][2] = LEv_[1] + LEv_[2];
        }
    }

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
            openingPos[kk] = findIndex(openingPoints, ev.prime);
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
                Polinomial::mulAddElement_adim3(&(evals_acc[thread_idx][i * FIELD_EXTENSION]), &(LEv[index * N + k][0]), &(LEv_Helpers[index * N + k][0]), ordPols[i], k << extendBits);
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
            (params.evals[indx[i]])[0] = evals_acc[0][offset];
            (params.evals[indx[i]])[1] = evals_acc[0][offset + 1];
            (params.evals[indx[i]])[2] = evals_acc[0][offset + 2];
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
int Starks<ElementType>::findIndex(std::vector<uint64_t> openingPoints, int prime)
{
    auto it = std::find_if(openingPoints.begin(), openingPoints.end(), [prime](int p)
                           { return p == prime; });

    if (it != openingPoints.end())
    {
        return it - openingPoints.begin();
    }
    else
    {
        return -1;
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
void Starks<ElementType>::calculateHash(ElementType* hash, Polinomial &pol) {
    TranscriptType transcriptHash(merkleTreeArity, merkleTreeCustom);
    for (uint64_t i = 0; i < pol.degree(); i++)
    {
        transcriptHash.put(pol[i], pol.dim());
    }
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
void Starks<ElementType>::addTranscript(TranscriptType &transcript, Polinomial &pol)
{
    for (uint64_t i = 0; i < pol.degree(); i++)
    {
        transcript.put(pol[i], pol.dim());
    }
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
void *Starks<ElementType>::ffi_create_steps_params(Polinomial *pChallenges, Polinomial *pSubproofValues, Polinomial *pEvals, Polinomial *pXDivXSubXi, Goldilocks::Element *pPublicInputs)
{
    StepsParams *params = new StepsParams{
        pols : mem,
        pConstPols : pConstPols,
        pConstPols2ns : pConstPols2ns,
        challenges : *pChallenges,
        subproofValues : *pSubproofValues,
        x_n : x_n,
        x_2ns : x_2ns,
        zi : zi,
        evals : *pEvals,
        xDivXSubXi : *pXDivXSubXi,
        publicInputs : pPublicInputs,
        q_2ns : &mem[starkInfo.mapOffsets[std::make_pair("q", true)]],
        f_2ns : &mem[starkInfo.mapOffsets[std::make_pair("f", true)]]
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