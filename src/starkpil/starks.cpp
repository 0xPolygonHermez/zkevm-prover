#include "definitions.hpp"
#include "starks.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

USING_PROVER_FORK_NAMESPACE;

template <typename ElementType>
void Starks<ElementType>::genProof(Goldilocks::Element *pAddress, FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs, bool debug)
{
    TimerStart(STARK_PROOF);

    // Initialize vars
    TimerStart(STARK_INITIALIZATION);

    ExpressionsAvx expressionsAvx(setupCtx);

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element* evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* subproofValues = new Goldilocks::Element[setupCtx.starkInfo.nSubProofValues * FIELD_EXTENSION];
    
    vector<bool> subProofValuesCalculated(setupCtx.starkInfo.nSubProofValues, false);
    vector<bool> commitsCalculated(setupCtx.starkInfo.cmPolsMap.size(), false);

    StepsParams params = {
        pols : pAddress,
        publicInputs : publicInputs,
        challenges : challenges,
        subproofValues : subproofValues,
        evals : evals,
        prover_initialized : true,
    };

    for (uint64_t i = 0; i < setupCtx.starkInfo.mapSectionsN["cm1"]; ++i)
    {
        commitsCalculated[i] = true;
    }

    TimerStopAndLog(STARK_INITIALIZATION);

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    TimerStart(STARK_STEP_0);
    if(!debug) {
        ElementType verkey[nFieldElements];
        treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
        addTranscript(transcript, &verkey[0], nFieldElements);
    }
    
    if(setupCtx.starkInfo.starkStruct.hashCommits) {
        ElementType hash[nFieldElements];
        calculateHash(hash, &publicInputs[0], setupCtx.starkInfo.nPublics);
        addTranscript(transcript, hash, nFieldElements);
    } else {
        addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
    }

    TimerStopAndLog(STARK_STEP_0);

    bool validConstraints = true;

    for (uint64_t step = 1; step <= setupCtx.starkInfo.nStages; step++)
    {
        TimerStartExpr(STARK_STEP, step);
        for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
        {
            if(setupCtx.starkInfo.challengesMap[i].stage == step) {
                getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
            }
        }

        computeStageExpressions(step, expressionsAvx, params, proof, commitsCalculated, subProofValuesCalculated);

        calculateImPolsExpressions(step, pAddress, publicInputs, challenges, subproofValues, evals);

        for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
            if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
               commitsCalculated[i] = true;
            }
        }

        if(step == setupCtx.starkInfo.nStages) {
            for(uint64_t i = 0; i < setupCtx.starkInfo.nSubProofValues; i++) {
                if(!subProofValuesCalculated[i]) {
                    zklog.info("Subproofvalue " + to_string(i) + " is not calculated");
                    exitProcess();
                    exit(-1);
                }
            }
        }

        if(step <= setupCtx.starkInfo.nStages) {
            for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
                if(setupCtx.starkInfo.cmPolsMap[i].stage == step && !commitsCalculated[i]) {
                    zklog.info("Witness polynomial " + setupCtx.starkInfo.cmPolsMap[i].name + " is not calculated");
                    exitProcess();
                    exit(-1);
                }
            }
        }

        if(!debug) {
            commitStage(step, pAddress, proof);
        }

        if (debug)
        {
            Goldilocks::Element randomValues[4] = {Goldilocks::fromU64(0), Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(3)};
            addTranscriptGL(transcript, randomValues, 4);
        }
        else
        {
            addTranscript(transcript, &proof.proof.roots[step - 1][0], nFieldElements);
        }

        TimerStopAndLogExpr(STARK_STEP, step);
    }

    proof.proof.setSubproofValues(subproofValues);
    
    if (debug) {
        if(validConstraints) {
            TimerLog(ALL_CONSTRAINTS_ARE_VALID);
        } else {
            TimerLog(INVALID_CONSTRAINTS);
        }
        return;
    }

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    
    calculateQuotientPolynomial(pAddress, publicInputs, challenges, subproofValues, evals);

    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            commitsCalculated[i] = true;
        }
    }
    commitStage(setupCtx.starkInfo.nStages + 1, pAddress, proof);

    if (debug)
    {
        Goldilocks::Element randomValues[4] = {Goldilocks::fromU64(0), Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(3)};
        addTranscriptGL(transcript, randomValues, 4);
    }
    else
    {
        addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], nFieldElements);
    }
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    computeEvals(pAddress, challenges, evals, proof);

    if(setupCtx.starkInfo.starkStruct.hashCommits) {
        ElementType hash[nFieldElements];
        calculateHash(hash, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        addTranscript(transcript, hash, nFieldElements);
    } else {
        addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    }    

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    prepareFRIPolynomial(pAddress, challenges);
    calculateFRIPolynomial(pAddress, publicInputs, challenges, subproofValues, evals);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    Goldilocks::Element *friPol = &pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
    
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        computeFRIFolding(step, pAddress, challenge, proof);
        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            addTranscript(transcript, &proof.proof.fri.trees[step + 1].root[0], nFieldElements);
        }
        else
        {
            if(setupCtx.starkInfo.starkStruct.hashCommits) {
                ElementType hash[nFieldElements];
                calculateHash(hash, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                addTranscript(transcript, hash, nFieldElements);
            } else {
                addTranscriptGL(transcript, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            }
        }
        getChallenge(transcript, *challenge);
    }

    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];

    TranscriptType transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    computeFRIQueries(proof, friQueries);

    TimerStopAndLog(STARK_STEP_FRI);

    delete challenges;
    delete evals;
    delete subproofValues;
        
    TimerStopAndLog(STARK_PROOF);
}

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelize(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof)
{    
    TimerStartExpr(STARK_LDE_AND_MERKLETREE_STEP, step);
    TimerStartExpr(STARK_LDE_STEP, step);

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    std::string section = "cm" + to_string(step);  
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)];
    
    Goldilocks::Element *pBuff = &buffer[setupCtx.starkInfo.mapOffsets[make_pair(section, false)]];
    Goldilocks::Element *pBuffExtended = &buffer[setupCtx.starkInfo.mapOffsets[make_pair(section, true)]];

    std::pair<uint64_t, uint64_t> nttOffsetHelper = setupCtx.starkInfo.mapNTTOffsetsHelpers[section];
    Goldilocks::Element *pBuffHelper = &buffer[nttOffsetHelper.first];

    uint64_t buffHelperElements = NExtended * nCols;

    uint64_t nBlocks = 1;
    while((nttOffsetHelper.second * nBlocks < buffHelperElements + 8) ||  (nCols > 256*nBlocks) ) {
        nBlocks++;
    }

    NTT_Goldilocks ntt(N);
    ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_LDE_STEP, step);
    TimerStartExpr(STARK_MERKLETREE_STEP, step);
    treesGL[step - 1]->setSource(pBuffExtended);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proof.roots[step - 1][0]);
    TimerStopAndLogExpr(STARK_MERKLETREE_STEP, step);
    TimerStopAndLogExpr(STARK_LDE_AND_MERKLETREE_STEP, step);
}

template <typename ElementType>
void Starks<ElementType>::commitStage(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof)
{  

    if (step <= setupCtx.starkInfo.nStages)
    {
        extendAndMerkelize(step, buffer, proof);
    }
    else
    {
        computeQ(step, buffer, proof);
    }
}

template <typename ElementType>
void Starks<ElementType>::computeQ(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    std::string section = "cm" + to_string(setupCtx.starkInfo.nStages + 1);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(setupCtx.starkInfo.nStages + 1)];
    Goldilocks::Element *cmQ = &buffer[setupCtx.starkInfo.mapOffsets[make_pair(section, true)]];

    std::pair<uint64_t, uint64_t> nttOffsetHelper = setupCtx.starkInfo.mapNTTOffsetsHelpers[section];
    Goldilocks::Element *pBuffHelper = &buffer[nttOffsetHelper.first];

    uint64_t buffHelperElements = NExtended * nCols;
    
    uint64_t nBlocks = 1;
    while((nttOffsetHelper.second * nBlocks < buffHelperElements) || (nCols > 256*nBlocks) ) {
        nBlocks++;
    }

    NTT_Goldilocks nttExtended(NExtended);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);
    nttExtended.INTT(&buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], NExtended, setupCtx.starkInfo.qDim, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_INTT_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);

    for (uint64_t p = 0; p < setupCtx.starkInfo.qDeg; p++)
    {   
        #pragma omp parallel for
        for(uint64_t i = 0; i < N; i++)
        { 
            Goldilocks3::mul((Goldilocks3::Element &)cmQ[(i * setupCtx.starkInfo.qDeg + p) * FIELD_EXTENSION], (Goldilocks3::Element &)buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)] + (p * N + i) * FIELD_EXTENSION], setupCtx.constPols.S[p]);
        }
    }

    memset(&cmQ[N * setupCtx.starkInfo.qDeg * setupCtx.starkInfo.qDim], 0, (NExtended - N) * setupCtx.starkInfo.qDeg * setupCtx.starkInfo.qDim * sizeof(Goldilocks::Element));

    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_MUL_STEP, step);

    TimerStartExpr(STARK_CALCULATE_EXPS_2NS_NTT_STEP, step);
    nttExtended.NTT(cmQ, cmQ, NExtended, nCols, pBuffHelper, 3, nBlocks);
    TimerStopAndLogExpr(STARK_CALCULATE_EXPS_2NS_NTT_STEP, step);

    TimerStartExpr(STARK_MERKLETREE_STEP, step);
    treesGL[step - 1]->setSource(&buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), true)]]);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proof.roots[step - 1][0]);

    TimerStopAndLogExpr(STARK_MERKLETREE_STEP, step);
}

template <typename ElementType>
void Starks<ElementType>::computeEvals(Goldilocks::Element *buffer, Goldilocks::Element *challenges, Goldilocks::Element *evals, FRIProof<ElementType> &proof)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    auto evalsStage = setupCtx.starkInfo.nStages + 2;
    auto xiChallenge = std::find_if(setupCtx.starkInfo.challengesMap.begin(), setupCtx.starkInfo.challengesMap.end(), [evalsStage](const PolMap& c) {
        return c.stage == evalsStage && c.stageId == 0;
    });

    uint64_t xiChallengeIndex = std::distance(setupCtx.starkInfo.challengesMap.begin(), xiChallenge);

    TimerStart(STARK_CALCULATE_LEv);
    
    Goldilocks::Element* LEv = &buffer[setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)]];
    
    Goldilocks::Element xis[setupCtx.starkInfo.openingPoints.size() * FIELD_EXTENSION];
    Goldilocks::Element xisShifted[setupCtx.starkInfo.openingPoints.size() * FIELD_EXTENSION];

    Goldilocks::Element shift_inv = Goldilocks::inv(Goldilocks::shift());
    for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
    {
        Goldilocks::Element w = Goldilocks::one();
        uint64_t openingAbs = setupCtx.starkInfo.openingPoints[i] < 0 ? -setupCtx.starkInfo.openingPoints[i] : setupCtx.starkInfo.openingPoints[i];
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w = w * Goldilocks::w(setupCtx.starkInfo.starkStruct.nBits);
        }

        if (setupCtx.starkInfo.openingPoints[i] < 0)
        {
            w = Goldilocks::inv(w);
        }

        Goldilocks3::mul((Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(challenges[xiChallengeIndex * FIELD_EXTENSION]), w);
        Goldilocks3::mul((Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), shift_inv);

        Goldilocks3::one((Goldilocks3::Element &)LEv[i * FIELD_EXTENSION]);
    }


#pragma omp parallel for
    for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
    {
        for (uint64_t k = 1; k < N; k++)
        {
            Goldilocks3::mul((Goldilocks3::Element &)(LEv[(k*setupCtx.starkInfo.openingPoints.size() + i)*FIELD_EXTENSION]), (Goldilocks3::Element &)(LEv[((k-1)*setupCtx.starkInfo.openingPoints.size() + i)*FIELD_EXTENSION]), (Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]));
        }
    }

    std::pair<uint64_t, uint64_t> nttOffsetHelper = setupCtx.starkInfo.mapNTTOffsetsHelpers["LEv"];
    Goldilocks::Element *pBuffHelper = &buffer[nttOffsetHelper.first];
    
    NTT_Goldilocks ntt(N);
    ntt.INTT(&LEv[0], &LEv[0], N, FIELD_EXTENSION * setupCtx.starkInfo.openingPoints.size(), pBuffHelper);

    TimerStopAndLog(STARK_CALCULATE_LEv);

    TimerStart(STARK_CALCULATE_EVALS);
    evmap(buffer, evals, LEv);
    proof.proof.setEvals(evals);
    TimerStopAndLog(STARK_CALCULATE_EVALS);
}

template <typename ElementType>
void Starks<ElementType>::prepareFRIPolynomial(Goldilocks::Element *buffer, Goldilocks::Element *challenges)
{
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    auto evalsStage = setupCtx.starkInfo.nStages + 2;
    auto xiChallenge = std::find_if(setupCtx.starkInfo.challengesMap.begin(), setupCtx.starkInfo.challengesMap.end(), [evalsStage](const PolMap& c) {
        return c.stage == evalsStage && c.stageId == 0;
    });

    uint64_t xiChallengeIndex = std::distance(setupCtx.starkInfo.challengesMap.begin(), xiChallenge);

    Goldilocks::Element xis[setupCtx.starkInfo.openingPoints.size() * FIELD_EXTENSION];
    for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
    {
        Goldilocks::Element w = Goldilocks::one();
        uint64_t openingAbs = setupCtx.starkInfo.openingPoints[i] < 0 ? -setupCtx.starkInfo.openingPoints[i] : setupCtx.starkInfo.openingPoints[i];
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w = w * Goldilocks::w(setupCtx.starkInfo.starkStruct.nBits);
        }

        if (setupCtx.starkInfo.openingPoints[i] < 0)
        {
            w = Goldilocks::inv(w);
        }

        Goldilocks3::mul((Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(challenges[xiChallengeIndex * FIELD_EXTENSION]), w);
    }

    TimerStart(STARK_CALCULATE_XDIVXSUB);

    for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
    {
#pragma omp parallel for
        for (uint64_t k = 0; k < NExtended; k++)
        {
            Goldilocks3::sub((Goldilocks3::Element &)(buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)]  + (k + i * NExtended) * FIELD_EXTENSION]), setupCtx.constPols.x[k], (Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]));
        }
    }

    Polinomial xDivXSubXi_(&buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)]], NExtended * setupCtx.starkInfo.openingPoints.size(), FIELD_EXTENSION, FIELD_EXTENSION);
    Polinomial::batchInverseParallel(xDivXSubXi_, xDivXSubXi_);

    for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
    {
#pragma omp parallel for
        for (uint64_t k = 0; k < NExtended; k++)
        {
            Goldilocks3::mul((Goldilocks3::Element &)(buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)] + (k + i * NExtended) * FIELD_EXTENSION]), (Goldilocks3::Element &)(buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)] + (k + i * NExtended) * FIELD_EXTENSION]), setupCtx.constPols.x[k]);
        }
    }
    TimerStopAndLog(STARK_CALCULATE_XDIVXSUB);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIFolding(uint64_t step, Goldilocks::Element *buffer, Goldilocks::Element *challenge, FRIProof<ElementType> &fproof)
{
    Goldilocks::Element* pol = &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
    FRI<ElementType>::fold(step, fproof, pol, challenge, setupCtx.starkInfo, treesFRI);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIQueries(FRIProof<ElementType> &fproof, uint64_t *friQueries)
{
    FRI<ElementType>::proveQueries(friQueries, fproof, treesGL, treesFRI, setupCtx.starkInfo);
}


template <typename ElementType>
void Starks<ElementType>::evmap(Goldilocks::Element *buffer, Goldilocks::Element *evals, Goldilocks::Element *LEv)
{
    uint64_t extendBits = setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits;
    u_int64_t size_eval = setupCtx.starkInfo.evMap.size();

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

    int num_threads = omp_get_max_threads();
    int size_thread = size_eval * FIELD_EXTENSION;
    Goldilocks::Element *evals_acc = &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("evals", true)]];
    memset(&evals_acc[0], 0, num_threads * size_thread * sizeof(Goldilocks::Element));
    
    Polinomial *ordPols = new Polinomial[size_eval];

    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = setupCtx.starkInfo.evMap[i];
        bool committed = ev.type == EvMap::eType::cm ? true : false;
        Goldilocks::Element *pols = committed ? buffer : setupCtx.constPols.pConstPolsAddressExtended;
        setupCtx.starkInfo.getPolynomial(ordPols[i], pols, committed, ev.id, true);
    }

#pragma omp parallel
    {
        int thread_idx = omp_get_thread_num();
        Goldilocks::Element *evals_acc_thread = &evals_acc[thread_idx * size_thread];
#pragma omp for
        for (uint64_t k = 0; k < N; k++)
        {
            Goldilocks3::Element LEv_[setupCtx.starkInfo.openingPoints.size()];
            for(uint64_t o = 0; o < setupCtx.starkInfo.openingPoints.size(); o++) {
                uint64_t pos = (o + k*setupCtx.starkInfo.openingPoints.size()) * FIELD_EXTENSION;
                LEv_[o][0] = LEv[pos];
                LEv_[o][1] = LEv[pos + 1];
                LEv_[o][2] = LEv[pos + 2];
            }
            uint64_t row = (k << extendBits);
            for (uint64_t i = 0; i < size_eval; i++)
            {
                EvMap ev = setupCtx.starkInfo.evMap[i];
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
            std::memcpy((Goldilocks3::Element &)(evals[i * FIELD_EXTENSION]), sum, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
    delete[] ordPols;
}

template <typename ElementType>
void Starks<ElementType>::getChallenge(TranscriptType &transcript, Goldilocks::Element &challenge)
{
    transcript.getField((uint64_t *)&challenge);
}

template <typename ElementType>
void Starks<ElementType>::calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements) {
    TranscriptType transcriptHash(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
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
void Starks<ElementType>::ffi_extend_and_merkelize(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> *proof)
{
    extendAndMerkelize(step, buffer, *proof);
}

template <typename ElementType>
void Starks<ElementType>::ffi_treesGL_get_root(uint64_t index, ElementType *dst)
{
    treesGL[index]->getRoot(dst);
}

template <typename ElementType>
void Starks<ElementType>::calculateImPolsExpressions(uint64_t step, Goldilocks::Element *buffer, Goldilocks::Element *publicInputs, Goldilocks::Element *challenges, Goldilocks::Element *subproofValues, Goldilocks::Element *evals) {
    TimerStart(STARK_CALCULATE_IMPOLS_EXPS);

    ExpressionsAvx expressionsAvx(setupCtx);

    StepsParams params {
        pols : buffer,
        publicInputs,
        challenges,
        subproofValues,
        evals,
        prover_initialized: true,
    };

    expressionsAvx.calculateExpressions(params, nullptr, setupCtx.expressionsBin.expressionsBinArgsImPols, setupCtx.expressionsBin.imPolsInfo[step - 1], false, false, true);

    // uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    // Goldilocks::Element* pAddr = &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]];
    // for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
    //     if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
    //         expressionsAvx.calculateExpression(params, pAddr, setupCtx.starkInfo.cmPolsMap[i].expId);
    //         Goldilocks::Element* imAddr = &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos];
    //     #pragma omp parallel
    //         for(uint64_t j = 0; j < N; ++j) {
    //             std::memcpy(&imAddr[j*setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)]], &pAddr[j*setupCtx.starkInfo.cmPolsMap[i].dim], setupCtx.starkInfo.cmPolsMap[i].dim * sizeof(Goldilocks::Element));
    //         }
    //     }
    // }
    
    TimerStopAndLog(STARK_CALCULATE_IMPOLS_EXPS);
}

template <typename ElementType>
void Starks<ElementType>::calculateQuotientPolynomial(Goldilocks::Element *buffer, Goldilocks::Element *publicInputs, Goldilocks::Element *challenges, Goldilocks::Element *subproofValues, Goldilocks::Element *evals) {
    TimerStart(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);
    ExpressionsAvx expressionsAvx(setupCtx);
    StepsParams params {
        pols : buffer,
        publicInputs,
        challenges,
        subproofValues,
        evals,
        prover_initialized: true,
    };
    expressionsAvx.calculateExpression(params, &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], setupCtx.starkInfo.cExpId);
    TimerStopAndLog(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);
}

template <typename ElementType>
void Starks<ElementType>::calculateFRIPolynomial(Goldilocks::Element *buffer, Goldilocks::Element *publicInputs, Goldilocks::Element *challenges, Goldilocks::Element *subproofValues, Goldilocks::Element *evals) {
    TimerStart(STARK_CALCULATE_FRI_POLYNOMIAL);
    ExpressionsAvx expressionsAvx(setupCtx);
    StepsParams params {
        pols : buffer,
        publicInputs,
        challenges,
        subproofValues,
        evals,
        prover_initialized: true,
    };
    expressionsAvx.calculateExpression(params, &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]], setupCtx.starkInfo.friExpId);
    TimerStopAndLog(STARK_CALCULATE_FRI_POLYNOMIAL);
}

// ALL THIS FUNCTIONS WILL BE REMOVED WHEN CONVERTED TO LIBRARY

template <typename ElementType>
void Starks<ElementType>::computeStageExpressions(uint64_t step, ExpressionsCtx& expressionsCtx, StepsParams &params, FRIProof<ElementType> &proof, vector<bool> &commitsCalculated, vector<bool> &subProofValuesCalculated)
{
    TimerStartExpr(STARK_TRY_CALCULATE_EXPS_STEP, step);
    uint64_t symbolsToBeCalculated = isStageCalculated(step, params, commitsCalculated, subProofValuesCalculated);
    while (symbolsToBeCalculated > 0)
    {
        calculateHints(step, expressionsCtx, params, commitsCalculated, subProofValuesCalculated);
        uint64_t newSymbolsToBeCalculated = isStageCalculated(step, params, commitsCalculated, subProofValuesCalculated);
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
bool Starks<ElementType>::canExpressionBeCalculated(ParserParams &parserParams, StepsParams &params, vector<bool> &commitsCalculated, vector<bool> &subProofValuesCalculated) {
    for(uint64_t i = 0; i < parserParams.nCmPolsUsed; i++) {
        uint64_t cmPolUsedId = setupCtx.expressionsBin.expressionsBinArgsExpressions.cmPolsIds[parserParams.cmPolsOffset + i];
        if (!isSymbolCalculated(opType::cm, cmPolUsedId, params, commitsCalculated, subProofValuesCalculated)) {
            return false;
        }
    }

    for(uint64_t i = 0; i < parserParams.nSubproofValuesUsed; i++) {
        uint64_t subproofValueUsedId = setupCtx.expressionsBin.expressionsBinArgsExpressions.subproofValuesIds[parserParams.subproofValuesOffset + i];
        if (!isSymbolCalculated(opType::subproofvalue, subproofValueUsedId, params, commitsCalculated, subProofValuesCalculated)) {
            return false;
        }
    }
    return true;
}

template <typename ElementType>
bool Starks<ElementType>::isHintResolved(Hint &hint, vector<string> dstFields, StepsParams &params, vector<bool> &commitsCalculated, vector<bool> &subProofValuesCalculated)
{
    for (uint64_t i = 0; i < dstFields.size(); i++)
    {
        auto dstField = dstFields[i];
        auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [dstField](const HintField& hintField) {
            return hintField.name == dstField;
        });

        if(hintField == hint.fields.end()) {
            zklog.error("Hint field " + dstField + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        if (!isSymbolCalculated(hintField->operand, hintField->id, params, commitsCalculated, subProofValuesCalculated)) {
            return false;
        }
    }

    return true;
}

template <typename ElementType>
bool Starks<ElementType>::canHintBeResolved(Hint &hint, vector<string> srcFields, StepsParams &params, vector<bool> &commitsCalculated, vector<bool> &subProofValuesCalculated)
{
    for (uint64_t i = 0; i < srcFields.size(); i++)
    {
        auto srcField = srcFields[i];
        auto hintField= std::find_if(hint.fields.begin(), hint.fields.end(), [srcField](const HintField& hintField) {
            return hintField.name == srcField;
        });

        if(hintField == hint.fields.end()) {
            zklog.error("Hint field " + srcField + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        if (hintField->operand == opType::number) continue;
        if (!isSymbolCalculated(hintField->operand, hintField->id, params, commitsCalculated, subProofValuesCalculated)) {
            return false;
        }
    }

    return true;
}

template <typename ElementType>
void Starks<ElementType>::calculateHints(uint64_t step, ExpressionsCtx& expressionsCtx, StepsParams &params, vector<bool> &commitsCalculated, vector<bool> &subProofValuesCalculated)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

    Polinomial* polynomials = new Polinomial[setupCtx.starkInfo.cmPolsMap.size()];

    Polinomial* polynomialsExps = new Polinomial[setupCtx.starkInfo.friExpId + 1];

    vector<bool> srcPolsExpsNames(setupCtx.starkInfo.friExpId + 1, false);

    vector<uint64_t> srcPolsNames;
    vector<uint64_t> dstPolsNames;    

    vector<uint64_t> hintsToCalculate;
    
    TimerStartExpr(STARK_PREPARE_HINTS_STEP, step);
    for (uint64_t i = 0; i < setupCtx.expressionsBin.hints.size(); i++)
    {
        Hint hint = setupCtx.expressionsBin.hints[i];
        auto hintHandler = HintHandlerBuilder::create(hint.name)->build();
        vector<string> srcFields = hintHandler->getSources();
        vector<string> dstFields = hintHandler->getDestinations();
        if (!isHintResolved(hint, dstFields, params, commitsCalculated, subProofValuesCalculated) && canHintBeResolved(hint, srcFields, params, commitsCalculated, subProofValuesCalculated))
        {
            hintsToCalculate.push_back(i);

            for (uint64_t i = 0; i < srcFields.size(); i++)
            {
                auto srcField = srcFields[i];
                auto hintField= std::find_if(hint.fields.begin(), hint.fields.end(), [srcField](const HintField& hintField) {
                    return hintField.name == srcField;
                });

                if(hintField->operand != opType::tmp && hintField->operand != opType::cm) continue;
                if (hintField->operand != opType::tmp) {
                    PolMap polInfo = setupCtx.starkInfo.cmPolsMap[hintField->id];
                    srcPolsNames.push_back(hintField->id);
                    polynomials[hintField->id].potConstruct(N, polInfo.dim);
                } else {
                    srcPolsExpsNames[hintField->id] = true;
                    polynomialsExps[hintField->id].potConstruct(N, hintField->dim);                    
                }
            }

            for (uint64_t i = 0; i < dstFields.size(); i++)
            {
                auto dstField = dstFields[i];
                auto hintField= std::find_if(hint.fields.begin(), hint.fields.end(), [dstField](const HintField& hintField) {
                    return hintField.name == dstField;
                });
                if(hintField->operand == opType::tmp) exitProcess();
                if(hintField->operand != opType::cm) continue;
                PolMap polInfo = setupCtx.starkInfo.cmPolsMap[hintField->id];
                dstPolsNames.push_back(hintField->id);
                polynomials[hintField->id].potConstruct(N, polInfo.dim);
            }
        }
    }
    TimerStopAndLogExpr(STARK_PREPARE_HINTS_STEP, step);

    if (hintsToCalculate.size() == 0)
        return;

    
    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);
    Polinomial *srcTransposedPols = new Polinomial[srcPolsNames.size()];
#pragma omp parallel for
    for(uint64_t i = 0; i < srcPolsNames.size(); i++) {
        setupCtx.starkInfo.getPolynomial(polynomials[srcPolsNames[i]], params.pols, true, srcPolsNames[i], false);
    }
    delete[] srcTransposedPols;
    TimerStopAndLogExpr(STARK_CALCULATE_TRANSPOSE_STEP, step);

    TimerStart(STARK_CALCULATE_EXPRESSIONS);
    
    for(uint64_t i = 0; i < srcPolsExpsNames.size(); i++) {
        if(srcPolsExpsNames[i]) {
            expressionsCtx.calculateExpression(params, polynomialsExps[i].address(), i);
        }    
    }

    TimerStopAndLog(STARK_CALCULATE_EXPRESSIONS);

    TimerStartExpr(STARK_CALCULATE_HINTS_STEP, step);
    uint64_t maxThreads = omp_get_max_threads();
    uint64_t nThreads = hintsToCalculate.size() > maxThreads ? maxThreads : hintsToCalculate.size();

#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < hintsToCalculate.size(); i++)
    {
        Hint hint = setupCtx.expressionsBin.hints[hintsToCalculate[i]];
        
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
            auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [polName](const HintField& hintField) {
                return hintField.name == polName;
            });
            if (hintField->operand == opType::cm) {
                polynomialsHint[polName] = &polynomials[hintField->id];
            } else if(hintField->operand == opType::tmp) {
                polynomialsHint[polName] = &polynomialsExps[hintField->id];
            }
        }

        if(hint.name == "gsum") {
            auto reference = std::find_if(hint.fields.begin(), hint.fields.end(), [](const HintField& hintField) {
                return hintField.name == "reference";
            });

            auto numerator = std::find_if(hint.fields.begin(), hint.fields.end(), [](const HintField& hintField) {
                return hintField.name == "numerator";
            });

            auto denominator = std::find_if(hint.fields.begin(), hint.fields.end(), [](const HintField& hintField) {
                return hintField.name == "denominator";
            });

            auto result = std::find_if(hint.fields.begin(), hint.fields.end(), [](const HintField& hintField) {
                return hintField.name == "result";
            });

            calculateS(polynomials[reference->id], polynomialsExps[denominator->id], Goldilocks::fromU64(numerator->value));
                
            params.subproofValues[result->id * FIELD_EXTENSION] = polynomials[reference->id][N - 1][0];
            params.subproofValues[result->id * FIELD_EXTENSION + 1] = polynomials[reference->id][N - 1][1];
            params.subproofValues[result->id * FIELD_EXTENSION + 2] = polynomials[reference->id][N - 1][2];
        } else {
            hintHandler->resolveHint(N, params, hint, polynomialsHint);
        }
    }

    TimerStopAndLogExpr(STARK_CALCULATE_HINTS_STEP, step);

    TimerStartExpr(STARK_CALCULATE_TRANSPOSE_2_STEP, step);
    Polinomial *dstTransposedPols = new Polinomial[dstPolsNames.size()];
    for(uint64_t i = 0; i < dstPolsNames.size(); i++) {
        setupCtx.starkInfo.getPolynomial(dstTransposedPols[i], params.pols, true, dstPolsNames[i], false);
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
        Hint hint = setupCtx.expressionsBin.hints[hintsToCalculate[i]];

        // Build the Hint object
        auto hintHandler = HintHandlerBuilder::create(hint.name)->build();
        auto dstFields = hintHandler->getDestinations();

        for (uint64_t i = 0; i < dstFields.size(); i++)
        {
            auto dstField = dstFields[i];
            auto hintField= std::find_if(hint.fields.begin(), hint.fields.end(), [dstField](const HintField& hintField) {
                return hintField.name == dstField;
            });
            if(hintField->operand == opType::cm) {
                commitsCalculated[hintField->id] = true;
            } else if(setupCtx.starkInfo.nSubProofValues > 0 && hintField->operand == opType::subproofvalue) {
                subProofValuesCalculated[hintField->id] = true;
            }
        }
    }
}

template <typename ElementType>
uint64_t Starks<ElementType>::isStageCalculated(uint64_t step, StepsParams &params, vector<bool> &commitsCalculated, vector<bool> &subProofValuesCalculated) {

    uint64_t symbolsToBeCalculated = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].stage != step || setupCtx.starkInfo.cmPolsMap[i].imPol) continue;
        if(!isSymbolCalculated(opType::cm, i, params, commitsCalculated, subProofValuesCalculated)) symbolsToBeCalculated++;
    }

    if(step == setupCtx.starkInfo.nStages) {
        for(uint64_t i = 0; i < setupCtx.starkInfo.nSubProofValues; i++) {
            if(!isSymbolCalculated(opType::subproofvalue, i, params, commitsCalculated, subProofValuesCalculated)) symbolsToBeCalculated++;
        }
    }

    return symbolsToBeCalculated;
}

template <typename ElementType>
bool Starks<ElementType>::isSymbolCalculated(opType operand, uint64_t id, StepsParams &params, vector<bool> &commitsCalculated, vector<bool> &subProofValuesCalculated)
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
void Starks<ElementType>::merkelizeMemory(Goldilocks::Element *pAddress)
{
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t polsSize = setupCtx.starkInfo.mapTotalN + setupCtx.starkInfo.mapSectionsN["cm3"] * NExtended;
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