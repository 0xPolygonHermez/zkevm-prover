#include "definitions.hpp"
#include "starks.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"


USING_PROVER_FORK_NAMESPACE;

template <typename ElementType>
void Starks<ElementType>::genProof(FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs, CHelpersSteps* chelpersSteps)
{
    TimerStart(STARK_PROOF);
    
    // Initialize vars
    TimerStart(STARK_INITIALIZATION);

    TranscriptType transcript;

    Polinomial evals(starkInfo.evMap.size(), FIELD_EXTENSION);
    Polinomial challenges(starkInfo.nChallenges, FIELD_EXTENSION);

    Polinomial xDivXSubXi(starkInfo.openingPoints.size() * NExtended, FIELD_EXTENSION);

    ElementType verkey[hashSize];
    treesGL[starkInfo.nStages + 1]->getRoot(verkey);

    StepsParams params = {
        pols : mem,
        pConstPols : pConstPols,
        pConstPols2ns : pConstPols2ns,
        challenges : challenges,
        x_n : x_n,
        x_2ns : x_2ns,
        zi : zi,
        evals : evals,
        xDivXSubXi : xDivXSubXi,
        publicInputs : publicInputs,
        q_2ns : &mem[starkInfo.mapOffsets.section[eSection::q_2ns]],
        f_2ns : &mem[starkInfo.mapOffsets.section[eSection::f_2ns]]
    };
    
    TimerStopAndLog(STARK_INITIALIZATION);

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    addTranscript(transcript, &verkey[0], hashSize);
    addTranscriptPublics(transcript, &publicInputs[0], starkInfo.nPublics);

    for(uint64_t step = 1; step <= starkInfo.nStages + 1; step++) {
        TimerStartStep(STARK, step);
        computeStage(step, params, proof, transcript, chelpersSteps);
        TimerStopAndLogStep(STARK, step);
    }
   
    TimerStartStep(STARK, starkInfo.nStages + 2);

    getChallenge(transcript, *params.challenges[7]);

    computeEvals(params, proof);

    addTranscript(transcript, evals);

    getChallenge(transcript, *params.challenges[5]);
    getChallenge(transcript, *params.challenges[6]);


    Polinomial* friPol = computeFRIPol(starkInfo.nStages + 2, params, chelpersSteps);

    TimerStopAndLogStep(STARK, starkInfo.nStages + 2);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);
    
    for (uint64_t step = 0; step < starkInfo.starkStruct.steps.size(); step++) {
        Polinomial challenge(1, FIELD_EXTENSION);
        getChallenge(transcript, *challenge[0]);
        computeFRIFolding(proof, friPol[0], step, challenge);
        if(step < starkInfo.starkStruct.steps.size() - 1) {
            addTranscript(transcript, &proof.proofs.fri.trees[step + 1].root[0], hashSize);
        } else {
            addTranscript(transcript, *friPol);
        }
    }

    uint64_t friQueries[starkInfo.starkStruct.nQueries];
    transcript.getPermutations(friQueries, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.steps[0].nBits);

    computeFRIQueries(proof, *friPol, friQueries);

    delete friPol;

    TimerStopAndLog(STARK_STEP_FRI);

    TimerStopAndLog(STARK_PROOF);

}

template <typename ElementType>
void Starks<ElementType>::calculateExpressions(uint64_t step, bool after, StepsParams &params, CHelpersSteps *chelpersSteps) {
    TimerStartStep(STARK_CALCULATE_EXPS, step);
    std::string stepName = "step" + to_string(step);
    if(after) stepName += "_after";
    if(chelpers.stagesInfo[stepName].nOps > 0) {
            chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgs, chelpers.stagesInfo[stepName], USE_GENERIC_PARSER);
    }
    TimerStopAndLogStep(STARK_CALCULATE_EXPS, step);
}

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelize(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof) {
    TimerStartStep(STARK_LDE_AND_MERKLETREE, step);
    TimerStartStep(STARK_LDE, step);
    
    std::string section = "cm" + to_string(step) + "_n";
    std::string sectionExtended = "cm" + to_string(step) + "_2ns";

    std::string nttBufferHelperSectionStart = step < starkInfo.nStages || !optimizeMemoryNTT
        ? "cm" + to_string(step + 1) + "_2ns"
        : "cm1_n";

    uint64_t nCols = starkInfo.mapSectionsN.section[string2section(section)];

    Goldilocks::Element* pBuff = &params.pols[starkInfo.mapOffsets.section[string2section(section)]];
    Goldilocks::Element* pBuffExtended = &params.pols[starkInfo.mapOffsets.section[string2section(sectionExtended)]];
    Goldilocks::Element* pBuffHelper = &params.pols[starkInfo.mapOffsets.section[string2section(nttBufferHelperSectionStart)]];
      
    ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols, pBuffHelper);
    TimerStopAndLogStep(STARK_LDE, step);
    TimerStartStep(STARK_MERKLETREE, step);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proofs.roots[step - 1][0]);
    TimerStopAndLogStep(STARK_MERKLETREE, step);
    TimerStopAndLogStep(STARK_LDE_AND_MERKLETREE, step);
}

template <typename ElementType>
void Starks<ElementType>::computeStage(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof,TranscriptType &transcript, CHelpersSteps *chelpersSteps) {
    uint64_t challengeIndex = 0;
    for(uint64_t i = 0; i < step - 1; i++) {
        challengeIndex += starkInfo.numChallenges[i];
    }

    uint64_t nChallenges = step <= starkInfo.nStages ? starkInfo.numChallenges[step - 1] : 1;

    for(uint64_t i = 0; i < nChallenges; i++) {
        getChallenge(transcript, *params.challenges[challengeIndex + i]);
    }

    calculateExpressions(step, false, params, chelpersSteps);

    calculateHints(step, params);

    if(step == starkInfo.nStages) {
        calculateExpressions(step, true, params, chelpersSteps);
    }

    if(step <= starkInfo.nStages) {
        extendAndMerkelize(step, params, proof);
    } else {
        computeQ(step, params, proof);  
    }

    addTranscript(transcript, &proof.proofs.roots[step - 1][0], hashSize);
}

template <typename ElementType>
void Starks<ElementType>::computeQ(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof) {
    TimerStartStep(STARK_CALCULATE_EXPS_2NS_INTT, step);
    Polinomial qq1 = Polinomial(NExtended, starkInfo.qDim, "qq1");
    Polinomial qq2 = Polinomial(NExtended * starkInfo.qDeg, starkInfo.qDim, "qq2");
    nttExtended.INTT(qq1.address(), &params.pols[starkInfo.mapOffsets.section[eSection::q_2ns]], NExtended, starkInfo.qDim, NULL, 2, 1);
    TimerStopAndLogStep(STARK_CALCULATE_EXPS_2NS_INTT, step);

    TimerStartStep(STARK_CALCULATE_EXPS_2NS_MUL, step);
    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);

    std::string sectionStageQ = "cm" + to_string(starkInfo.nStages + 1) + "_2ns";
    Goldilocks::Element* pBuffExtended = &params.pols[starkInfo.mapOffsets.section[string2section(sectionStageQ)]];
    uint64_t stride = 2048;
#pragma omp parallel for
    for (uint64_t ii = 0; ii < N; ii += stride)
    {
        Goldilocks::Element curS = Goldilocks::one();
        for (uint64_t p = 0; p < starkInfo.qDeg; p++)
        {
            for (uint64_t k = ii; k < min(N, ii + stride); ++k)
            {
                Goldilocks3::mul((Goldilocks3::Element &)*qq2[k * starkInfo.qDeg + p], (Goldilocks3::Element &)*qq1[p * N + k], curS);
            }
            curS = Goldilocks::mul(curS, shiftIn);
        }
    }
    TimerStopAndLogStep(STARK_CALCULATE_EXPS_2NS_MUL, step);

    TimerStartStep(STARK_CALCULATE_EXPS_2NS_NTT, step);
    nttExtended.NTT(pBuffExtended, qq2.address(), NExtended, starkInfo.qDim * starkInfo.qDeg);
    TimerStopAndLogStep(STARK_CALCULATE_EXPS_2NS_NTT, step);

    TimerStartStep(STARK_MERKLETREE, step);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proofs.roots[step - 1][0]);

    TimerStopAndLogStep(STARK_MERKLETREE, step);
}

template <typename ElementType>
void Starks<ElementType>::computeEvals(StepsParams& params, FRIProof<ElementType> &proof) {
    TimerStart(STARK_CALCULATE_LEv);
    
    vector<uint64_t> openingPoints = starkInfo.openingPoints;

    Polinomial LEv(openingPoints.size() * N, FIELD_EXTENSION);

    Polinomial w(openingPoints.size(), FIELD_EXTENSION);
    Polinomial c_w(openingPoints.size(), FIELD_EXTENSION);
    Polinomial xi(openingPoints.size(), FIELD_EXTENSION);

    for (uint64_t i = 0; i < openingPoints.size(); ++i) {
        uint64_t offset = i*N;
        Goldilocks3::one((Goldilocks3::Element &)*LEv[offset]);
        uint64_t opening = openingPoints[i] < 0 ? -openingPoints[i] : openingPoints[i];
        Goldilocks3::one((Goldilocks3::Element &)*w[i]);
        for (uint64_t j = 0; j < opening; ++j) {
            Polinomial::mulElement(w, i, w, i, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));
        }

        if(openingPoints[i] < 0) {
            Polinomial::divElement(w, i, (Goldilocks::Element &)Goldilocks::one(), w, i);
        }

        Polinomial::mulElement(c_w, i, params.challenges, 7, w, i);

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
Polinomial* Starks<ElementType>::computeFRIPol(uint64_t step, StepsParams& params, CHelpersSteps *chelpersSteps) {

    TimerStart(STARK_CALCULATE_XDIVXSUB);

    vector<uint64_t> openingPoints = starkInfo.openingPoints;

    Polinomial xi(openingPoints.size(), FIELD_EXTENSION);
    Polinomial w(openingPoints.size(), FIELD_EXTENSION);

    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

#pragma omp parallel for
    for (uint64_t i = 0; i < openingPoints.size(); ++i) {
        uint64_t opening = openingPoints[i] < 0 ? -openingPoints[i] : openingPoints[i];
        Goldilocks3::one((Goldilocks3::Element &)*w[i]);
        for (uint64_t j = 0; j < opening; ++j) {
            Polinomial::mulElement(w, i, w, i, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));
        }

        if(openingPoints[i] < 0) {
            Polinomial::divElement(w, i, (Goldilocks::Element &)Goldilocks::one(), w, i);
        }

        Polinomial::mulElement(xi, i, params.challenges, 7, w, i);

        #pragma omp parallel for
        for (uint64_t k = 0; k < (N << extendBits); k++) {
            Polinomial::subElement(params.xDivXSubXi, k + i * NExtended, x, k, xi, i);
        }
    }

    Polinomial::batchInverseParallel(params.xDivXSubXi, params.xDivXSubXi);

#pragma omp parallel for
    for (uint64_t i = 0; i < openingPoints.size(); ++i) {
        for (uint64_t k = 0; k < (N << extendBits); k++) {
            Polinomial::mulElement(params.xDivXSubXi, k + i*NExtended, params.xDivXSubXi, k + i*NExtended, x, k);
        }
    }
    TimerStopAndLog(STARK_CALCULATE_XDIVXSUB);

    calculateExpressions(step, false, params, chelpersSteps);

    Polinomial *friPol = new Polinomial(params.f_2ns, NExtended, FIELD_EXTENSION, FIELD_EXTENSION, "friPol");

    return friPol;
}

template <typename ElementType>
void Starks<ElementType>::computeFRIFolding(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t step, Polinomial &challenge) {
    FRI<ElementType>::fold(step, fproof, friPol, challenge, starkInfo, treesFRI);
}

template <typename ElementType>
void Starks<ElementType>::computeFRIQueries(FRIProof<ElementType> &fproof, Polinomial &friPol, uint64_t* friQueries) {
    FRI<ElementType>::proveQueries(friQueries, fproof, treesGL, treesFRI, starkInfo);
}

template <typename ElementType>
void Starks<ElementType>::transposePolsColumns(StepsParams& params, Polinomial* transPols, uint64_t &indx, Hint hint, Goldilocks::Element *pBuffer) {
    u_int64_t stride_pol_ = N * FIELD_EXTENSION + 8;

    for(uint64_t i = 0; i < hint.fields.size(); i++) {
        uint64_t id = hint.fieldId[hint.fields[i]];
        Polinomial p = starkInfo.getPolinomial(params.pols, starkInfo.getPolinomialRef("exp", id), N);
        transPols[indx].potConstruct(&(pBuffer[indx * stride_pol_]), p.degree(), p.dim(), p.dim());
        Polinomial::copy(transPols[indx], p);
        indx++;
    }

    for(uint64_t i = 0; i < hint.dests.size(); i++) {
        uint64_t id = hint.destId[hint.dests[i]];
        Polinomial p = starkInfo.getPolinomial(params.pols, starkInfo.getPolinomialRef("cm_n", id), N);
        transPols[indx].potConstruct(&(pBuffer[indx * stride_pol_]), p.degree(), p.dim(), p.dim());
        cm2Transposed[id] = indx;
        Polinomial::copy(transPols[indx], p);
        indx++;
    }
}

template <typename ElementType>
void Starks<ElementType>::transposePolsRows(uint64_t step, StepsParams& params, Polinomial *transPols)
{
    for (uint64_t i = 0; i < starkInfo.hints[step].size(); i++)
    {
        for(uint64_t j = 0; j < starkInfo.hints[step][i].dests.size(); j++) {
            uint64_t polId = starkInfo.hints[step][i].destId[starkInfo.hints[step][i].dests[j]];
            uint64_t transposedId = cm2Transposed[polId];
            Polinomial cmPol = starkInfo.getPolinomial(params.pols, starkInfo.getPolinomialRef("cm_n", polId), N);
            Polinomial::copy(cmPol, transPols[transposedId]);
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::calculateHints(uint64_t step, StepsParams& params) {
    std::string sectionExtended = "cm" + to_string(step) + "_2ns";
    uint64_t sectionExtendedOffset = starkInfo.mapOffsets.section[string2section(sectionExtended)];
    Goldilocks::Element *pBuffer = &params.pols[sectionExtendedOffset];

    uint64_t numHints = starkInfo.hints[step].size();
    uint64_t numPols = 0;
    for(uint64_t i = 0; i < numHints; ++i) {
        numPols += starkInfo.hints[step][i].fields.size() + starkInfo.hints[step][i].dests.size();
    }

    Polinomial *transPols = new Polinomial[numPols];
    
    TimerStartStep(STARK_CALCULATE_TRANSPOSE, step);
    uint64_t indx = 0;
    for(uint64_t i = 0; i < numHints; ++i) {
        transposePolsColumns(params, transPols, indx, starkInfo.hints[step][i], pBuffer);
    }
    TimerStopAndLogStep(STARK_CALCULATE_TRANSPOSE, step);

    TimerStartStep(STARK_CALCULATE_HINTS, step);
    uint64_t *mem_ = (uint64_t *)pAddress;
    uint64_t *pbufferH = &mem_[sectionExtendedOffset + numPols * (N * FIELD_EXTENSION + 8)];
    
    uint64_t maxThreads = omp_get_max_threads();
    uint64_t nThreads = numHints > maxThreads ? maxThreads : numHints;

#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < numHints; i++)
    {
        int index = starkInfo.hints[step][i].index;
        
        if(starkInfo.hints[step][i].type == hintType::h1h2) {
            if (transPols[index + 2].dim() == 1)
            {
                Polinomial::calculateH1H2_opt1(transPols[index + 2], transPols[index + 3], transPols[index], transPols[index + 1], i, &pbufferH[omp_get_thread_num() * sizeof(Goldilocks::Element) * N], (sizeof(Goldilocks::Element) - 3) * N);
            }
            else if(transPols[index + 2].dim() == 3)
            {
                Polinomial::calculateH1H2_opt3(transPols[index + 2], transPols[index + 3], transPols[index], transPols[index + 1], i, &pbufferH[omp_get_thread_num() * sizeof(Goldilocks::Element) * N], (sizeof(Goldilocks::Element) - 5) * N);
            } else {
                std::cerr << "Error: calculateH1H2_ invalid" << std::endl;   
                exit(-1);
            }
        } else if(starkInfo.hints[step][i].type == hintType::gprod) {
            Polinomial::calculateZ(transPols[index + 2], transPols[index], transPols[index + 1]);
        } else {
            zklog.error("Invalid hint type=" + starkInfo.hints[step][i].type);
            exitProcess();
            exit(-1);
        }
    }
    TimerStopAndLogStep(STARK_CALCULATE_HINTS, step);

    TimerStartStep(STARK_CALCULATE_TRANSPOSE_2, step);
    transposePolsRows(step, params, transPols);
    TimerStopAndLogStep(STARK_CALCULATE_TRANSPOSE_2, step);

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
    for(uint64_t k = 0; k < N; ++k) {
        for (uint64_t i = 0; i < openingPoints.size(); ++i) {
            Goldilocks::Element *LEv_ = &LEv[i*N + k][0]; 
            LEv_Helpers[i*N + k][0] = LEv_[0] + LEv_[1];
            LEv_Helpers[i*N + k][1] = LEv_[0] + LEv_[2];
            LEv_Helpers[i*N + k][2] = LEv_[1] + LEv_[2];
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
        else if (ev.type == EvMap::eType::cm)
        {
            Polinomial pol = starkInfo.getPolinomial(params.pols, starkInfo.getPolinomialRef("cm_2ns", ev.id), NExtended);
            map_offsets[reinterpret_cast<std::uintptr_t>(pol.address())].push_back(i);
        }
        else if (ev.type == EvMap::eType::q)
        {
            Polinomial pol = starkInfo.getPolinomial(params.pols,  starkInfo.getPolinomialRef("q", ev.id), NExtended);
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
            else if (ev.type == EvMap::eType::cm)
            {
                ordPols[kk] = starkInfo.getPolinomial(params.pols, starkInfo.getPolinomialRef("cm_2ns", ev.id), NExtended);
            }
            else if (ev.type == EvMap::eType::q)
            {
                ordPols[kk] = starkInfo.getPolinomial(params.pols, starkInfo.getPolinomialRef("q", ev.id), NExtended);
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
                Polinomial::mulAddElement_adim3(&(evals_acc[thread_idx][i * FIELD_EXTENSION]),  &(LEv[index*N + k][0]), &(LEv_Helpers[index*N + k][0]), ordPols[i], k << extendBits);
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
int Starks<ElementType>::findIndex(std::vector<uint64_t> openingPoints, int prime) {
    auto it = std::find_if(openingPoints.begin(), openingPoints.end(), [prime](int p) {
        return p == prime;
    });

    if (it != openingPoints.end()) {
        return it - openingPoints.begin();
    } else {
        return -1;
    }
}

template <typename ElementType>
void Starks<ElementType>::getChallenge(TranscriptType &transcript, Goldilocks::Element& challenge) {
    transcript.getField((uint64_t*)&challenge);
}


template <typename ElementType>
void Starks<ElementType>::addTranscriptPublics(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements) {
    transcript.put(buffer, nElements);
};

template <typename ElementType>
void Starks<ElementType>::addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements) {
    transcript.put(buffer, nElements);
};

template <typename ElementType>
void Starks<ElementType>::addTranscript(TranscriptType &transcript, Polinomial& pol) {
    for (uint64_t i = 0; i < pol.degree(); i++) {
        transcript.put(pol[i], pol.dim());
    }
};

template <typename ElementType>
void Starks<ElementType>::merkelizeMemory()
{
    uint64_t polsSize = starkInfo.mapTotalN + starkInfo.mapSectionsN.section[eSection::cm3_2ns] * NExtended;
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
void * Starks<ElementType>::ffi_create_steps_params(Polinomial *pChallenges, Polinomial *pEvals, Polinomial *pXDivXSubXi, Goldilocks::Element *pPublicInputs) {
    StepsParams* params = new StepsParams {
        pols : mem,
        pConstPols : pConstPols,
        pConstPols2ns : pConstPols2ns,
        challenges : *pChallenges,
        x_n : x_n,
        x_2ns : x_2ns,
        zi : zi,
        evals : *pEvals,
        xDivXSubXi : *pXDivXSubXi,
        publicInputs : pPublicInputs,
        q_2ns : &mem[starkInfo.mapOffsets.section[eSection::q_2ns]],
        f_2ns : &mem[starkInfo.mapOffsets.section[eSection::f_2ns]]
    };

    return params;
}

template <typename ElementType>
void Starks<ElementType>::ffi_extend_and_merkelize(uint64_t step, StepsParams* params, FRIProof<ElementType>* proof) {
    extendAndMerkelize(step, *params, *proof);
}

template <typename ElementType>
void Starks<ElementType>::ffi_treesGL_get_root(uint64_t index, ElementType *dst) {
    treesGL[index]->getRoot(dst);
}

