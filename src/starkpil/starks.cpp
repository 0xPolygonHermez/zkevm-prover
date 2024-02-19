#include "definitions.hpp"
#include "starks.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"


USING_PROVER_FORK_NAMESPACE;

template <typename ElementType>
void Starks<ElementType>::genProof(FRIProof<ElementType> &proof, Goldilocks::Element *publicInputs, Steps *steps)
{
    // Initialize vars
    TimerStart(STARK_INITIALIZATION);

    TranscriptType transcript;

    Polinomial evals(starkInfo.evMap.size(), FIELD_EXTENSION);
    Polinomial challenges(starkInfo.nChallenges, FIELD_EXTENSION);

    Polinomial xDivXSub(starkInfo.openingPoints.size() * NExtended, FIELD_EXTENSION);

    Polinomial xDivXSubXi(xDivXSub[0], NExtended, FIELD_EXTENSION, FIELD_EXTENSION);
    Polinomial xDivXSubWXi(xDivXSub[NExtended], NExtended, FIELD_EXTENSION, FIELD_EXTENSION);

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
        xDivXSubWXi : xDivXSubWXi,
        publicInputs : publicInputs,
        q_2ns : &mem[starkInfo.mapOffsets.section[eSection::q_2ns]],
        f_2ns : &mem[starkInfo.mapOffsets.section[eSection::f_2ns]]
    };

    uint64_t step = 1;
    
    TimerStopAndLog(STARK_INITIALIZATION);

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    addTranscript(transcript, &verkey[0], hashSize);
    addTranscriptPublics(transcript, &publicInputs[0], starkInfo.nPublics);

    //--------------------------------
    // 1.- Calculate Stage 1
    //--------------------------------
    TimerStart(STARK_STEP_1);
   
    extendAndMerkelize(step, params, proof);

    addTranscript(transcript, &proof.proofs.roots[step - 1][0], hashSize);

    //--------------------------------
    // 2.- Calculate plookups h1 and h2
    //--------------------------------
    TimerStart(STARK_STEP_2);
    step = 2;
    getChallenges(transcript, params.challenges[0], starkInfo.numChallenges[step - 1]);

    calculateExpressions("step2prev", nrowsStepBatch, steps, params, N);

    calculateH1H2(params);

    extendAndMerkelize(step, params, proof);

    addTranscript(transcript, &proof.proofs.roots[step - 1][0], hashSize);

    TimerStopAndLog(STARK_STEP_2);
    //--------------------------------
    // 3.- Compute Z polynomials
    //--------------------------------
    TimerStart(STARK_STEP_3);
    step = 3;

    getChallenges(transcript, params.challenges[2], starkInfo.numChallenges[step - 1]);
    
    calculateExpressions("step3prev", nrowsStepBatch, steps, params, N);

    calculateZ(params);
    
    calculateExpressions("step3", nrowsStepBatch, steps, params, N);

    extendAndMerkelize(step, params, proof);

    addTranscript(transcript, &proof.proofs.roots[step - 1][0], hashSize);

    TimerStopAndLog(STARK_STEP_3);

    //--------------------------------
    // 4. Compute C Polynomial
    //--------------------------------
    TimerStart(STARK_STEP_4);
    step = 4;

    getChallenges(transcript, params.challenges[4], 1);
    
    calculateExpressions("step42ns", nrowsStepBatch, steps, params, NExtended);

    computeQ(params, proof);

    addTranscript(transcript, &proof.proofs.roots[step - 1][0], hashSize);

    TimerStopAndLog(STARK_STEP_4);

    //--------------------------------
    // 5. Compute Evals
    //--------------------------------
    TimerStart(STARK_STEP_5);

    getChallenges(transcript, params.challenges[7], 1);

    computeEvals(params, proof);

    addTranscript(transcript, evals);

    getChallenges(transcript, params.challenges[5], 2);

    Polinomial* friPol = computeFRIPol(params, steps, nrowsStepBatch);

    TimerStopAndLog(STARK_STEP_5);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);
    
    for (uint64_t step = 0; step < starkInfo.starkStruct.steps.size(); step++) {
        Polinomial challenge(1, FIELD_EXTENSION);
        getChallenges(transcript, challenge[0], 1);
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
}

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelize(uint64_t step, StepsParams& params, FRIProof<ElementType> &proof) {
    TimerStart(STARK_STEP_LDE_AND_MERKLETREE);
    TimerStart(STARK_STEP_LDE);
    
    std::string section = "cm" + to_string(step) + "_n";
    std::string sectionExtended = "cm" + to_string(step) + "_2ns";
    std::string nextSectionExtended = "cm" + to_string(step + 1) + "_2ns";

    uint64_t nCols = starkInfo.mapSectionsN.section[string2section(section)];

    Goldilocks::Element* pBuff = &params.pols[starkInfo.mapOffsets.section[string2section(section)]];
    Goldilocks::Element* pBuffExtended = &params.pols[starkInfo.mapOffsets.section[string2section(sectionExtended)]];
    Goldilocks::Element* pBuffHelper = &params.pols[starkInfo.mapOffsets.section[string2section(nextSectionExtended)]];
      
    ntt.extendPol(pBuffExtended, pBuff, 1 << starkInfo.starkStruct.nBitsExt, 1 << starkInfo.starkStruct.nBits, nCols, pBuffHelper);
    TimerStopAndLog(STARK_STEP_LDE);
    TimerStart(STARK_STEP_MERKLETREE);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proofs.roots[step - 1][0]);
    TimerStopAndLog(STARK_STEP_MERKLETREE);
    TimerStopAndLog(STARK_STEP_LDE_AND_MERKLETREE);
}

template <typename ElementType>
void Starks<ElementType>::computeQ(StepsParams& params, FRIProof<ElementType> &proof) {
    uint64_t step = starkInfo.nStages + 1;

    TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_INTT);
    Polinomial qq1 = Polinomial(NExtended, starkInfo.qDim, "qq1");
    Polinomial qq2 = Polinomial(NExtended * starkInfo.qDeg, starkInfo.qDim, "qq2");
    nttExtended.INTT(qq1.address(), &params.pols[starkInfo.mapOffsets.section[eSection::q_2ns]], NExtended, starkInfo.qDim, NULL, 2, 1);
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_INTT);

    TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_MUL);
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
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_MUL);

    TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_NTT);
    nttExtended.NTT(pBuffExtended, qq2.address(), NExtended, starkInfo.qDim * starkInfo.qDeg);
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_NTT);

    TimerStart(STARK_STEP_4_MERKLETREE);
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proofs.roots[step - 1][0]);

    TimerStopAndLog(STARK_STEP_4_MERKLETREE);
}

template <typename ElementType>
void Starks<ElementType>::computeEvals(StepsParams& params, FRIProof<ElementType> &proof) {
    TimerStart(STARK_STEP_5_LEv);
    
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
    
    TimerStopAndLog(STARK_STEP_5_LEv);

    TimerStart(STARK_STEP_5_EVMAP);
    evmap(params, LEv);
    proof.proofs.setEvals(params.evals.address());
    TimerStopAndLog(STARK_STEP_5_EVMAP);
}

template <typename ElementType>
Polinomial* Starks<ElementType>::computeFRIPol(StepsParams& params, Steps *steps, uint64_t nrowsStepBatch) {

    TimerStart(STARK_STEP_5_XDIVXSUB);

    vector<uint64_t> openingPoints = starkInfo.openingPoints;

    Polinomial xi(openingPoints.size(), FIELD_EXTENSION);
    Polinomial w(openingPoints.size(), FIELD_EXTENSION);

    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

    for (uint64_t i = 0; i < openingPoints.size(); ++i) {
        Polinomial& xDiv = i == 0 ? params.xDivXSubXi : params.xDivXSubWXi;
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
            Polinomial::subElement(xDiv, k, x, k, xi, i);
        }

        Polinomial::batchInverseParallel(xDiv, xDiv);

    #pragma omp parallel for
        for (uint64_t k = 0; k < (N << extendBits); k++) {
            Polinomial::mulElement(xDiv, k, xDiv, k, x, k);
        }

    }
    TimerStopAndLog(STARK_STEP_5_XDIVXSUB);

    calculateExpressions("step52ns", nrowsStepBatch, steps, params, NExtended);

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
void Starks<ElementType>::calculateExpressions(std::string step, uint64_t nrowsStepBatch, Steps *steps, StepsParams &params, uint64_t N) {
    if (nrowsStepBatch == 4)
    {
        TimerStart(STARK_STEP_CALCULATE_EXPS_AVX);
        if(step == "step2prev") {
            steps->step2prev_parser_first_avx(params, N, nrowsStepBatch);
        } else if (step == "step3prev") {
            steps->step3prev_parser_first_avx(params, N, nrowsStepBatch);
        } else if (step == "step3") {
            steps->step3_parser_first_avx(params, N, nrowsStepBatch);
        } else if (step == "step42ns") {
            steps->step42ns_parser_first_avx(params, N, nrowsStepBatch);
        } else if (step == "step52ns") {
            steps->step52ns_parser_first_avx(params, N, nrowsStepBatch);
        }
        TimerStopAndLog(STARK_STEP_CALCULATE_EXPS_AVX);
    }
    else if (nrowsStepBatch == 8)
    {
        TimerStart(STARK_STEP_CALCULATE_EXPS_AVX512);
        if(step == "step2prev") {
            steps->step2prev_parser_first_avx512(params, N, nrowsStepBatch);
        } else if (step == "step3prev") {
            steps->step3prev_parser_first_avx512(params, N, nrowsStepBatch);
        } else if (step == "step3") {
            steps->step3_parser_first_avx512(params, N, nrowsStepBatch);
        } else if (step == "step42ns") {
            steps->step42ns_parser_first_avx512(params, N, nrowsStepBatch);
        } else if (step == "step52ns") {
            steps->step52ns_parser_first_avx512(params, N, nrowsStepBatch);
        }
        TimerStopAndLog(STARK_STEP_CALCULATE_EXPS_AVX512);
    }
    else {
        TimerStart(STARK_STEP_CALCULATE_EXPS);
        if(step == "step2prev") {
            #pragma omp parallel for
            for (uint64_t i = 0; i < N; i++) steps->step2prev_first(params, i);
        } else if (step == "step3prev") {
            #pragma omp parallel for
            for (uint64_t i = 0; i < N; i++) steps->step3prev_first(params, i);
        } else if (step == "step3") {
            #pragma omp parallel for
            for (uint64_t i = 0; i < N; i++) steps->step3_first(params, i);
        } else if (step == "step42ns") {
            #pragma omp parallel for
            for (uint64_t i = 0; i < N; i++) steps->step42ns_first(params, i);
        } else if (step == "step52ns") {
            #pragma omp parallel for
            for (uint64_t i = 0; i < N; i++) steps->step52ns_first(params, i);
        } 
        TimerStopAndLog(STARK_STEP_CALCULATE_EXPS);
    }
}

template <typename ElementType>
Polinomial *Starks<ElementType>::transposeH1H2Columns(StepsParams& params)
{
    Goldilocks::Element *pBuffer = &params.pols[starkInfo.mapTotalN];
    uint64_t numCommited = starkInfo.nCm1;

    u_int64_t stride_pol0 = N * FIELD_EXTENSION + 8;
    uint64_t tot_pols0 = 4 * starkInfo.puCtx.size();
    Polinomial *transPols = new Polinomial[tot_pols0];

    assert(starkInfo.mapSectionsN.section[eSection::cm1_n] * NExtended * FIELD_EXTENSION >= 3 * tot_pols0 * N);

    // #pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial fPol = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].fExpId)]);
        Polinomial tPol = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].tExpId)]);
        Polinomial h1 = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + i * 2]);
        Polinomial h2 = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + i * 2 + 1]);

        uint64_t indx = i * 4;
        transPols[indx].potConstruct(&(pBuffer[indx * stride_pol0]), fPol.degree(), fPol.dim(), fPol.dim());
        Polinomial::copy(transPols[indx], fPol);
        indx++;
        transPols[indx].potConstruct(&(pBuffer[indx * stride_pol0]), tPol.degree(), tPol.dim(), tPol.dim());
        Polinomial::copy(transPols[indx], tPol);
        indx++;

        transPols[indx].potConstruct(&(pBuffer[indx * stride_pol0]), h1.degree(), h1.dim(), h1.dim());
        indx++;

        transPols[indx].potConstruct(&(pBuffer[indx * stride_pol0]), h2.degree(), h2.dim(), h2.dim());
    }
    return transPols;
}

template <typename ElementType>
void Starks<ElementType>::transposeH1H2Rows(StepsParams& params, Polinomial *transPols)
{
    uint64_t numCommited = starkInfo.nCm1;

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        int indx1 = 4 * i + 2;
        int indx2 = 4 * i + 3;
        Polinomial h1 = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + i * 2]);
        Polinomial h2 = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + i * 2 + 1]);
        Polinomial::copy(h1, transPols[indx1]);
        Polinomial::copy(h2, transPols[indx2]);
    }
    if (starkInfo.puCtx.size() > 0)
    {
        delete[] transPols;
    }
}

template <typename ElementType>
Polinomial *Starks<ElementType>::transposeZColumns(StepsParams& params)
{
    Goldilocks::Element *pBuffer = &params.pols[starkInfo.mapTotalN];

    uint64_t numCommited = starkInfo.nCm1 + starkInfo.puCtx.size()*2;

    u_int64_t stride_pol_ = N * FIELD_EXTENSION + 8; // assuming all polinomials have same degree
    uint64_t tot_pols = 3 * (starkInfo.puCtx.size() + starkInfo.peCtx.size() + starkInfo.ciCtx.size());
    Polinomial *newpols_ = new Polinomial[tot_pols];
    assert(starkInfo.mapSectionsN.section[eSection::cm1_n] * NExtended * FIELD_EXTENSION >= tot_pols * stride_pol_);

    if (pBuffer == NULL || newpols_ == NULL)
    {
        zklog.error("Starks::transposeZColumns() failed calling new Polinomial[" + to_string(tot_pols) + "]");
        exitProcess();
    }

    // #pragma omp parallel for (better without)
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].numId)]);
        Polinomial pDen = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].denId)]);
        Polinomial z = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + i]);
        u_int64_t indx = i * 3;
        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), pNum.degree(), pNum.dim(), pNum.dim());
        Polinomial::copy(newpols_[indx], pNum);
        indx++;
        assert(pNum.degree() <= N);

        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), pDen.degree(), pDen.dim(), pDen.dim());
        Polinomial::copy(newpols_[indx], pDen);
        indx++;
        assert(pDen.degree() <= N);

        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), z.degree(), z.dim(), z.dim());
        assert(z.degree() <= N);
    }
    u_int64_t offset = 3 * starkInfo.puCtx.size();
    for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.peCtx[i].numId)]);
        Polinomial pDen = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.peCtx[i].denId)]);
        Polinomial z = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + starkInfo.puCtx.size() + i]);
        u_int64_t indx = 3 * i + offset;
        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), pNum.degree(), pNum.dim(), pNum.dim());
        Polinomial::copy(newpols_[indx], pNum);
        indx++;
        assert(pNum.degree() <= N);

        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), pDen.degree(), pDen.dim(), pDen.dim());
        Polinomial::copy(newpols_[indx], pDen);
        indx++;
        assert(pDen.degree() <= N);

        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), z.degree(), z.dim(), z.dim());
        assert(z.degree() <= N);
    }
    offset += 3 * starkInfo.peCtx.size();
    for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
    {

        Polinomial pNum = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.ciCtx[i].numId)]);
        Polinomial pDen = starkInfo.getPolinomial(params.pols, starkInfo.exp2pol[to_string(starkInfo.ciCtx[i].denId)]);
        Polinomial z = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + starkInfo.puCtx.size() + starkInfo.peCtx.size() + i]);
        u_int64_t indx = 3 * i + offset;

        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), pNum.degree(), pNum.dim(), pNum.dim());
        Polinomial::copy(newpols_[indx], pNum);
        indx++;
        assert(pNum.degree() <= N);

        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), pDen.degree(), pDen.dim(), pDen.dim());
        Polinomial::copy(newpols_[indx], pDen);
        indx++;
        assert(pDen.degree() <= N);

        newpols_[indx].potConstruct(&(pBuffer[indx * stride_pol_]), z.degree(), z.dim(), z.dim());
        assert(z.degree() <= N);
    }
    return newpols_;
}

template <typename ElementType>
void Starks<ElementType>::transposeZRows(StepsParams& params, Polinomial *transPols)
{
    u_int64_t numpols = starkInfo.ciCtx.size() + starkInfo.peCtx.size() + starkInfo.puCtx.size();
    uint64_t numCommited = starkInfo.nCm1 + starkInfo.puCtx.size()*2;
    for (uint64_t i = 0; i < numpols; i++)
    {
        int indx1 = 3 * i;
        Polinomial z = starkInfo.getPolinomial(params.pols, starkInfo.cm_n[numCommited + i]);
        Polinomial::copy(z, transPols[indx1 + 2]);
    }
    if (numpols > 0)
    {
        delete[] transPols;
    }
}

template <typename ElementType>
void Starks<ElementType>::calculateH1H2(StepsParams& params) {
    TimerStart(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE);
    Polinomial *transPols = transposeH1H2Columns(params);
    TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE);
    TimerStart(STARK_STEP_2_CALCULATEH1H2);

    uint64_t nthreads = starkInfo.puCtx.size();
    if (nthreads == 0)
    {
        nthreads += 1;
    }
    uint64_t buffSize = 8 * starkInfo.puCtx.size() * N;
    assert(buffSize <= starkInfo.mapSectionsN.section[eSection::cm3_2ns] * NExtended);
    uint64_t *mam = (uint64_t *)pAddress;
    uint64_t *pbufferH = &mam[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
    uint64_t buffSizeThread = buffSize / nthreads;

#pragma omp parallel for num_threads(nthreads)
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        int indx1 = 4 * i;
        if (transPols[indx1 + 2].dim() == 1)
        {
            uint64_t buffSizeThreadValues = 3 * N;
            uint64_t buffSizeThreadKeys = buffSizeThread - buffSizeThreadValues;
            Polinomial::calculateH1H2_opt1(transPols[indx1 + 2], transPols[indx1 + 3], transPols[indx1], transPols[indx1 + 1], i, &pbufferH[omp_get_thread_num() * buffSizeThread], buffSizeThreadKeys, buffSizeThreadValues);
        }
        else
        {
            assert(transPols[indx1 + 2].dim() == 3);
            uint64_t buffSizeThreadValues = 5 * N;
            uint64_t buffSizeThreadKeys = buffSizeThread - buffSizeThreadValues;
            Polinomial::calculateH1H2_opt3(transPols[indx1 + 2], transPols[indx1 + 3], transPols[indx1], transPols[indx1 + 1], i, &pbufferH[omp_get_thread_num() * buffSizeThread], buffSizeThreadKeys, buffSizeThreadValues);
        }
    }
    TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2);

    TimerStart(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE_2);
    transposeH1H2Rows(params, transPols);
    TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE_2);
}

template <typename ElementType>
void Starks<ElementType>::calculateZ(StepsParams& params) {
    TimerStart(STARK_STEP_3_CALCULATE_Z_TRANSPOSE);
    Polinomial *newpols_ = transposeZColumns(params);
    TimerStopAndLog(STARK_STEP_3_CALCULATE_Z_TRANSPOSE);

    TimerStart(STARK_STEP_3_CALCULATE_Z);
    u_int64_t numpols = starkInfo.ciCtx.size() + starkInfo.peCtx.size() + starkInfo.puCtx.size();
#pragma omp parallel for
    for (uint64_t i = 0; i < numpols; i++)
    {
        int indx1 = 3 * i;
        Polinomial::calculateZ(newpols_[indx1 + 2], newpols_[indx1], newpols_[indx1 + 1]);
    }
    TimerStopAndLog(STARK_STEP_3_CALCULATE_Z);
    TimerStart(STARK_STEP_3_CALCULATE_Z_TRANSPOSE_2);
    transposeZRows(params, newpols_);
    TimerStopAndLog(STARK_STEP_3_CALCULATE_Z_TRANSPOSE_2);
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
            Polinomial pol = starkInfo.getPolinomial(params.pols, starkInfo.cm_2ns[ev.id]);
            map_offsets[reinterpret_cast<std::uintptr_t>(pol.address())].push_back(i);
        }
        else if (ev.type == EvMap::eType::q)
        {
            Polinomial pol = starkInfo.getPolinomial(params.pols, starkInfo.qs[ev.id]);
            map_offsets[reinterpret_cast<std::uintptr_t>(pol.address())].push_back(i);
        }
        else
        {
            throw std::invalid_argument("Invalid ev type: " + ev.type);
        }
    }
    Polinomial *ordPols = new Polinomial[size_eval];
    vector<bool> isPrime(size_eval);
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
                ordPols[kk] = starkInfo.getPolinomial(params.pols, starkInfo.cm_2ns[ev.id]);
            }
            else if (ev.type == EvMap::eType::q)
            {
                ordPols[kk] = starkInfo.getPolinomial(params.pols, starkInfo.qs[ev.id]);
            }
            isPrime[kk] = ev.prime;
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
                int index = findIndex(openingPoints, isPrime[i]);
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
void Starks<ElementType>::getChallenges(TranscriptType &transcript, Goldilocks::Element* challenges, uint64_t nChallenges) {
    for(uint64_t i = 0; i < nChallenges; i++) {
        transcript.getField((uint64_t*)&challenges[i*FIELD_EXTENSION]);
    }
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
    uint64_t polsSize = starkInfo.mapTotalN + starkInfo.mapSectionsN.section[eSection::cm3_2ns] * (1 << starkInfo.starkStruct.nBitsExt);
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
void * Starks<ElementType>::ffi_create_steps_params(Polinomial *pChallenges, Polinomial *pEvals, Polinomial *pXDivXSubXi, Polinomial *pXDivXSubWXi, Goldilocks::Element *pPublicInputs) {
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
        xDivXSubWXi : *pXDivXSubWXi,
        publicInputs : pPublicInputs,
        q_2ns : &mem[starkInfo.mapOffsets.section[eSection::q_2ns]],
        f_2ns : &mem[starkInfo.mapOffsets.section[eSection::f_2ns]]
    };

    return params;
}

// void Starks::treeMerkelize(uint64_t index) {
//     treesGL[index]->merkelize();
// }
// void Starks::treeGetRoot(uint64_t index, Goldilocks::Element *root) {
//     treesGL[index]->getRoot(root);
// }

// void Starks::extendPol(uint64_t step) {
//     if(step == 1) {
//         ntt.extendPol(p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n], p_cm2_2ns);
//     } else if(step == 2) {
//         ntt.extendPol(p_cm2_2ns, p_cm2_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm2_n], pBuffer);
//     } else if(step == 3) {
//         ntt.extendPol(p_cm3_2ns, p_cm3_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm3_n], pBuffer);
//     }
// }

// void *Starks::getPBuffer() {
//     return pBuffer;
// }

// void Starks::ffi_calculateH1H2(Polinomial *transPols) {
//     uint64_t nthreads = starkInfo.puCtx.size();
//     if (nthreads == 0)
//     {
//         nthreads += 1;
//     }
//     uint64_t buffSize = 8 * starkInfo.puCtx.size() * N;
//     assert(buffSize <= starkInfo.mapSectionsN.section[eSection::cm3_2ns] * NExtended);
//     uint64_t *mam = (uint64_t *)pAddress;
//     uint64_t *pbufferH = &mam[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
//     uint64_t buffSizeThread = buffSize / nthreads;

// #pragma omp parallel for num_threads(nthreads)
//     for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
//     {
//         int indx1 = 4 * i;
//         if (transPols[indx1 + 2].dim() == 1)
//         {
//             uint64_t buffSizeThreadValues = 3 * N;
//             uint64_t buffSizeThreadKeys = buffSizeThread - buffSizeThreadValues;
//             Polinomial::calculateH1H2_opt1(transPols[indx1 + 2], transPols[indx1 + 3], transPols[indx1], transPols[indx1 + 1], i, &pbufferH[omp_get_thread_num() * buffSizeThread], buffSizeThreadKeys, buffSizeThreadValues);
//         }
//         else
//         {
//             assert(transPols[indx1 + 2].dim() == 3);
//             uint64_t buffSizeThreadValues = 5 * N;
//             uint64_t buffSizeThreadKeys = buffSizeThread - buffSizeThreadValues;
//             Polinomial::calculateH1H2_opt3(transPols[indx1 + 2], transPols[indx1 + 3], transPols[indx1], transPols[indx1 + 1], i, &pbufferH[omp_get_thread_num() * buffSizeThread], buffSizeThreadKeys, buffSizeThreadValues);
//         }
//     }
// }

// void Starks::ffi_calculateZ(Polinomial *newPols) {
//     u_int64_t numpols = starkInfo.ciCtx.size() + starkInfo.peCtx.size() + starkInfo.puCtx.size();

// #pragma omp parallel for
//     for (uint64_t i = 0; i < numpols; i++)
//     {
//         int indx1 = 3 * i;
//         Polinomial::calculateZ(newPols[indx1 + 2], newPols[indx1], newPols[indx1 + 1]);
//     }
// }

// void Starks::ffi_exps_2ns(Polinomial *qq1, Polinomial *qq2) {
//     TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_INTT);
//     nttExtended.INTT(qq1->address(), p_q_2ns, NExtended, starkInfo.qDim, NULL, 2, 1);
//     TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_INTT);

//     TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_MUL);
//     Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);

//     uint64_t stride = 2048;
// #pragma omp parallel for
//     for (uint64_t ii = 0; ii < N; ii += stride)
//     {
//         Goldilocks::Element curS = Goldilocks::one();
//         for (uint64_t p = 0; p < starkInfo.qDeg; p++)
//         {
//             for (uint64_t k = ii; k < min(N, ii + stride); ++k)
//             {
//                 Goldilocks3::mul((Goldilocks3::Element &)*(*qq2)[k * starkInfo.qDeg + p], (Goldilocks3::Element &)*(*qq1)[p * N + k], curS);
//             }
//             curS = Goldilocks::mul(curS, shiftIn);
//         }
//     }
//     TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_MUL);

//     TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_NTT);
//     nttExtended.NTT(cm4_2ns, qq2->address(), NExtended, starkInfo.qDim * starkInfo.qDeg);
//     TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_NTT);  
// }

// void Starks::ffi_lev_lpev(Polinomial *LEv, Polinomial *LpEv, Polinomial *xis, Polinomial *wxis, Polinomial *c_w, Polinomial *challenges) {
//     Goldilocks3::one((Goldilocks3::Element &)*(*LEv)[0]);
//     Goldilocks3::one((Goldilocks3::Element &)*(*LpEv)[0]);

//     Polinomial::divElement(*xis, 0, *challenges, 7, (Goldilocks::Element &)Goldilocks::shift());
//     Polinomial::mulElement(*c_w, 0, *challenges, 7, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));
//     Polinomial::divElement(*wxis, 0, *c_w, 0, (Goldilocks::Element &)Goldilocks::shift());

//     for (uint64_t k = 1; k < N; k++)
//     {
//         Polinomial::mulElement(*LEv, k, *LEv, k - 1, *xis, 0);
//         Polinomial::mulElement(*LpEv, k, *LpEv, k - 1, *wxis, 0);
//     }
//     ntt.INTT((*LEv).address(), (*LEv).address(), N, 3);
//     ntt.INTT((*LpEv).address(), (*LpEv).address(), N, 3);
// }

// void Starks::ffi_xdivxsubxi(uint64_t extendBits, Polinomial *xi, Polinomial *wxi, Polinomial *challenges, Polinomial *xDivXSubXi, Polinomial *xDivXSubWXi) {
//     Polinomial::copyElement(*xi, 0, *challenges, 7);
//     Polinomial::mulElement(*wxi, 0, *challenges, 7, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));

// #pragma omp parallel for
//     for (uint64_t k = 0; k < (N << extendBits); k++)
//     {
//         Polinomial::subElement(*xDivXSubXi, k, x, k, *xi, 0);
//         Polinomial::subElement(*xDivXSubWXi, k, x, k, *wxi, 0);
//     }

//     Polinomial::batchInverseParallel(*xDivXSubXi, *xDivXSubXi);
//     Polinomial::batchInverseParallel(*xDivXSubWXi, *xDivXSubWXi);

// #pragma omp parallel for
//     for (uint64_t k = 0; k < (N << extendBits); k++)
//     {
//         Polinomial::mulElement(*xDivXSubXi, k, *xDivXSubXi, k, x, k);
//         Polinomial::mulElement(*xDivXSubWXi, k, *xDivXSubWXi, k, x, k);
//     }
// }

// void Starks::ffi_finalize_proof(FRIProof *proof, Transcript *transcript, Polinomial *evals, Polinomial *root0, Polinomial *root1, Polinomial *root2, Polinomial *root3) {
//     Polinomial friPol = Polinomial(p_f_2ns, NExtended, 3, 3, "friPol");
//     FRIProve::prove(*proof, treesGL, *transcript, friPol, starkInfo.starkStruct.nBitsExt, starkInfo);

//     (*proof).proofs.setEvals((*evals).address());

//     std::memcpy(&(*proof).proofs.root1[0], (*root0).address(), HASH_SIZE * sizeof(Goldilocks::Element));
//     std::memcpy(&(*proof).proofs.root2[0], (*root1).address(), HASH_SIZE * sizeof(Goldilocks::Element));
//     std::memcpy(&(*proof).proofs.root3[0], (*root2).address(), HASH_SIZE * sizeof(Goldilocks::Element));
//     std::memcpy(&(*proof).proofs.root4[0], (*root3).address(), HASH_SIZE * sizeof(Goldilocks::Element));
// }

template <typename ElementType>
void Starks<ElementType>::ffi_extend_and_merkelize(uint64_t step, StepsParams* params, FRIProof<ElementType>* proof) {
    extendAndMerkelize(step, *params, *proof);
}

template <typename ElementType>
void Starks<ElementType>::ffi_treesGL_get_root(uint64_t index, ElementType *dst) {
    treesGL[index]->getRoot(dst);
}

