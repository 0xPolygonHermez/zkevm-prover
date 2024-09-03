#include "definitions.hpp"
#include "starks.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

USING_PROVER_FORK_NAMESPACE;

void Starks::genProof(FRIProof &proof, Goldilocks::Element *publicInputs, Goldilocks::Element verkey[4], CHelpersSteps *chelpersSteps)
{
    // Initialize vars
    TimerStart(STARK_INITIALIZATION);

    uint64_t numCommited = starkInfo.nCm1;
    Transcript transcript;
    Polinomial evals(starkInfo.evMap.size(), FIELD_EXTENSION, true);
    Polinomial xDivXSubXi(&mem[starkInfo.mapOffsets.section[eSection::xDivXSubXi_2ns]], 2 * NExtended, FIELD_EXTENSION, FIELD_EXTENSION);
    Polinomial challenges(NUM_CHALLENGES, FIELD_EXTENSION, true);

    CommitPols cmPols(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);

    Polinomial root0(HASH_SIZE, 1);
    Polinomial root1(HASH_SIZE, 1);
    Polinomial root2(HASH_SIZE, 1);
    Polinomial root3(HASH_SIZE, 1);

    transcript.put(&verkey[0], 4);
    transcript.put(&publicInputs[0], starkInfo.nPublics);
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
        q_2ns : p_q_2ns,
        f_2ns : p_f_2ns
    };

    if(reduceMemory) {
        treesGL[0]->setSource(p_cm1_2ns_tmp);
        treesGL[1]->setSource(p_cm2_2ns_tmp);
    }
    TimerStopAndLog(STARK_INITIALIZATION);
    //--------------------------------
    // 1.- Calculate p_cm1_2ns
    //--------------------------------
    TimerStart(STARK_STEP_1);
    TimerStart(STARK_STEP_1_LDE_AND_MERKLETREE);
    TimerStart(STARK_STEP_1_LDE);
    string nttHelperStage1 = reduceMemory ? "cm1_tmp" : "cm1";
    std::pair<uint64_t, uint64_t> nttOffsetHelperStage1 = starkInfo.mapNTTOffsetsHelpers[nttHelperStage1];
    Goldilocks::Element *pBuffHelperStage1 = &params.pols[nttOffsetHelperStage1.first];

    uint64_t buffHelperElementsStage1 = NExtended * starkInfo.mapSectionsN.section[cm1_n];

    uint64_t nBlocksStage1 = 1;
    while((nttOffsetHelperStage1.second * nBlocksStage1 < buffHelperElementsStage1) || (starkInfo.mapSectionsN.section[cm1_n] > 256*nBlocksStage1) ) {
        nBlocksStage1++;
    }

    ntt.extendPol(reduceMemory?p_cm1_2ns_tmp:p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n], pBuffHelperStage1, 3, nBlocksStage1);
    TimerStopAndLog(STARK_STEP_1_LDE);

    TimerStart(STARK_STEP_1_MERKLETREE);
    treesGL[0]->merkelize();
    TimerStopAndLog(STARK_STEP_1_MERKLETREE);

    treesGL[0]->getRoot(root0.address());
    zklog.info("MerkleTree rootGL 0: [ " + root0.toString(4) + " ]");
    transcript.put(root0.address(), HASH_SIZE);
    TimerStopAndLog(STARK_STEP_1_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_STEP_1);

    //--------------------------------
    // 2.- Caluculate plookups h1 and h2
    //--------------------------------
    TimerStart(STARK_STEP_2);
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal
    TimerStart(STARK_STEP_2_CALCULATE_EXPS);
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgs, chelpers.stagesInfo["step2"]);
    TimerStopAndLog(STARK_STEP_2_CALCULATE_EXPS);
    TimerStart(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE);
    Polinomial *transPols = transposeH1H2Columns(pAddress, numCommited, params);
    TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE);
    TimerStart(STARK_STEP_2_CALCULATEH1H2);

    uint64_t nthreads = starkInfo.puCtx.size();
    if (nthreads == 0)
    {
        nthreads += 1;
    }
    uint64_t buffSize = sizeof(Goldilocks::Element) * starkInfo.puCtx.size() * N;
    uint64_t buffSizeThread = buffSize / nthreads;

#pragma omp parallel for num_threads(nthreads)
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        int indx1 = 4 * i;
        if (transPols[indx1 + 2].dim() == 1)
        {
            uint64_t buffSizeThreadValues = 3 * N;
            uint64_t buffSizeThreadKeys = buffSizeThread - buffSizeThreadValues;
            Polinomial::calculateH1H2_opt1(transPols[indx1 + 2], transPols[indx1 + 3], transPols[indx1], transPols[indx1 + 1], i, (uint64_t *)(&params.pols[starkInfo.offsetsExtraMemoryH1H2[i]]), buffSizeThreadKeys, buffSizeThreadValues);
        }
        else
        {
            assert(transPols[indx1 + 2].dim() == 3);
            uint64_t buffSizeThreadValues = 5 * N;
            uint64_t buffSizeThreadKeys = buffSizeThread - buffSizeThreadValues;
            Polinomial::calculateH1H2_opt3(transPols[indx1 + 2], transPols[indx1 + 3], transPols[indx1], transPols[indx1 + 1], i, (uint64_t *)(&params.pols[starkInfo.offsetsExtraMemoryH1H2[i]]), buffSizeThreadKeys, buffSizeThreadValues);
        }
    }
    TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2);

    TimerStart(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE_2);
    transposeH1H2Rows(pAddress, numCommited, transPols);
    TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE_2);

    TimerStart(STARK_STEP_2_LDE_AND_MERKLETREE);
    TimerStart(STARK_STEP_2_LDE);
    string nttHelperStage2 = reduceMemory ? "cm2_tmp" : "cm2";
    std::pair<uint64_t, uint64_t> nttOffsetHelperStage2 = starkInfo.mapNTTOffsetsHelpers[nttHelperStage2];
    Goldilocks::Element *pBuffHelperStage2 = &params.pols[nttOffsetHelperStage2.first];

    uint64_t buffHelperElementsStage2 = NExtended * starkInfo.mapSectionsN.section[cm2_n];

    uint64_t nBlocksStage2 = 1;
    while((nttOffsetHelperStage2.second * nBlocksStage2 < buffHelperElementsStage2) || (starkInfo.mapSectionsN.section[cm2_n] > 256*nBlocksStage2) ) {
        nBlocksStage2++;
    }

    ntt.extendPol(reduceMemory?p_cm2_2ns_tmp:p_cm2_2ns, p_cm2_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm2_n], pBuffHelperStage2, 3, nBlocksStage2);
    TimerStopAndLog(STARK_STEP_2_LDE);

    TimerStart(STARK_STEP_2_MERKLETREE);
    treesGL[1]->merkelize();
    TimerStopAndLog(STARK_STEP_2_MERKLETREE);

    treesGL[1]->getRoot(root1.address());
    zklog.info("MerkleTree rootGL 1: [ " + root1.toString(4) + " ]");
    transcript.put(root1.address(), HASH_SIZE);

    TimerStopAndLog(STARK_STEP_2_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_STEP_2);

    //--------------------------------
    // 3.- Compute Z polynomials
    //--------------------------------
    TimerStart(STARK_STEP_3);
    transcript.getField(challenges[2]); // gamma
    transcript.getField(challenges[3]); // betta
    TimerStart(STARK_STEP_3_CALCULATE_EXPS);
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgs, chelpers.stagesInfo["step3"]);
    TimerStopAndLog(STARK_STEP_3_CALCULATE_EXPS);
    TimerStart(STARK_STEP_3_CALCULATE_Z_TRANSPOSE);
    Polinomial *newpols_ = transposeZColumns(pAddress, numCommited, params);
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
    transposeZRows(pAddress, numCommited, newpols_);
    TimerStopAndLog(STARK_STEP_3_CALCULATE_Z_TRANSPOSE_2);
    TimerStart(STARK_STEP_3_LDE_AND_MERKLETREE);
    TimerStart(STARK_STEP_3_LDE);
    std::pair<uint64_t, uint64_t> nttOffsetHelperStage3 = starkInfo.mapNTTOffsetsHelpers["cm3"];
    Goldilocks::Element *pBuffHelperStage3 = &params.pols[nttOffsetHelperStage3.first];

    uint64_t buffHelperElementsStage3 = NExtended * starkInfo.mapSectionsN.section[cm3_n];

    uint64_t nBlocksStage3 = 1;
    while((nttOffsetHelperStage3.second * nBlocksStage3 < buffHelperElementsStage3) || (starkInfo.mapSectionsN.section[cm3_n] > 256*nBlocksStage3) ) {
        nBlocksStage3++;
    }

    ntt.extendPol(p_cm3_2ns, p_cm3_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm3_n], pBuffHelperStage3, 3, nBlocksStage3);
    TimerStopAndLog(STARK_STEP_3_LDE);

    TimerStart(STARK_STEP_3_MERKLETREE);
    treesGL[2]->merkelize();
    TimerStopAndLog(STARK_STEP_3_MERKLETREE);

    treesGL[2]->getRoot(root2.address());
    zklog.info("MerkleTree rootGL 2: [ " + root2.toString(4) + " ]");
    transcript.put(root2.address(), HASH_SIZE);
    TimerStopAndLog(STARK_STEP_3_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_STEP_3);

    if(reduceMemory) {
        TimerStart(STARK_STEP_1_RECALCULATING_LDE);
        std::pair<uint64_t, uint64_t> nttOffsetHelperStage1_ = starkInfo.mapNTTOffsetsHelpers["cm1"];
        Goldilocks::Element *pBuffHelperStage1_ = &params.pols[nttOffsetHelperStage1_.first];

        uint64_t buffHelperElementsStage1_ = NExtended * starkInfo.mapSectionsN.section[cm1_n];

        uint64_t nBlocksStage1_ = 1;
        while((nttOffsetHelperStage1_.second * nBlocksStage1_ < buffHelperElementsStage1_) || (starkInfo.mapSectionsN.section[cm1_n] > 256*nBlocksStage1_) ) {
            nBlocksStage1_++;
        }

        ntt.extendPol(p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n], pBuffHelperStage1_, 3, nBlocksStage1_);
        TimerStopAndLog(STARK_STEP_1_RECALCULATING_LDE);

        TimerStart(STARK_STEP_2_RECALCULATING_LDE);
        std::pair<uint64_t, uint64_t> nttOffsetHelperStage2_ = starkInfo.mapNTTOffsetsHelpers["cm2"];
        Goldilocks::Element *pBuffHelperStage2_ = &params.pols[nttOffsetHelperStage2_.first];

        uint64_t buffHelperElementsStage2_ = NExtended * starkInfo.mapSectionsN.section[cm2_n];

        uint64_t nBlocksStage2_ = 1;
        while((nttOffsetHelperStage2_.second * nBlocksStage2_ < buffHelperElementsStage2_) || (starkInfo.mapSectionsN.section[cm2_n] > 256*nBlocksStage2_) ) {
            nBlocksStage2_++;
        }

        ntt.extendPol(p_cm2_2ns, p_cm2_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm2_n], pBuffHelperStage2_, 3, nBlocksStage2_);
        TimerStopAndLog(STARK_STEP_2_RECALCULATING_LDE);

        treesGL[0]->setSource(p_cm1_2ns);
        treesGL[1]->setSource(p_cm2_2ns);
    }

    //--------------------------------
    // 4. Compute C Polynomial
    //--------------------------------
    TimerStart(STARK_STEP_4);
    TimerStart(STARK_STEP_4_INIT);
    transcript.getField(challenges[4]); // gamma
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    TimerStopAndLog(STARK_STEP_4_INIT);
    TimerStart(STARK_STEP_4_CALCULATE_EXPS);
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgs, chelpers.stagesInfo["step4"]);
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS);
    TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_INTT);
    std::pair<uint64_t, uint64_t> nttOffsetHelperStage4 = starkInfo.mapNTTOffsetsHelpers["cm4"];
    Goldilocks::Element *pBuffHelperStage4 = &params.pols[nttOffsetHelperStage4.first];

    uint64_t buffHelperElementsStage4 = NExtended * starkInfo.mapSectionsN.section[eSection::cm4_2ns];
    
    uint64_t nBlocksStage4 = 1;
    while((nttOffsetHelperStage4.second * nBlocksStage4 < buffHelperElementsStage4) || (starkInfo.mapSectionsN.section[eSection::cm4_2ns] > 256*nBlocksStage4) ) {
        nBlocksStage4++;
    }
    nttExtended.INTT(params.q_2ns, params.q_2ns, NExtended, starkInfo.qDim, pBuffHelperStage4, 3, nBlocksStage4);
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_INTT);

    TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_MUL);
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
                Goldilocks3::mul((Goldilocks3::Element &)cm4_2ns[(k * starkInfo.qDeg + p) * FIELD_EXTENSION], (Goldilocks3::Element &)params.q_2ns[(p * N + k) * FIELD_EXTENSION], curS);
            }
            curS = Goldilocks::mul(curS, shiftIn);
        }
    }

    memset(&cm4_2ns[N * starkInfo.qDeg * starkInfo.qDim], 0, (NExtended - N) * starkInfo.qDeg * starkInfo.qDim * sizeof(Goldilocks::Element));

    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_MUL);

    TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS_NTT);
    nttExtended.NTT(cm4_2ns, cm4_2ns, NExtended, starkInfo.mapSectionsN.section[eSection::cm4_2ns], pBuffHelperStage4, 3, nBlocksStage4);
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS_NTT);

    TimerStart(STARK_STEP_4_MERKLETREE);

    treesGL[3]->merkelize();
    treesGL[3]->getRoot(root3.address());
    zklog.info("MerkleTree rootGL 3: [ " + root3.toString(4) + " ]");
    transcript.put(root3.address(), HASH_SIZE);

    TimerStopAndLog(STARK_STEP_4_MERKLETREE);
    TimerStopAndLog(STARK_STEP_4);

    //--------------------------------
    // 5. Compute FRI Polynomial
    //--------------------------------
    TimerStart(STARK_STEP_5);
    TimerStart(STARK_STEP_5_LEv_LpEv);

    // transcript.getField(challenges[5]); // v1
    // transcript.getField(challenges[6]); // v2
    transcript.getField(challenges[7]); // xi

    Polinomial LEv(&params.pols[starkInfo.mapOffsets.section[eSection::LEv]], N, 3, 3, "LEv");
    Polinomial LpEv(&params.pols[starkInfo.mapOffsets.section[eSection::LEv]] + N*FIELD_EXTENSION, N, 3, 3, "LpEv");
    Polinomial xis(1, 3);
    Polinomial wxis(1, 3);
    Polinomial c_w(1, 3);

    Goldilocks3::one((Goldilocks3::Element &)*LEv[0]);
    Goldilocks3::one((Goldilocks3::Element &)*LpEv[0]);

    Polinomial::divElement(xis, 0, challenges, 7, (Goldilocks::Element &)Goldilocks::shift());
    Polinomial::mulElement(c_w, 0, challenges, 7, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));
    Polinomial::divElement(wxis, 0, c_w, 0, (Goldilocks::Element &)Goldilocks::shift());

    for (uint64_t k = 1; k < N; k++)
    {
        Polinomial::mulElement(LEv, k, LEv, k - 1, xis, 0);
        Polinomial::mulElement(LpEv, k, LpEv, k - 1, wxis, 0);
    }
    
    std::pair<uint64_t, uint64_t> nttOffsetHelper = starkInfo.mapNTTOffsetsHelpers["LEv"];
    Goldilocks::Element *pBuffHelperLEv = &params.pols[nttOffsetHelper.first];

    ntt.INTT(LEv.address(), LEv.address(), N, 3, pBuffHelperLEv);
    ntt.INTT(LpEv.address(), LpEv.address(), N, 3, pBuffHelperLEv);

    TimerStopAndLog(STARK_STEP_5_LEv_LpEv);

    TimerStart(STARK_STEP_5_EVMAP);
    evmap(pAddress, evals, LEv, LpEv);
    TimerStopAndLog(STARK_STEP_5_EVMAP);
    TimerStart(STARK_STEP_5_XDIVXSUB);

    for (uint64_t i = 0; i < starkInfo.evMap.size(); i++)
    {
        transcript.put(evals[i], 3);
    }

    transcript.getField(challenges[5]); // v1
    transcript.getField(challenges[6]); // v2

    // Calculate xDivXSubXi
    Polinomial xi(1, FIELD_EXTENSION);
    Polinomial wxi(1, FIELD_EXTENSION);

    Polinomial::copyElement(xi, 0, challenges, 7);
    Polinomial::mulElement(wxi, 0, challenges, 7, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));

#pragma omp parallel for
    for (uint64_t k = 0; k < (N << extendBits); k++)
    {
        Polinomial::subElement(xDivXSubXi, k, x, k, xi, 0);
        Polinomial::subElement(xDivXSubXi, k + NExtended, x, k, wxi, 0);
    }

    Polinomial::batchInverseParallel(xDivXSubXi, xDivXSubXi);

#pragma omp parallel for
    for (uint64_t k = 0; k < (N << extendBits); k++)
    {
        Polinomial::mulElement(xDivXSubXi, k, xDivXSubXi, k, x, k);
        Polinomial::mulElement(xDivXSubXi, k + NExtended, xDivXSubXi, k + NExtended, x, k);
    }
    TimerStopAndLog(STARK_STEP_5_XDIVXSUB);
    TimerStart(STARK_STEP_5_CALCULATE_EXPS);
    chelpersSteps->calculateExpressions(starkInfo, params, chelpers.cHelpersArgs, chelpers.stagesInfo["step5"]);
    TimerStopAndLog(STARK_STEP_5_CALCULATE_EXPS);

    TimerStopAndLog(STARK_STEP_5);
    TimerStart(STARK_STEP_FRI);

    Polinomial friPol = Polinomial(p_f_2ns, NExtended, 3, 3, "friPol");
    FRIProve::prove(proof, treesGL, transcript, friPol, starkInfo.starkStruct.nBitsExt, starkInfo);

    proof.proofs.setEvals(evals.address());

    std::memcpy(&proof.proofs.root1[0], root0.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root2[0], root1.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root3[0], root2.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root4[0], root3.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    TimerStopAndLog(STARK_STEP_FRI);
}

Polinomial *Starks::transposeH1H2Columns(void *pAddress, uint64_t &numCommited, StepsParams &params)
{
    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

    uint64_t tot_pols0 = 4 * starkInfo.puCtx.size();
    Polinomial *transPols = new Polinomial[tot_pols0];

    // #pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        uint64_t fPolId = starkInfo.exp2pol[to_string(starkInfo.puCtx[i].fExpId)];
        uint64_t tPolId = starkInfo.exp2pol[to_string(starkInfo.puCtx[i].tExpId)];

        Polinomial fPol = starkInfo.getPolinomial(mem, fPolId);
        Polinomial tPol = starkInfo.getPolinomial(mem, tPolId);
        Polinomial h1 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2]);
        Polinomial h2 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2 + 1]);

        uint64_t indx = i * 4;
        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[fPolId].section == eSection::tmpExp_n) {
            transPols[indx] = fPol;
        } else {
            transPols[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsH1H2[fPolId]]), fPol.degree(), fPol.dim(), fPol.dim());
        }
        indx++;
        
        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[tPolId].section == eSection::tmpExp_n) {
            transPols[indx] = tPol;
        } else {
            transPols[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsH1H2[tPolId]]), tPol.degree(), tPol.dim(), tPol.dim());
        }
        indx++;

        transPols[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsH1H2[starkInfo.cm_n[numCommited + 2*i]]]), h1.degree(), h1.dim(), h1.dim());
        indx++;

        transPols[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsH1H2[starkInfo.cm_n[numCommited + 2*i + 1]]]), h2.degree(), h2.dim(), h2.dim());
    }

    #pragma omp parallel for
    for(uint64_t l = 0; l < N; l += 4096) {
        for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
        {
            int indx1 = 4 * i;
            int indx2 = 4 * i + 1;
            uint64_t fPolId = starkInfo.exp2pol[to_string(starkInfo.puCtx[i].fExpId)];
            uint64_t tPolId = starkInfo.exp2pol[to_string(starkInfo.puCtx[i].tExpId)];

            Polinomial fPol = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].fExpId)]);
            Polinomial tPol = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].tExpId)]);
            if(!TRANSPOSE_TMP_POLS || starkInfo.varPolMap[fPolId].section != eSection::tmpExp_n) {
                for(uint64_t k = l; k < min(N, l + 4096); ++k) {
                    std::memcpy(transPols[indx1][k], fPol[k], fPol.dim() * sizeof(Goldilocks::Element));
                }
            }

            if(!TRANSPOSE_TMP_POLS || starkInfo.varPolMap[tPolId].section != eSection::tmpExp_n) {
                for(uint64_t k = l; k < min(N, l + 4096); ++k) {
                    std::memcpy(transPols[indx2][k], tPol[k], tPol.dim() * sizeof(Goldilocks::Element));
                }
            }
        }
    }

    return transPols;
}
void Starks::transposeH1H2Rows(void *pAddress, uint64_t &numCommited, Polinomial *transPols)
{
    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

#pragma omp parallel for
    for(uint64_t l = 0; l < N; l += 4096) {
        for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
        {
            int indx1 = 4 * i + 2;
            int indx2 = 4 * i + 3;
            Polinomial h1 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2]);
            Polinomial h2 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2 + 1]);
            for(uint64_t k = l; k < min(N, l + 4096); ++k) {
                std::memcpy(h1[k], transPols[indx1][k], h1.dim() * sizeof(Goldilocks::Element));
                std::memcpy(h2[k], transPols[indx2][k], h2.dim() * sizeof(Goldilocks::Element));
            }
        }
    }
    if (starkInfo.puCtx.size() > 0)
    {
        delete[] transPols;
    }
    numCommited = numCommited + starkInfo.puCtx.size() * 2;
}
Polinomial *Starks::transposeZColumns(void *pAddress, uint64_t &numCommited, StepsParams &params)
{
    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

    uint64_t tot_pols = 3 * (starkInfo.puCtx.size() + starkInfo.peCtx.size() + starkInfo.ciCtx.size());
    Polinomial *newpols_ = new Polinomial[tot_pols];

    // #pragma omp parallel for (better without)
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        uint64_t numId = starkInfo.exp2pol[to_string(starkInfo.puCtx[i].numId)];
        uint64_t denId = starkInfo.exp2pol[to_string(starkInfo.puCtx[i].denId)];
        Polinomial pNum = starkInfo.getPolinomial(mem, numId);
        Polinomial pDen = starkInfo.getPolinomial(mem, denId);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
        u_int64_t indx = i * 3;
        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[numId].section == eSection::tmpExp_n) {
            newpols_[indx] = pNum;
        } else {
            newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[numId]]), pNum.degree(), pNum.dim(), pNum.dim());
            Polinomial::copy(newpols_[indx], pNum);
        }
        indx++;
        assert(pNum.degree() <= N);

        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[denId].section == eSection::tmpExp_n) {
            newpols_[indx] = pDen;
        } else {
            newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[denId]]), pDen.degree(), pDen.dim(), pDen.dim());
            Polinomial::copy(newpols_[indx], pDen);
        }
        indx++;
        assert(pDen.degree() <= N);

        newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[starkInfo.cm_n[numCommited + i]]]), z.degree(), z.dim(), z.dim());
        assert(z.degree() <= N);
    }
    numCommited += starkInfo.puCtx.size();
    u_int64_t offset = 3 * starkInfo.puCtx.size();
    for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
    {
        uint64_t numId = starkInfo.exp2pol[to_string(starkInfo.peCtx[i].numId)];
        uint64_t denId = starkInfo.exp2pol[to_string(starkInfo.peCtx[i].denId)];
        Polinomial pNum = starkInfo.getPolinomial(mem, numId);
        Polinomial pDen = starkInfo.getPolinomial(mem, denId);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
        u_int64_t indx = 3 * i + offset;
        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[numId].section == eSection::tmpExp_n) {
            newpols_[indx] = pNum;
        } else {
            newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[numId]]), pNum.degree(), pNum.dim(), pNum.dim());
            Polinomial::copy(newpols_[indx], pNum);
        }
        indx++;
        assert(pNum.degree() <= N);

        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[denId].section == eSection::tmpExp_n) {
            newpols_[indx] = pDen;
        } else {
            newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[denId]]), pDen.degree(), pDen.dim(), pDen.dim());
            Polinomial::copy(newpols_[indx], pDen);
        }
        indx++;
        assert(pDen.degree() <= N);

        newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[starkInfo.cm_n[numCommited + i]]]), z.degree(), z.dim(), z.dim());
        assert(z.degree() <= N);
    }
    numCommited += starkInfo.peCtx.size();
    offset += 3 * starkInfo.peCtx.size();
    for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
    {

        uint64_t numId = starkInfo.exp2pol[to_string(starkInfo.ciCtx[i].numId)];
        uint64_t denId = starkInfo.exp2pol[to_string(starkInfo.ciCtx[i].denId)];
        Polinomial pNum = starkInfo.getPolinomial(mem, numId);
        Polinomial pDen = starkInfo.getPolinomial(mem, denId);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
        u_int64_t indx = 3 * i + offset;

        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[numId].section == eSection::tmpExp_n) {
            newpols_[indx] = pNum;
        } else {
            newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[numId]]), pNum.degree(), pNum.dim(), pNum.dim());
            Polinomial::copy(newpols_[indx], pNum);
        }
        indx++;
        assert(pNum.degree() <= N);

        if(TRANSPOSE_TMP_POLS && starkInfo.varPolMap[denId].section == eSection::tmpExp_n) {
            newpols_[indx] = pDen;
        } else {
            newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[denId]]), pDen.degree(), pDen.dim(), pDen.dim());
            Polinomial::copy(newpols_[indx], pDen);
        }
        indx++;
        assert(pDen.degree() <= N);

        newpols_[indx].potConstruct(&(params.pols[starkInfo.mapOffsetsPolsGrandProduct[starkInfo.cm_n[numCommited + i]]]), z.degree(), z.dim(), z.dim());
        assert(z.degree() <= N);
    }
    numCommited += starkInfo.ciCtx.size();
    numCommited -= starkInfo.ciCtx.size() + starkInfo.peCtx.size() + starkInfo.puCtx.size();
    return newpols_;
}
void Starks::transposeZRows(void *pAddress, uint64_t &numCommited, Polinomial *transPols)
{
    u_int64_t numpols = starkInfo.ciCtx.size() + starkInfo.peCtx.size() + starkInfo.puCtx.size();
    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;
#pragma omp parallel for
    for(uint64_t l = 0; l < N; l += 4096) {
        for (uint64_t i = 0; i < numpols; i++)
        {
            int indx1 = 3 * i;
            Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
            for(uint64_t k = l; k < min(N, l + 4096); ++k) {
                std::memcpy(z[k], transPols[indx1 + 2][k], 3 * sizeof(Goldilocks::Element));
            }
        }
    }
    if (numpols > 0)
    {
        delete[] transPols;
    }
}
void Starks::evmap(void *pAddress, Polinomial &evals, Polinomial &LEv, Polinomial &LpEv)
{
    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    u_int64_t size_eval = starkInfo.evMap.size();
    // Order polinomials by address, note that there are collisions!
    map<uintptr_t, vector<uint>> map_offsets;
    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = starkInfo.evMap[i];
        if (ev.type == EvMap::eType::_const)
        {
            map_offsets[reinterpret_cast<std::uintptr_t>(&((Goldilocks::Element *)pConstPols2ns->address())[ev.id])].push_back(i);
        }
        else if (ev.type == EvMap::eType::cm)
        {
            Polinomial pol = starkInfo.getPolinomial(mem, starkInfo.cm_2ns[ev.id]);
            map_offsets[reinterpret_cast<std::uintptr_t>(pol.address())].push_back(i);
        }
        else if (ev.type == EvMap::eType::q)
        {
            Polinomial pol = starkInfo.getPolinomial(mem, starkInfo.qs[ev.id]);
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
                ordPols[kk].potConstruct(&((Goldilocks::Element *)pConstPols2ns->address())[ev.id], pConstPols2ns->degree(), 1, pConstPols2ns->numPols());
            }
            else if (ev.type == EvMap::eType::cm)
            {
                ordPols[kk] = starkInfo.getPolinomial(mem, starkInfo.cm_2ns[ev.id]);
            }
            else if (ev.type == EvMap::eType::q)
            {
                ordPols[kk] = starkInfo.getPolinomial(mem, starkInfo.qs[ev.id]);
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
            Goldilocks::Element *LpEv_ = &(LpEv[k][0]);
            Goldilocks::Element *LEv_ = &(LEv[k][0]);
            for (uint64_t i = 0; i < size_eval; i++)
            {
                Polinomial::mulAddElement_adim3(&(evals_acc[thread_idx][i * FIELD_EXTENSION]), isPrime[i] ? LpEv_ : LEv_, ordPols[i], k << extendBits);
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
            (evals[indx[i]])[0] = evals_acc[0][offset];
            (evals[indx[i]])[1] = evals_acc[0][offset + 1];
            (evals[indx[i]])[2] = evals_acc[0][offset + 2];
        }
    }
    delete[] ordPols;
    for (int i = 0; i < num_threads; ++i)
    {
        free(evals_acc[i]);
    }
    free(evals_acc);
}

void Starks::merkelizeMemory()
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

void Starks::printPolRoot(uint64_t polId, StepsParams &params)
{
    Polinomial p = starkInfo.getPolinomial(params.pols, polId);

    Polinomial pCol;
    Goldilocks::Element *pBuffCol = new Goldilocks::Element[p.dim() * N];
    pCol.potConstruct(pBuffCol, p.degree(), p.dim(), p.dim());
    Polinomial::copy(pCol, p);

    MerkleTreeGL *mt_ = new MerkleTreeGL(N, p.dim(), pBuffCol);
    mt_->merkelize();
    
    Goldilocks::Element root[4];
    cout << "--------------------" << endl;
    cout << "ID: " << polId << endl;
    mt_->getRoot(&root[0]);
    cout <<  "MerkleTree rootGL : [ " << Goldilocks::toString(root[0]) << " " << Goldilocks::toString(root[1]) << " " << Goldilocks::toString(root[2]) << " " << Goldilocks::toString(root[3]) << " ]" << endl;

    cout << "--------------------" << endl;

    delete mt_;
    delete pBuffCol;
}
