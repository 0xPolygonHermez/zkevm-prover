#include "starkRecursiveF.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include "ntt_goldilocks.hpp"
#include "fr.hpp"
#include "poseidon_opt.hpp"
#include "starkRecursiveFSteps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

#define NUM_CHALLENGES 8

StarkRecursiveF::StarkRecursiveF(const Config &config, void *_pAddress) : config(config),
                                                                          starkInfo(config, config.recursivefStarkInfo),
                                                                          zi(config.generateProof() ? starkInfo.starkStruct.nBits : 0,
                                                                             config.generateProof() ? starkInfo.starkStruct.nBitsExt : 0),
                                                                          N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                          NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                          ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                                          nttExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                                          x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0),
                                                                          x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0),
                                                                          pAddress(_pAddress)
{
    // Avoid unnecessary initialization if we are not going to generate any proof
    if (!config.generateProof())
        return;

    // Allocate an area of memory, mapped to file, to read all the constant polynomials,
    // and create them using the allocated address
    TimerStart(LOAD_RECURSIVE_F_CONST_POLS_TO_MEMORY);
    pConstPolsAddress = NULL;
    if (config.recursivefConstPols.size() == 0)
    {
        zklog.error("StarkRecursiveF::StarkRecursiveF() received an empty config.recursivefConstPols");
        exitProcess();
    }
    constPolsDegree = (1 << starkInfo.starkStruct.nBits);
    constPolsSize = starkInfo.nConstants * sizeof(Goldilocks::Element) * constPolsDegree;

    if (config.mapConstPolsFile)
    {
        pConstPolsAddress = mapFile(config.recursivefConstPols, constPolsSize, false);
        zklog.info("StarkRecursiveF::StarkRecursiveF() successfully mapped " + to_string(constPolsSize) + " bytes from constant file " + config.recursivefConstPols);
    }
    else
    {
        pConstPolsAddress = copyFile(config.recursivefConstPols, constPolsSize);
        zklog.info("StarkRecursiveF::StarkRecursiveF() successfully copied " + to_string(constPolsSize) + " bytes from constant file " + config.recursivefConstPols);
    }
    pConstPols = new ConstantPolsStarks(pConstPolsAddress, constPolsDegree, starkInfo.nConstants);
    TimerStopAndLog(LOAD_RECURSIVE_F_CONST_POLS_TO_MEMORY);

    // Map constants tree file to memory

    TimerStart(LOAD_RECURSIVE_F_CONST_TREE_TO_MEMORY);
    pConstTreeAddress = NULL;
    if (config.recursivefConstantsTree.size() == 0)
    {
        zklog.error("StarkRecursiveF::StarkRecursiveF() received an empty config.recursivefConstantsTree");
        exitProcess();
    }

    if (config.mapConstantsTreeFile)
    {
        pConstTreeAddress = mapFile(config.recursivefConstantsTree, getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants), false);
        zklog.info("StarkRecursiveF::StarkRecursiveF() successfully mapped " + to_string(getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants)) + " bytes from constant tree file " + config.recursivefConstantsTree);
    }
    else
    {
        pConstTreeAddress = copyFile(config.recursivefConstantsTree, getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants));
        zklog.info("StarkRecursiveF::StarkRecursiveF() successfully copied " + to_string(getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants)) + " bytes from constant file " + config.recursivefConstantsTree);
    }
    TimerStopAndLog(LOAD_RECURSIVE_F_CONST_TREE_TO_MEMORY);

    // Initialize and allocate ConstantPols2ns
    TimerStart(LOAD_RECURSIVE_F_CONST_POLS_2NS_TO_MEMORY);
    pConstPolsAddress2ns = (void *)calloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt), sizeof(Goldilocks::Element));
    pConstPols2ns = new ConstantPolsStarks(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants);
    std::memcpy(pConstPolsAddress2ns, (uint8_t *)pConstTreeAddress + 2 * sizeof(Goldilocks::Element), starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element));

    TimerStopAndLog(LOAD_RECURSIVE_F_CONST_POLS_2NS_TO_MEMORY);

    // TODO x_n and x_2ns could be precomputed
    TimerStart(COMPUTE_X_N_AND_X_2_NS);
    Goldilocks::Element xx = Goldilocks::one();
    for (uint i = 0; i < N; i++)
    {
        *x_n[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBits));
    }
    xx = Goldilocks::shift();
    for (uint i = 0; i < NExtended; i++)
    {
        *x_2ns[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBitsExt));
    }
    TimerStopAndLog(COMPUTE_X_N_AND_X_2_NS);

    mem = (Goldilocks::Element *)_pAddress;
    pBuffer = (Goldilocks::Element *)malloc(starkInfo.mapSectionsN.section[eSection::cm1_n] * NExtended * FIELD_EXTENSION * sizeof(Goldilocks::Element));

    p_cm1_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns]];
    p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];
    p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
    p_cm2_n = &mem[starkInfo.mapOffsets.section[eSection::cm2_n]];
    p_cm3_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
    p_cm3_n = &mem[starkInfo.mapOffsets.section[eSection::cm3_n]];
    cm4_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm4_2ns]];
    p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];
    p_f_2ns = &mem[starkInfo.mapOffsets.section[eSection::f_2ns]];
}

StarkRecursiveF::~StarkRecursiveF()
{
    if (!config.generateProof())
        return;

    delete pConstPols;
    delete pConstPols2ns;
    free(pConstPolsAddress2ns);

    if (config.mapConstPolsFile)
    {
        unmapFile(pConstPolsAddress, constPolsSize);
    }
    else
    {
        free(pConstPolsAddress);
    }

    if (config.mapConstantsTreeFile)
    {
        unmapFile(pConstTreeAddress, getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants));
    }
    else
    {
        free(pConstTreeAddress);
    }

    free(pBuffer);
}

void StarkRecursiveF::genProof(FRIProofC12 &proof, Goldilocks::Element publicInputs[8])
{

    StarkRecursiveFSteps recurisveFsteps;
    StarkRecursiveFSteps *steps = &recurisveFsteps;
    // Initialize vars
    uint64_t numCommited = starkInfo.nCm1;
    TranscriptBN128 transcript;
    Polinomial evals(N, FIELD_EXTENSION);
    Polinomial xDivXSubXi(NExtended, FIELD_EXTENSION);
    Polinomial xDivXSubWXi(NExtended, FIELD_EXTENSION);
    Polinomial challenges(NUM_CHALLENGES, FIELD_EXTENSION);

    CommitPolsStarks cmPols(pAddress, starkInfo.mapDeg.section[eSection::cm1_n], numCommited);

    RawFr::Element rootC;
    RawFr::Element root0;
    RawFr::Element root1;
    RawFr::Element root2;
    RawFr::Element root3;

    MerkleTreeBN128 *treesBN128[STARK_RECURSIVE_F_NUM_TREES];
    treesBN128[0] = new MerkleTreeBN128(NExtended, starkInfo.mapSectionsN.section[eSection::cm1_n], p_cm1_2ns);
    treesBN128[1] = new MerkleTreeBN128(NExtended, starkInfo.mapSectionsN.section[eSection::cm2_n], p_cm2_2ns);
    treesBN128[2] = new MerkleTreeBN128(NExtended, starkInfo.mapSectionsN.section[eSection::cm3_n], p_cm3_2ns);
    treesBN128[3] = new MerkleTreeBN128(NExtended, starkInfo.mapSectionsN.section[eSection::cm4_2ns], cm4_2ns);
    treesBN128[4] = new MerkleTreeBN128(pConstTreeAddress);

    treesBN128[4]->getRoot(&rootC);

    transcript.put(&rootC, 1);
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
        xDivXSubWXi : xDivXSubWXi,
        publicInputs : publicInputs,
        q_2ns : p_q_2ns,
        f_2ns : p_f_2ns
    };
    //--------------------------------
    // 1.- Calculate p_cm1_2ns
    //--------------------------------
    TimerStart(STARK_RECURSIVE_F_STEP_1);
    TimerStart(STARK_RECURSIVE_F_STEP_1_LDE_AND_MERKLETREE);
    TimerStart(STARK_RECURSIVE_F_STEP_1_LDE);

    ntt.extendPol(p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n], pBuffer);

    treesBN128[0]->merkelize();
    treesBN128[0]->getRoot(&root0);
    zklog.info("MerkleTree root 0: [ " + RawFr::field.toString(root0, 10) + " ]");
    transcript.put(&root0, 1);

    TimerStopAndLog(STARK_RECURSIVE_F_STEP_1_LDE);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_1_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_1);

    //--------------------------------
    // 2.- Caluculate plookups h1 and h2
    //--------------------------------
    TimerStart(STARK_RECURSIVE_F_STEP_2);
    transcript.getField((uint64_t *)challenges[0]); // u
    transcript.getField((uint64_t *)challenges[1]); // defVal
    TimerStart(STARK_RECURSIVE_F_STEP_2_CALCULATE_EXPS);

#pragma omp parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        steps->step2prev_first(params, i);
    }
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_2_CALCULATE_EXPS);

    TimerStart(STARK_RECURSIVE_F_STEP_2_CALCULATEH1H2);
#pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial fPol = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].fExpId)]);
        Polinomial tPol = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].tExpId)]);
        Polinomial h1 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial h2 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);

        Polinomial::calculateH1H2(h1, h2, fPol, tPol);
    }
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_2_CALCULATEH1H2);

    TimerStart(STARK_RECURSIVE_F_STEP_2_LDE_AND_MERKLETREE);

    ntt.extendPol(p_cm2_2ns, p_cm2_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm2_n], pBuffer);

    treesBN128[1]->merkelize();
    treesBN128[1]->getRoot(&root1);
    zklog.info("MerkleTree root 1: [ " + RawFr::field.toString(root1, 10) + " ]");
    transcript.put(&root1, 1);

    TimerStopAndLog(STARK_RECURSIVE_F_STEP_2_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_2);

    //--------------------------------
    // 3.- Compute Z polynomials
    //--------------------------------
    TimerStart(STARK_RECURSIVE_F_STEP_3);
    transcript.getField((uint64_t *)challenges[2]); // gamma
    transcript.getField((uint64_t *)challenges[3]); // betta

    TimerStart(STARK_RECURSIVE_F_STEP_3_PREV_CALCULATE_EXPS);
#pragma omp parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        steps->step3prev_first(params, i);
    }
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_3_PREV_CALCULATE_EXPS);
    TimerStart(STARK_RECURSIVE_F_STEP_3_CALCULATE_Z);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].numId)]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.puCtx[i].denId)]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial::calculateZ(z, pNum, pDen);
    }

    for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.peCtx[i].numId)]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.peCtx[i].denId)]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial::calculateZ(z, pNum, pDen);
    }

    for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.ciCtx[i].numId)]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exp2pol[to_string(starkInfo.ciCtx[i].denId)]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial::calculateZ(z, pNum, pDen);
    }
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_3_CALCULATE_Z);

    TimerStart(STARK_RECURSIVE_F_STEP_3_CALCULATE_EXPS);
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        steps->step3_first(params, i);
    }
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_3_CALCULATE_EXPS);

    TimerStart(STARK_RECURSIVE_F_STEP_3_LDE_AND_MERKLETREE);

    ntt.extendPol(p_cm3_2ns, p_cm3_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm3_n], pBuffer);

    treesBN128[2]->merkelize();
    treesBN128[2]->getRoot(&root2);
    zklog.info("MerkleTree root 2: [ " + RawFr::field.toString(root2, 10) + " ]");
    transcript.put(&root2, 1);

    TimerStopAndLog(STARK_RECURSIVE_F_STEP_3_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_3);

    //--------------------------------
    // 4. Compute C Polynomial
    //--------------------------------
    TimerStart(STARK_RECURSIVE_F_STEP_4);
    TimerStart(STARK_RECURSIVE_F_STEP_4_CALCULATE_EXPS);

    transcript.getField((uint64_t *)challenges[4]); // gamma

    TimerStopAndLog(STARK_RECURSIVE_F_STEP_4_CALCULATE_EXPS);

    TimerStart(STARK_RECURSIVE_F_STEP_4_CALCULATE_EXPS_2NS);
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;

#pragma omp parallel for
    for (uint64_t i = 0; i < NExtended; i++)
    {
        steps->step42ns_first(params, i);
    }

    Polinomial qq1 = Polinomial(NExtended, starkInfo.qDim, "qq1");
    Polinomial qq2 = Polinomial(NExtended * starkInfo.qDeg, starkInfo.qDim, "qq2");

    nttExtended.INTT(qq1.address(), p_q_2ns, NExtended, starkInfo.qDim, NULL, 2, 1);

    Goldilocks::Element curS = Goldilocks::one();
    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);

    for (uint64_t p = 0; p < starkInfo.qDeg; p++)
    {
        for (uint64_t i = 0; i < N; i++)
        {
            Goldilocks3::mul((Goldilocks3::Element &)*qq2[i * starkInfo.qDeg + p], (Goldilocks3::Element &)*qq1[p * N + i], curS);
        }
        curS = Goldilocks::mul(curS, shiftIn);
    }

    nttExtended.NTT(cm4_2ns, qq2.address(), NExtended, starkInfo.qDim * starkInfo.qDeg);

    TimerStopAndLog(STARK_RECURSIVE_F_STEP_4_CALCULATE_EXPS_2NS);
    TimerStart(STARK_RECURSIVE_F_STEP_4_MERKLETREE);

    treesBN128[3]->merkelize();
    treesBN128[3]->getRoot(&root3);
    zklog.info("MerkleTree root 3: [ " + RawFr::field.toString(root3, 10) + " ]");
    transcript.put(&root3, 1);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_4_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_4);

    //--------------------------------
    // 5. Compute FRI Polynomial
    //--------------------------------
    TimerStart(STARK_RECURSIVE_F_STEP_5);
    TimerStart(STARK_RECURSIVE_F_STEP_5_LEv_LpEv);

    // transcript.getField((uint64_t *)challenges[5]); // v1
    // transcript.getField((uint64_t *)challenges[6]); // v2
    transcript.getField((uint64_t *)challenges[7]); // xi

    Polinomial LEv(N, 3, "LEv");
    Polinomial LpEv(N, 3, "LpEv");
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
    ntt.INTT(LEv.address(), LEv.address(), N, 3);
    ntt.INTT(LpEv.address(), LpEv.address(), N, 3);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_5_LEv_LpEv);

#if 0
#pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.evMap.size(); i++)
    {
        EvMap ev = starkInfo.evMap[i];

        Polinomial acc(1, FIELD_EXTENSION);
        Polinomial tmp(1, FIELD_EXTENSION);
        if (ev.type == EvMap::eType::_const)
        {
            Polinomial p(&((Goldilocks::Element *)pConstPols2ns->address())[ev.id], pConstPols2ns->degree(), 1, pConstPols2ns->numPols());
            for (uint64_t k = 0; k < N; k++)
            {
                Polinomial::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                Polinomial::addElement(acc, 0, acc, 0, tmp, 0);
            }
        }
        else if (ev.type == EvMap::eType::cm || ev.type == EvMap::eType::q)
        {
            Polinomial p;
            p = (ev.type == EvMap::eType::cm) ? starkInfo.getPolinomial(mem, starkInfo.cm_2ns[ev.id]) : starkInfo.getPolinomial(mem, starkInfo.qs[ev.id]);
            for (uint64_t k = 0; k < N; k++)
            {
                Polinomial::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                Polinomial::addElement(acc, 0, acc, 0, tmp, 0);
            }
        }
        else
        {
            throw std::invalid_argument("Invalid ev type: " + ev.type);
        }

        Polinomial::copyElement(evals, i, acc, 0);
    }
#else
    /* sort polinomials depending on its type

            Subsets:
                0. const
                1. cm , dim=1
                2. qs , dim=1  //1 and 2 to be joined
                3. cm , dim=3
                4. qs, dim=3   //3 and 4 to be joined
         */

    u_int64_t size_eval = starkInfo.evMap.size();
    u_int64_t *sorted_evMap = (u_int64_t *)malloc(5 * size_eval * sizeof(u_int64_t));
    u_int64_t counters[5] = {0, 0, 0, 0, 0};

    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = starkInfo.evMap[i];
        if (ev.type == EvMap::eType::_const)
        {
            sorted_evMap[counters[0]] = i;
            ++counters[0];
        }
        else if (ev.type == EvMap::eType::cm)
        {
            uint16_t idPol = (ev.type == EvMap::eType::cm) ? starkInfo.cm_2ns[ev.id] : starkInfo.qs[ev.id];
            VarPolMap polInfo = starkInfo.varPolMap[idPol];
            uint64_t dim = polInfo.dim;
            if (dim == 1)
            {
                sorted_evMap[size_eval + counters[1]] = i;
                ++counters[1];
            }
            else
            {
                sorted_evMap[3 * size_eval + counters[3]] = i;
                ++counters[3];
            }
        }
        else if (ev.type == EvMap::eType::q)
        {
            uint16_t idPol = (ev.type == EvMap::eType::cm) ? starkInfo.cm_2ns[ev.id] : starkInfo.qs[ev.id];
            VarPolMap polInfo = starkInfo.varPolMap[idPol];
            uint64_t dim = polInfo.dim;
            if (dim == 1)
            {
                sorted_evMap[2 * size_eval + counters[2]] = i;
                ++counters[2];
            }
            else
            {
                sorted_evMap[4 * size_eval + counters[4]] = i;
                ++counters[4];
            }
        }
        else
        {
            throw std::invalid_argument("Invalid ev type: " + ev.type);
        }
    }
    // join subsets 1 and 2 in 1
    int offset1 = size_eval + counters[1];
    int offset2 = 2 * size_eval;
    for (uint64_t i = 0; i < counters[2]; ++i)
    {
        sorted_evMap[offset1 + i] = sorted_evMap[offset2 + i];
        ++counters[1];
    }
    // join subsets 3 and 4 in 3
    offset1 = 3 * size_eval + counters[3];
    offset2 = 4 * size_eval;
    for (uint64_t i = 0; i < counters[4]; ++i)
    {
        sorted_evMap[offset1 + i] = sorted_evMap[offset2 + i];
        ++counters[3];
    }
    // Buffer for partial results of the matrix-vector product (columns distribution)
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
            for (uint64_t i = 0; i < counters[0]; i++)
            {
                int indx = sorted_evMap[i];
                EvMap ev = starkInfo.evMap[indx];
                Polinomial tmp(1, FIELD_EXTENSION);
                Polinomial acc(1, FIELD_EXTENSION);

                Polinomial p(&((Goldilocks::Element *)pConstPols2ns->address())[ev.id], pConstPols2ns->degree(), 1, pConstPols2ns->numPols());

                Polinomial::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                for (int j = 0; j < FIELD_EXTENSION; ++j)
                {
                    evals_acc[thread_idx][indx * FIELD_EXTENSION + j] = evals_acc[thread_idx][indx * FIELD_EXTENSION + j] + tmp[0][j];
                }
            }
            for (uint64_t i = 0; i < counters[1]; i++)
            {
                int indx = sorted_evMap[size_eval + i];
                EvMap ev = starkInfo.evMap[indx];
                Polinomial tmp(1, FIELD_EXTENSION);

                Polinomial p;
                p = (ev.type == EvMap::eType::cm) ? starkInfo.getPolinomial(mem, starkInfo.cm_2ns[ev.id]) : starkInfo.getPolinomial(mem, starkInfo.qs[ev.id]);

                Polinomial ::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                for (int j = 0; j < FIELD_EXTENSION; ++j)
                {
                    evals_acc[thread_idx][indx * FIELD_EXTENSION + j] = evals_acc[thread_idx][indx * FIELD_EXTENSION + j] + tmp[0][j];
                }
            }
            for (uint64_t i = 0; i < counters[3]; i++)
            {
                int indx = sorted_evMap[3 * size_eval + i];
                EvMap ev = starkInfo.evMap[indx];
                Polinomial tmp(1, FIELD_EXTENSION);

                Polinomial p;
                p = (ev.type == EvMap::eType::cm) ? starkInfo.getPolinomial(mem, starkInfo.cm_2ns[ev.id]) : starkInfo.getPolinomial(mem, starkInfo.qs[ev.id]);

                Polinomial ::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                for (int j = 0; j < FIELD_EXTENSION; ++j)
                {
                    evals_acc[thread_idx][indx * FIELD_EXTENSION + j] = evals_acc[thread_idx][indx * FIELD_EXTENSION + j] + tmp[0][j];
                }
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
            (evals[i])[0] = sum0;
            (evals[i])[1] = sum1;
            (evals[i])[2] = sum2;
        }
    }
    free(sorted_evMap);
    for (int i = 0; i < num_threads; ++i)
    {
        free(evals_acc[i]);
    }
    free(evals_acc);
#endif

    for (uint64_t i = 0; i < starkInfo.evMap.size(); i++)
    {
        transcript.put(evals[i], 3);
    }

    transcript.getField((uint64_t *)challenges[5]); // v1
    transcript.getField((uint64_t *)challenges[6]); // v2

    // Calculate xDivXSubXi, xDivXSubWXi
    Polinomial xi(1, FIELD_EXTENSION);
    Polinomial wxi(1, FIELD_EXTENSION);

    Polinomial::copyElement(xi, 0, challenges, 7);
    Polinomial::mulElement(wxi, 0, challenges, 7, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));

    Polinomial x(1, FIELD_EXTENSION);
    *x[0] = Goldilocks::shift();

    for (uint64_t k = 0; k < (N << extendBits); k++)
    {
        Polinomial::subElement(xDivXSubXi, k, x, 0, xi, 0);
        Polinomial::subElement(xDivXSubWXi, k, x, 0, wxi, 0);
        Polinomial::mulElement(x, 0, x, 0, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
    }

    Polinomial::batchInverseParallel(xDivXSubXi, xDivXSubXi);
    Polinomial::batchInverseParallel(xDivXSubWXi, xDivXSubWXi);

    Polinomial x1(1, FIELD_EXTENSION);
    *x1[0] = Goldilocks::shift();

    for (uint64_t k = 0; k < (N << extendBits); k++)
    {
        Polinomial::mulElement(xDivXSubXi, k, xDivXSubXi, k, x1, 0);
        Polinomial::mulElement(xDivXSubWXi, k, xDivXSubWXi, k, x1, 0);
        Polinomial::mulElement(x1, 0, x1, 0, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
    }
    TimerStart(STARK_RECURSIVE_F_STEP_5_CALCULATE_EXPS);

#pragma omp parallel for
    for (uint64_t i = 0; i < NExtended; i++)
    {
        steps->step52ns_first(params, i);
    }

    TimerStopAndLog(STARK_RECURSIVE_F_STEP_5_CALCULATE_EXPS);
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_5);
    TimerStart(STARK_RECURSIVE_F_STEP_FRI);

    Polinomial friPol = Polinomial(p_f_2ns, NExtended, 3, 3, "friPol");
    FRIProveC12::prove(proof, treesBN128, transcript, friPol, starkInfo.starkStruct.nBitsExt, starkInfo);

    proof.proofs.setEvals(evals.address());

    std::memcpy(&proof.proofs.root1[0], &root0, sizeof(RawFr::Element));
    std::memcpy(&proof.proofs.root2[0], &root1, sizeof(RawFr::Element));
    std::memcpy(&proof.proofs.root3[0], &root2, sizeof(RawFr::Element));
    std::memcpy(&proof.proofs.root4[0], &root3, sizeof(RawFr::Element));
    for (uint i = 0; i < 5; i++)
    {
        delete treesBN128[i];
    }
    TimerStopAndLog(STARK_RECURSIVE_F_STEP_FRI);
}
