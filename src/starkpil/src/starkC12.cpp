#include "starkC12.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include "ntt_goldilocks.hpp"
#include "fr.hpp"
#include "poseidon_opt.hpp"

#define NUM_CHALLENGES 8

StarkC12::StarkC12(const Config &config) : config(config),
                                           starkInfo(config,config.starkInfoC12File),
                                           zi(config.generateProof() ? starkInfo.starkStruct.nBits : 0,
                                              config.generateProof() ? starkInfo.starkStruct.nBitsExt : 0),
                                           numCommited(starkInfo.nCm1),
                                           N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                           NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                           ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                           x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0),
                                           x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0),
                                           challenges(config.generateProof() ? NUM_CHALLENGES : 0, config.generateProof() ? FIELD_EXTENSION : 0),
                                           xDivXSubXi(config.generateProof() ? NExtended : 0, config.generateProof() ? FIELD_EXTENSION : 0),
                                           xDivXSubWXi(config.generateProof() ? NExtended : 0, config.generateProof() ? FIELD_EXTENSION : 0),
                                           evals(config.generateProof() ? N : 0, config.generateProof() ? FIELD_EXTENSION : 0)

{
    // Avoid unnecessary initialization if we are not going to generate any proof
    if (!config.generateProof())
        return;

    // Allocate an area of memory, mapped to file, to read all the constant polynomials,
    // and create them using the allocated address
    TimerStart(LOAD_CONST_POLS_TO_MEMORY);
    pConstPolsAddress = NULL;
    if (config.constPolsC12File.size() == 0)
    {
        cerr << "Error: StarkC12::StarkC12() received an empty config.constPolsC12File" << endl;
        exit(-1);
    }

    if (config.mapConstPolsFile)
    {
        pConstPolsAddress = mapFile(config.constPolsC12File, ConstantPolsC12::pilSize(), false);
        cout << "StarkC12::StarkC12() successfully mapped " << ConstantPolsC12::pilSize() << " bytes from constant file " << config.constPolsC12File << endl;
    }
    else
    {
        pConstPolsAddress = copyFile(config.constPolsC12File, ConstantPolsC12::pilSize());
        cout << "StarkC12::StarkC12() successfully copied " << ConstantPolsC12::pilSize() << " bytes from constant file " << config.constPolsC12File << endl;
    }
    pConstPols = new ConstantPolsC12(pConstPolsAddress, ConstantPolsC12::pilDegree());
    TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

    // Map constants tree file to memory

    TimerStart(LOAD_CONST_TREE_TO_MEMORY);
    pConstTreeAddress = NULL;
    if (config.constantsTreeC12File.size() == 0)
    {
        cerr << "Error: StarkC12::StarkC12() received an empty config.constantsTreeC12File" << endl;
        exit(-1);
    }

    if (config.mapConstantsTreeFile)
    {
        pConstTreeAddress = mapFile(config.constantsTreeC12File, getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants), false);
        cout << "StarkC12::StarkC12() successfully mapped " << getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants) << " bytes from constant tree file " << config.constantsTreeC12File << endl;
    }
    else
    {
        pConstTreeAddress = copyFile(config.constantsTreeC12File, getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants));
        cout << "StarkC12::StarkC12() successfully copied " << getTreeSize((1 << starkInfo.starkStruct.nBitsExt), starkInfo.nConstants) << " bytes from constant file " << config.constantsTreeC12File << endl;
    }
    TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);

    // Initialize and allocate ConstantPols2ns
    TimerStart(LOAD_CONST_POLS_2NS_TO_MEMORY);
    pConstPolsAddress2ns = (void *)calloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt), sizeof(Goldilocks::Element));
    pConstPols2ns = new ConstantPolsC12(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt));
    std::memcpy(pConstPolsAddress2ns, (uint8_t *)pConstTreeAddress + 2 * sizeof(Goldilocks::Element), starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element));

    TimerStopAndLog(LOAD_CONST_POLS_2NS_TO_MEMORY);

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
}

StarkC12::~StarkC12()
{
    if (!config.generateProof())
        return;

    delete pConstPols;
    if (config.mapConstPolsFile)
    {
        unmapFile(pConstPolsAddress, ConstantPolsC12::pilSize());
    }
    else
    {
        free(pConstPolsAddress);
    }
}

void StarkC12::genProof(void *pAddress, FRIProofC12 &proof, Goldilocks::Element publicInputs[8])
{
    CommitPolsC12 cmPols(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);

    ///////////
    // 1.- Calculate p_cm1_2ns
    ///////////
    TimerStart(STARK_STEP_1);

    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

    TimerStart(STARK_STEP_1_LDE_AND_MERKLETREE);

    Goldilocks::Element *p_cm1_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns]];
    Goldilocks::Element *p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];
    TimerStart(STARK_STEP_1_LDE);
    ntt.extendPol(p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n]);
    TimerStopAndLog(STARK_STEP_1_LDE);
    TimerStart(STARK_STEP_1_MERKLETREE);

    MerkleTreeBN128 tree1(NExtended, starkInfo.mapSectionsN.section[eSection::cm1_n], p_cm1_2ns);
    RawFr::Element root1 = tree1.root();

    TimerStopAndLog(STARK_STEP_1_MERKLETREE);
    TimerStopAndLog(STARK_STEP_1_LDE_AND_MERKLETREE);
    std::cout << "MerkleTree root 1: [ " << RawFr::field.toString(root1, 10) << " ]" << std::endl;
    transcript.put(&root1, 1);
    TimerStopAndLog(STARK_STEP_1);

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    TimerStart(STARK_STEP_2);
    transcript.getField((uint64_t *)challenges[0]); // u
    transcript.getField((uint64_t *)challenges[1]); // defVal

    TimerStart(STARK_STEP_2_CALCULATE_EXPS);

    step2prev_first(mem, &publicInputs[0], 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step2prev_first(mem, &publicInputs[0], i);
        // CalculateExpsAll::step2prev_i(mem, const_n, (Goldilocks3::Element *)challenges.address(), i);
    }
    // CalculateExpsAll::step2prev_last(mem, const_n, (Goldilocks3::Element *)challenges.address(), N - 1);
    step2prev_first(mem, &publicInputs[0], N - 1);
    TimerStopAndLog(STARK_STEP_2_CALCULATE_EXPS);
    TimerStart(STARK_STEP_2_CALCULATEH1H2);
#pragma omp parallel for
    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial fPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]);
        Polinomial tPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]);
        Polinomial h1 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2]);
        Polinomial h2 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2 + 1]);

        Polinomial::calculateH1H2(h1, h2, fPol, tPol);
    }
    numCommited = numCommited + starkInfo.puCtx.size() * 2;
    TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2);

    TimerStart(STARK_STEP_2_LDE_AND_MERKLETREE);
    Goldilocks::Element *p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
    Goldilocks::Element *p_cm2_n = &mem[starkInfo.mapOffsets.section[eSection::cm2_n]];
    TimerStart(STARK_STEP_2_LDE);
    ntt.extendPol(p_cm2_2ns, p_cm2_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm2_n]);
    TimerStopAndLog(STARK_STEP_2_LDE);
    TimerStart(STARK_STEP_2_MERKLETREE);

    MerkleTreeBN128 tree2(NExtended, starkInfo.mapSectionsN1.section[eSection::cm2_n] + starkInfo.mapSectionsN3.section[eSection::cm2_n] * FIELD_EXTENSION, p_cm2_2ns);
    RawFr::Element root2 = tree2.root();

    TimerStopAndLog(STARK_STEP_2_MERKLETREE);
    TimerStopAndLog(STARK_STEP_2_LDE_AND_MERKLETREE);
    std::cout << "MerkleTree root 2: [ " << RawFr::field.toString(root2, 10) << " ]" << std::endl;
    transcript.put(&root2, 1);
    TimerStopAndLog(STARK_STEP_2);

    ///////////
    // 3.- Compute Z polynomials
    ///////////
    TimerStart(STARK_STEP_3);
    transcript.getField((uint64_t *)challenges[2]); // gamma
    transcript.getField((uint64_t *)challenges[3]); // betta

    TimerStart(STARK_STEP_3_CALCULATE_EXPS);
    step3prev_first(mem, &publicInputs[0], 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step3prev_first(mem, &publicInputs[0], i);
    }
    step3prev_first(mem, &publicInputs[0], N - 1);
    TimerStopAndLog(STARK_STEP_3_CALCULATE_EXPS);
    TimerStart(STARK_STEP_3_CALCULATE_Z);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].denId]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial::calculateZ(z, pNum, pDen);
    }

    for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].denId]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial::calculateZ(z, pNum, pDen);
    }

    for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].denId]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial::calculateZ(z, pNum, pDen);
    }
    TimerStopAndLog(STARK_STEP_3_CALCULATE_Z);

    TimerStart(STARK_STEP_3_LDE_AND_MERKLETREE);
    TimerStart(STARK_STEP_3_LDE);

    Goldilocks::Element *p_cm3_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
    Goldilocks::Element *p_cm3_n = &mem[starkInfo.mapOffsets.section[eSection::cm3_n]];
    ntt.extendPol(p_cm3_2ns, p_cm3_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm3_n]);
    TimerStopAndLog(STARK_STEP_3_LDE);
    TimerStart(STARK_STEP_3_MERKLETREE);
    MerkleTreeBN128 tree3(NExtended, starkInfo.mapSectionsN1.section[eSection::cm3_n] + starkInfo.mapSectionsN3.section[eSection::cm3_n] * FIELD_EXTENSION, p_cm3_2ns);
    RawFr::Element root3 = tree3.root();
    TimerStopAndLog(STARK_STEP_3_MERKLETREE);
    TimerStopAndLog(STARK_STEP_3_LDE_AND_MERKLETREE);
    std::cout << "MerkleTree root 3: [ " << RawFr::field.toString(root3, 10) << " ]" << std::endl;
    transcript.put(&root3, 1);
    TimerStopAndLog(STARK_STEP_3);

    ///////////
    // 4. Compute C Polynomial
    ///////////
    TimerStart(STARK_STEP_4);
    TimerStart(STARK_STEP_4_CALCULATE_EXPS);

    transcript.getField((uint64_t *)challenges[4]); // gamma
    step4_first(mem, &publicInputs[0], 0);

#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step4_first(mem, &publicInputs[0], i);
    }
    step4_first(mem, &publicInputs[0], N - 1);

    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS);
    TimerStart(STARK_STEP_4_LDE);
    Goldilocks::Element *p_exps_withq_2ns = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_2ns]];
    Goldilocks::Element *p_exps_withq_n = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_n]];
    ntt.extendPol(p_exps_withq_2ns, p_exps_withq_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::exps_withq_n]);
    TimerStopAndLog(STARK_STEP_4_LDE);
    TimerStart(STARK_STEP_4_CALCULATE_EXPS_2NS);
    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    uint64_t next = 1 << extendBits;

    for (uint64_t i = 0; i < next; i++)
    {
        step42ns_first(mem, &publicInputs[0], i);
    }
#pragma omp parallel for
    for (uint64_t i = next; i < NExtended - next; i++)
    {
        step42ns_first(mem, &publicInputs[0], i);
    }
    for (uint64_t i = NExtended - next; i < NExtended; i++)
    {
        step42ns_first(mem, &publicInputs[0], i);
    }
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS);
    TimerStart(STARK_STEP_4_MERKLETREE);
    Goldilocks::Element *p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];

    MerkleTreeBN128 tree4(NExtended, starkInfo.mapSectionsN.section[eSection::q_2ns], p_q_2ns);
    RawFr::Element root4 = tree4.root();
    TimerStopAndLog(STARK_STEP_4_MERKLETREE);
    std::cout << "MerkleTree root 4: [ " << RawFr::field.toString(root4, 10) << " ]" << std::endl;
    transcript.put(&root4, 1);
    TimerStopAndLog(STARK_STEP_4);

    ///////////
    // 5. Compute FRI Polynomial
    ///////////
    TimerStart(STARK_STEP_5);
    TimerStart(STARK_STEP_5_LEv_LpEv);

    transcript.getField((uint64_t *)challenges[5]); // v1
    transcript.getField((uint64_t *)challenges[6]); // v2
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
    TimerStopAndLog(STARK_STEP_5_LEv_LpEv);
    TimerStart(STARK_STEP_5_EVMAP);

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
    TimerStopAndLog(STARK_STEP_5_EVMAP);
    TimerStart(STARK_STEP_5_XDIVXSUB);

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

    Polinomial::batchInverse(xDivXSubXi, xDivXSubXi);
    Polinomial::batchInverse(xDivXSubWXi, xDivXSubWXi);

    Polinomial x1(1, FIELD_EXTENSION);
    *x1[0] = Goldilocks::shift();

    for (uint64_t k = 0; k < (N << extendBits); k++)
    {
        Polinomial::mulElement(xDivXSubXi, k, xDivXSubXi, k, x1, 0);
        Polinomial::mulElement(xDivXSubWXi, k, xDivXSubWXi, k, x1, 0);
        Polinomial::mulElement(x1, 0, x1, 0, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
    }
    TimerStopAndLog(STARK_STEP_5_XDIVXSUB);
    TimerStart(STARK_STEP_5_CALCULATE_EXPS);

    next = 1 << extendBits;

    for (uint64_t i = 0; i < next; i++)
    {
        step52ns_first(mem, publicInputs, i);
    }
#pragma omp parallel for
    for (uint64_t i = next; i < NExtended - next; i++)
    {
        step52ns_first(mem, publicInputs, i);
    }
    for (uint64_t i = NExtended - next; i < NExtended; i++)
    {
        step52ns_first(mem, publicInputs, i);
    }
    TimerStopAndLog(STARK_STEP_5_CALCULATE_EXPS);
    TimerStopAndLog(STARK_STEP_5);
    TimerStart(STARK_STEP_FRI);

    MerkleTreeBN128 constTree(pConstTreeAddress);
    trees[0] = &tree1;
    trees[1] = &tree2;
    trees[2] = &tree3;
    trees[3] = &tree4;
    trees[4] = &constTree;

    Polinomial friPol = starkInfo.getPolinomial(mem, starkInfo.exps_2ns[starkInfo.friExpId]);
    FRIProveC12::prove(proof, trees, transcript, friPol, starkInfo.starkStruct.nBitsExt, starkInfo);

    proof.proofs.setEvals(evals.address());

    std::memcpy(&proof.proofs.root1[0], &root1, sizeof(RawFr::Element));
    std::memcpy(&proof.proofs.root2[0], &root2, sizeof(RawFr::Element));
    std::memcpy(&proof.proofs.root3[0], &root3, sizeof(RawFr::Element));
    std::memcpy(&proof.proofs.root4[0], &root4, sizeof(RawFr::Element));

    TimerStopAndLog(STARK_STEP_FRI);
}
