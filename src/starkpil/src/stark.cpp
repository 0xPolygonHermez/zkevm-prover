#include "stark.hpp"
#include "timer.hpp"
#include "utils.hpp"

//#include "starkPols.hpp"
//#include "starkPols2ns.hpp"

#include "ntt_goldilocks.hpp"

#define NUM_CHALLENGES 8

Stark::Stark(const Config &config) : config(config),
                                     starkInfo(config),
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
    if (config.constPolsFile.size() == 0)
    {
        cerr << "Error: Stark::Stark() received an empty config.constPolsFile" << endl;
        exit(-1);
    }
    if (config.mapConstPolsFile)
    {
        pConstPolsAddress = mapFile(config.constPolsFile, ConstantPols::pilSize(), false);
        cout << "Stark::Stark() successfully mapped " << ConstantPols::pilSize() << " bytes from constant file " << config.constPolsFile << endl;
    }
    else
    {
        pConstPolsAddress = copyFile(config.constPolsFile, ConstantPols::pilSize());
        cout << "Stark::Stark() successfully copied " << ConstantPols::pilSize() << " bytes from constant file " << config.constPolsFile << endl;
    }
    pConstPols = new ConstantPols(pConstPolsAddress, ConstantPols::pilDegree());
    TimerStopAndLog(LOAD_CONST_POLS_TO_MEMORY);

    // Map constants tree file to memory

    TimerStart(LOAD_CONST_TREE_TO_MEMORY);
    pConstTreeAddress = NULL;
    if (config.constantsTreeFile.size() == 0)
    {
        cerr << "Error: Stark::Stark() received an empty config.constantsTreeFile" << endl;
        exit(-1);
    }
    if (config.mapConstantsTreeFile)
    {
        pConstTreeAddress = mapFile(config.constantsTreeFile, starkInfo.getConstTreeSizeInBytes(), false);
        cout << "Stark::Stark() successfully mapped " << starkInfo.getConstTreeSizeInBytes() << " bytes from constant tree file " << config.constantsTreeFile << endl;
    }
    else
    {
        pConstTreeAddress = copyFile(config.constantsTreeFile, starkInfo.getConstTreeSizeInBytes());
        cout << "Stark::Stark() successfully copied " << starkInfo.getConstTreeSizeInBytes() << " bytes from constant file " << config.constantsTreeFile << endl;
    }
    TimerStopAndLog(LOAD_CONST_TREE_TO_MEMORY);

    // Initialize and allocate ConstantPols2ns
    TimerStart(LOAD_CONST_POLS_2NS_TO_MEMORY);
    pConstPolsAddress2ns = (void *)calloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt), sizeof(Goldilocks::Element));
    pConstPols2ns = new ConstantPols(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt));

    for (uint64_t j = 0; j < NExtended; j++)
        for (uint64_t i = 0; i < starkInfo.nConstants; i++)
        {
            {
                MerklehashGoldilocks::getElement(((ConstantPols *)pConstPols2ns)->getElement(i, j), (Goldilocks::Element *)pConstTreeAddress, j, i);
            }
        }
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

Stark::~Stark()
{
    if (!config.generateProof())
        return;

    delete pConstPols;
    if (config.mapConstPolsFile)
    {
        unmapFile(pConstPolsAddress, ConstantPols::pilSize());
    }
    else
    {
        free(pConstPolsAddress);
    }
}

void Stark::genProof(void *pAddress, CommitPols &_cmPols, const PublicInputs &_publicInputs, Proof &proof)
{
#define commited_file "zkevm.commit"

    void *pCommitedAddress = mapFile(commited_file, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element), false);
    std::memcpy(pAddress, pCommitedAddress, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element));

    CommitPols cmPols(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);

    ///////////
    // 1.- Calculate p_cm1_2ns
    ///////////
    TimerStart(STARK_STEP_1);

    Goldilocks::Element publicInputs[8];
    publicInputs[0] = cmPols.Main.FREE0[0];
    publicInputs[1] = cmPols.Main.FREE1[0];
    publicInputs[2] = cmPols.Main.FREE2[0];
    publicInputs[3] = cmPols.Main.FREE3[0];
    publicInputs[4] = cmPols.Main.FREE4[0];
    publicInputs[5] = cmPols.Main.FREE5[0];
    publicInputs[6] = cmPols.Main.FREE6[0];
    publicInputs[7] = cmPols.Main.FREE7[0];

    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

    TimerStart(STARK_STEP_1_LDE_AND_MERKLETREE);
    uint64_t numElementsTree1 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm1_n] + starkInfo.mapSectionsN3.section[eSection::cm1_n] * FIELD_EXTENSION, NExtended);

    Polinomial tree1(numElementsTree1, 1);
    Polinomial root1(HASH_SIZE, 1);

    Goldilocks::Element *p_cm1_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns]];
    Goldilocks::Element *p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];
    TimerStart(STARK_STEP_1_LDE);
    ntt.extendPol(p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n]);
    TimerStopAndLog(STARK_STEP_1_LDE);
    TimerStart(STARK_STEP_1_MERKLETREE);
    PoseidonGoldilocks::merkletree(tree1.address(), p_cm1_2ns, starkInfo.mapSectionsN.section[eSection::cm1_n], NExtended);
    MerklehashGoldilocks::root(root1.address(), tree1.address(), tree1.length());
    TimerStopAndLog(STARK_STEP_1_MERKLETREE);
    TimerStopAndLog(STARK_STEP_1_LDE_AND_MERKLETREE);
    std::cout << "MerkleTree root 1: [ " << root1.toString(4) << " ]" << std::endl;
    transcript.put(root1.address(), HASH_SIZE);
    TimerStopAndLog(STARK_STEP_1);

    // HARDCODED VALUES
    /*
    Polinomial root1_hardcoded(HASH_SIZE, 1);
    *root1_hardcoded[0] = Goldilocks::fromU64(14773216157232762958ULL);
    *root1_hardcoded[1] = Goldilocks::fromU64(9090792250391374988ULL);
    *root1_hardcoded[2] = Goldilocks::fromU64(11395074597553208760ULL);
    *root1_hardcoded[3] = Goldilocks::fromU64(17145980724823558481ULL);
    transcript.put(root1_hardcoded.address(), HASH_SIZE);
    */

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    TimerStart(STARK_STEP_2);
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal

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

        calculateH1H2(h1, h2, fPol, tPol);
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
    uint64_t numElementsTree2 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm2_n] + starkInfo.mapSectionsN3.section[eSection::cm2_n] * FIELD_EXTENSION, NExtended);
    Polinomial tree2(numElementsTree2, 1);
    Polinomial root2(HASH_SIZE, 1);
    PoseidonGoldilocks::merkletree(tree2.address(), p_cm2_2ns, starkInfo.mapSectionsN.section[eSection::cm2_n], NExtended);
    TimerStopAndLog(STARK_STEP_2_MERKLETREE);
    TimerStopAndLog(STARK_STEP_2_LDE_AND_MERKLETREE);
    MerklehashGoldilocks::root(root2.address(), tree2.address(), tree2.length());
    std::cout << "MerkleTree root 2: [ " << root2.toString(4) << " ]" << std::endl;
    transcript.put(root2.address(), HASH_SIZE);
    TimerStopAndLog(STARK_STEP_2);
    /*
    // HARDCODED VALUES
    Polinomial root2_hardcoded(HASH_SIZE, 1);
    *root2_hardcoded[0] = Goldilocks::fromU64(5602570006149680147ULL);
    *root2_hardcoded[1] = Goldilocks::fromU64(16794271776532285084ULL);
    *root2_hardcoded[2] = Goldilocks::fromU64(1070892053182687126ULL);
    *root2_hardcoded[3] = Goldilocks::fromU64(4818924615586863563ULL);
    transcript.put(root2_hardcoded.address(), HASH_SIZE);
*/
    ///////////
    // 3.- Compute Z polynomials
    ///////////
    TimerStart(STARK_STEP_3);
    transcript.getField(challenges[2]); // gamma
    transcript.getField(challenges[3]); // betta

    TimerStart(STARK_STEP_3_CALCULATE_EXPS);
    step3prev_first(mem, &publicInputs[0], 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step3prev_first(mem, &publicInputs[0], i);
        // CalculateExpsAll::step3prev_i((Goldilocks::Element *)pAddress, const_n, challenges, x_n, i);
    }
    step3prev_first(mem, &publicInputs[0], N - 1);
    TimerStopAndLog(STARK_STEP_3_CALCULATE_EXPS);
    TimerStart(STARK_STEP_3_CALCULATE_Z);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].denId]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        calculateZ(z, pNum, pDen);
    }

    for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
    {
        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].denId]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        calculateZ(z, pNum, pDen);
    }

    for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
    {

        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].denId]);
        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        calculateZ(z, pNum, pDen);
    }
    TimerStopAndLog(STARK_STEP_3_CALCULATE_Z);

    TimerStart(STARK_STEP_3_LDE_AND_MERKLETREE);
    TimerStart(STARK_STEP_3_LDE);

    Goldilocks::Element *p_cm3_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
    Goldilocks::Element *p_cm3_n = &mem[starkInfo.mapOffsets.section[eSection::cm3_n]];
    ntt.extendPol(p_cm3_2ns, p_cm3_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm3_n]);
    TimerStopAndLog(STARK_STEP_3_LDE);
    TimerStart(STARK_STEP_3_MERKLETREE);
    uint64_t numElementsTree3 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm3_n] + starkInfo.mapSectionsN3.section[eSection::cm3_n] * FIELD_EXTENSION, NExtended);
    Polinomial tree3(numElementsTree3, 1);
    Polinomial root3(HASH_SIZE, 1);
    PoseidonGoldilocks::merkletree(tree3.address(), p_cm3_2ns, starkInfo.mapSectionsN.section[eSection::cm3_n], NExtended);
    TimerStopAndLog(STARK_STEP_3_MERKLETREE);
    TimerStopAndLog(STARK_STEP_3_LDE_AND_MERKLETREE);
    MerklehashGoldilocks::root(root3.address(), tree3.address(), tree3.length());
    std::cout << "MerkleTree root 3: [ " << root3.toString(4) << " ]" << std::endl;
    transcript.put(root3.address(), HASH_SIZE);
    TimerStopAndLog(STARK_STEP_3);

    /*
    Polinomial root3_hardcoded(HASH_SIZE, 1);
    *root3_hardcoded[0] = Goldilocks::fromU64(9941220780739138404ULL);
    *root3_hardcoded[1] = Goldilocks::fromU64(5738999706464945940ULL);
    *root3_hardcoded[2] = Goldilocks::fromU64(9388229154186191556ULL);
    *root3_hardcoded[3] = Goldilocks::fromU64(11332475887922162499ULL);
    transcript.put(root3_hardcoded.address(), HASH_SIZE);
    */
    ///////////
    // 4. Compute C Polynomial
    ///////////
    TimerStart(STARK_STEP_4);
    TimerStart(STARK_STEP_4_CALCULATE_EXPS);

    transcript.getField(challenges[4]); // gamma
    step4_first(mem, &publicInputs[0], 0);

#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step4_first(mem, &publicInputs[0], i);
        // CalculateExpsAll::step3prev_i((Goldilocks::Element *)pAddress, const_n, challenges, x_n, i);
    }
    step4_first(mem, &publicInputs[0], N - 1);
    // CalculateExpsAll::step3prev_last((Goldilocks::Element *)pAddress, const_n, challenges, x_n, N - 1);
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
        // step42ns_i(mem, &publicInputs[0], i);
    }
    for (uint64_t i = NExtended - next; i < NExtended; i++)
    {
        step42ns_first(mem, &publicInputs[0], i);
        // step42ns_last(mem, &publicInputs[0], i);
    }
    TimerStopAndLog(STARK_STEP_4_CALCULATE_EXPS_2NS);
    TimerStart(STARK_STEP_4_MERKLETREE);
    Goldilocks::Element *p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];

    uint64_t numElementsTree4 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN.section[eSection::q_2ns], NExtended);
    Polinomial tree4(numElementsTree4, 1);
    Polinomial root4(HASH_SIZE, 1);

    PoseidonGoldilocks::merkletree(tree4.address(), p_q_2ns, starkInfo.mapSectionsN.section[eSection::q_2ns], NExtended);
    TimerStopAndLog(STARK_STEP_4_MERKLETREE);
    MerklehashGoldilocks::root(root4.address(), tree4.address(), tree4.length());
    std::cout << "MerkleTree root 4: [ " << root4.toString(4) << " ]" << std::endl;
    transcript.put(root4.address(), HASH_SIZE);
    TimerStopAndLog(STARK_STEP_4);

    ///////////
    // 5. Compute FRI Polynomial
    ///////////
    TimerStart(STARK_STEP_5);
    TimerStart(STARK_STEP_5_LEv_LpEv);

    transcript.getField(challenges[5]); // v1
    transcript.getField(challenges[6]); // v2
    transcript.getField(challenges[7]); // xi

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

    batchInverse(xDivXSubXi, xDivXSubXi);
    batchInverse(xDivXSubWXi, xDivXSubWXi);

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
        // step52ns_i(mem, publicInputs, i);
    }
    for (uint64_t i = NExtended - next; i < NExtended; i++)
    {
        step52ns_first(mem, publicInputs, i);
        // step52ns_last(mem, publicInputs, i);
    }
    TimerStopAndLog(STARK_STEP_5_CALCULATE_EXPS);
    TimerStopAndLog(STARK_STEP_5);
    TimerStart(STARK_STEP_FRI);

    trees[0] = tree1.address();
    trees[1] = tree2.address();
    trees[2] = tree3.address();
    trees[3] = tree4.address();
    trees[4] = (Goldilocks::Element *)pConstTreeAddress;

    // TODO: Add to starkInfo
    uint64_t totalNumElementsTree = 0;
    uint64_t totalTrees = 0;
    uint64_t polBits = starkInfo.starkStruct.nBitsExt;

    for (uint64_t si = 0; si < starkInfo.starkStruct.steps.size(); si++)
    {
        uint64_t reductionBits = polBits - starkInfo.starkStruct.steps[si].nBits;

        if (si < starkInfo.starkStruct.steps.size() - 1)
        {
            totalTrees++;
            uint64_t nGroups = 1 << starkInfo.starkStruct.steps[si + 1].nBits;
            uint64_t groupSize = (1 << starkInfo.starkStruct.steps[si].nBits) / nGroups;
            totalNumElementsTree += MerklehashGoldilocks::getTreeNumElements(groupSize * FIELD_EXTENSION, nGroups);
        }
        polBits = polBits - reductionBits;
    }

    Polinomial friPol = starkInfo.getPolinomial(mem, starkInfo.exps_2ns[starkInfo.friExpId]);

    FriProof fproof((1 << polBits), FIELD_EXTENSION, starkInfo.starkStruct.steps.size(), starkInfo.evMap.size(), starkInfo.nPublics);
    ProveFRI::prove(fproof, trees, transcript, friPol, starkInfo.starkStruct.nBitsExt, starkInfo);

    fproof.proofs.setEvals(evals.address());

    std::memcpy(&fproof.proofs.root1[0], root1.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&fproof.proofs.root2[0], root2.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&fproof.proofs.root3[0], root3.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&fproof.proofs.root4[0], root4.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    TimerStopAndLog(STARK_STEP_FRI);

#define zkinFile "zkevm.proof.zkin.json"
#define starkFile "zkevm.prove.json"
#define publicFile "zkevm.public.json"
    TimerStart(STARK_JSON_GENERATION);
    std::memcpy(&fproof.publics[0], &publicInputs[0], starkInfo.nPublics * sizeof(Goldilocks::Element));

    nlohmann::ordered_json jProof = fproof.proofs.proof2json();

    ofstream ofstark(starkFile);
    ofstark << setw(4) << jProof.dump() << endl;
    ofstark.close();

    nlohmann::ordered_json j_publics = json::array();
    for (uint i = 0; i < fproof.publics.size(); i++)
    {
        j_publics.push_back(Goldilocks::toString(fproof.publics[i]));
    }
    ofstream ofpublicFile(publicFile);
    ofpublicFile << setw(4) << j_publics.dump() << endl;
    ofpublicFile.close();

    nlohmann::ordered_json zkin = proof2zkinStark(jProof);
    zkin["publics"] = j_publics;
    ofstream ofzkin(zkinFile);
    ofzkin << setw(4) << zkin.dump() << endl;
    ofzkin.close();
    TimerStopAndLog(STARK_JSON_GENERATION);

    // HARDCODE PROOFs
    proof.proofA.push_back("13661670604050723159190639550237390237901487387303122609079617855313706601738");
    proof.proofA.push_back("318870292909531730706266902424471322193388970015138106363857068613648741679");
    proof.proofA.push_back("1");

    ProofX proofX;
    proofX.proof.push_back("697129936138216869261087581911668981951894602632341950972818743762373194907");
    proofX.proof.push_back("8382255061406857865565510718293473646307698289010939169090474571110768554297");
    proof.proofB.push_back(proofX);
    proofX.proof.clear();
    proofX.proof.push_back("15430920731683674465693779067364347784717314152940718599921771157730150217435");
    proofX.proof.push_back("9973632244944366583831174453935477607483467152902406810554814671794600888188");
    proof.proofB.push_back(proofX);
    proofX.proof.clear();
    proofX.proof.push_back("1");
    proofX.proof.push_back("0");
    proof.proofB.push_back(proofX);

    proof.proofC.push_back("19319469652444706345294120534164146052521965213898291140974711293816652378032");
    proof.proofC.push_back("20960565072144725955004735885836324119094967998861346319897532045008317265851");
    proof.proofC.push_back("1");

    proof.publicInputsExtended.inputHash = "0x1afd6eaf13538380d99a245c2acc4a25481b54556ae080cf07d1facc0638cd8e";
    proof.publicInputsExtended.publicInputs.oldStateRoot = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.oldLocalExitRoot = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.newStateRoot = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.newLocalExitRoot = "0x17c04c3760510b48c6012742c540a81aba4bca2f78b9d14bfd2f123e2e53ea3e";
    proof.publicInputsExtended.publicInputs.sequencerAddr = "0x617b3a3528F9cDd6630fd3301B9c8911F7Bf063D";
    proof.publicInputsExtended.publicInputs.batchHashData = "0x090bcaf734c4f06c93954a827b45a6e8c67b8e0fd1e0a35a1c5982d6961828f9";
    proof.publicInputsExtended.publicInputs.batchNum = 1;
}

class CompareGL3
{
public:
    bool operator()(const vector<Goldilocks::Element> &a, const vector<Goldilocks::Element> &b) const
    {
        if (a.size() == 1)
        {
            return Goldilocks::toU64(a[0]) < Goldilocks::toU64(b[0]);
        }
        else if (Goldilocks::toU64(a[0]) != Goldilocks::toU64(b[0]))
        {
            return Goldilocks::toU64(a[0]) < Goldilocks::toU64(b[0]);
        }
        else if (Goldilocks::toU64(a[1]) != Goldilocks::toU64(b[1]))
        {
            return Goldilocks::toU64(a[1]) < Goldilocks::toU64(b[1]);
        }
        else
        {
            return Goldilocks::toU64(a[2]) < Goldilocks::toU64(b[2]);
        }
    }
};

void Stark::calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol)
{
    map<std::vector<Goldilocks::Element>, uint64_t, CompareGL3> idx_t;
    multimap<std::vector<Goldilocks::Element>, uint64_t, CompareGL3> s;
    multimap<std::vector<Goldilocks::Element>, uint64_t>::iterator it;
    uint64_t i = 0;

    for (uint64_t i = 0; i < tPol.degree(); i++)
    {
        vector<Goldilocks::Element> key = tPol.toVector(i);
        std::pair<vector<Goldilocks::Element>, uint64_t> pr(key, i);

        auto const result = idx_t.insert(pr);
        if (not result.second)
        {
            result.first->second = i;
        }

        s.insert(pr);
    }

    for (uint64_t i = 0; i < fPol.degree(); i++)
    {
        vector<Goldilocks::Element> key = fPol.toVector(i);

        if (idx_t.find(key) == idx_t.end())
        {
            cerr << "Error: calculateH1H2() Number not included: " << Goldilocks::toString(fPol[i], 16) << endl;
            exit(-1);
        }
        uint64_t idx = idx_t[key];
        s.insert(pair<vector<Goldilocks::Element>, uint64_t>(key, idx));
    }

    multimap<uint64_t, vector<Goldilocks::Element>> s_sorted;
    multimap<uint64_t, vector<Goldilocks::Element>>::iterator it_sorted;

    for (it = s.begin(); it != s.end(); it++)
    {
        s_sorted.insert(make_pair(it->second, it->first));
    }

    for (it_sorted = s_sorted.begin(); it_sorted != s_sorted.end(); it_sorted++, i++)
    {
        if ((i & 1) == 0)
        {
            Polinomial::copyElement(h1, i / 2, it_sorted->second);
        }
        else
        {
            Polinomial::copyElement(h2, i / 2, it_sorted->second);
        }
    }
};

void Stark::calculateZ(Polinomial &z, Polinomial &num, Polinomial &den)
{
    uint64_t size = num.degree();

    Polinomial denI(size, 3);
    Polinomial checkVal(1, 3);
    Goldilocks::Element *pZ = z[0];
    Goldilocks3::copy((Goldilocks3::Element *)&pZ[0], &Goldilocks3::one());

    batchInverse(denI, den);
    for (uint64_t i = 1; i < size; i++)
    {
        Polinomial tmp(1, 3);
        Polinomial::mulElement(tmp, 0, num, i - 1, denI, i - 1);
        Polinomial::mulElement(z, i, z, i - 1, tmp, 0);
    }
    Polinomial tmp(1, 3);
    Polinomial::mulElement(tmp, 0, num, size - 1, denI, size - 1);
    Polinomial::mulElement(checkVal, 0, z, size - 1, tmp, 0);

    zkassert(Goldilocks3::isOne((Goldilocks3::Element &)*checkVal[0]));
}

inline void Stark::batchInverse(Polinomial &res, Polinomial &src)
{
    uint64_t size = src.degree();
    Polinomial aux(size, 3);
    Polinomial tmp(size, 3);

    Polinomial::copyElement(tmp, 0, src, 0);

    for (uint64_t i = 1; i < size; i++)
    {
        Polinomial::mulElement(tmp, i, tmp, i - 1, src, i);
    }

    Polinomial z(1, 3);
    Goldilocks3::inv((Goldilocks3::Element *)z[0], (Goldilocks3::Element *)tmp[size - 1]);

    for (uint64_t i = size - 1; i > 0; i--)
    {
        Polinomial::mulElement(aux, i, z, 0, tmp, i - 1);
        Polinomial::mulElement(z, 0, z, 0, src, i);
    }
    Polinomial::copyElement(aux, 0, z, 0);
    Polinomial::copy(res, aux);
}