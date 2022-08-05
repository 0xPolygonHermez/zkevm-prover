 #include <iomanip>
 #include "starkMock.hpp"

StarkMock::StarkMock(const Config &config) : config(config),
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
        pConstPolsAddress = mapFile(config.constPolsFile, ConstantPolsBasic::pilSize(), false);
        cout << "Stark::Stark() successfully mapped " << ConstantPolsBasic::pilSize() << " bytes from constant file " << config.constPolsFile << endl;
    }
    else
    {
        pConstPolsAddress = copyFile(config.constPolsFile, ConstantPolsBasic::pilSize());
        cout << "Stark::Stark() successfully copied " << ConstantPolsBasic::pilSize() << " bytes from constant file " << config.constPolsFile << endl;
    }
    pConstPols = new ConstantPolsBasic(pConstPolsAddress, ConstantPolsBasic::pilDegree());
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

    pConstPolsAddress2ns = (void *)calloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt), sizeof(Goldilocks::Element));
    pConstPols2ns = new ConstantPolsBasic(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt));

#pragma omp parallel for collapse(2)
    for (uint64_t i = 0; i < starkInfo.nConstants; i++)
    {
        for (uint64_t j = 0; j < NExtended; j++)
        {
            MerklehashGoldilocks::getElement(((ConstantPolsBasic *)pConstPols2ns)->getElement(i, j), (Goldilocks::Element *)pConstTreeAddress, j, i);
        }
    }

    // TODO x_n and x_2ns could be precomputed
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
}

StarkMock::~StarkMock()
{
    if (!config.generateProof())
        return;

    delete pConstPols;
    if (config.mapConstPolsFile)
    {
        unmapFile(pConstPolsAddress, ConstantPolsBasic::pilSize());
    }
    else
    {
        free(pConstPolsAddress);
    }
}
void StarkMock::genProof(void *pAddress, CommitPolsBasic &cmPols, Proof &proof)
{
    Goldilocks::Element publicInputs[8];
    publicInputs[0] = Goldilocks::fromU64(2043100198);
    publicInputs[1] = Goldilocks::fromU64(2909753411);
    publicInputs[2] = Goldilocks::fromU64(2146825699);
    publicInputs[3] = Goldilocks::fromU64(3866023039);
    publicInputs[4] = Goldilocks::fromU64(1719628537);
    publicInputs[5] = Goldilocks::fromU64(3739677152);
    publicInputs[6] = Goldilocks::fromU64(1596594856);
    publicInputs[7] = Goldilocks::fromU64(3497182697);

    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;
    std::cout << "Merkelizing 1...." << std::endl;

    uint64_t numElementsTree1 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm1_n] + starkInfo.mapSectionsN3.section[eSection::cm1_n] * FIELD_EXTENSION, NExtended);

    Polinomial tree1(numElementsTree1, 1);
    Polinomial root1(HASH_SIZE, 1);

    Goldilocks::Element *p_cm1_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns]];
    Goldilocks::Element *p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];

    ntt.extendPol(p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n]);

    PoseidonGoldilocks::merkletree(tree1.address(), p_cm1_2ns, starkInfo.mapSectionsN.section[eSection::cm1_n], NExtended);

    MerklehashGoldilocks::root(root1.address(), tree1.address(), tree1.length());
    std::cout << "MerkleTree root 1: [ " << root1.toString(4) << " ]" << std::endl;
    transcript.put(root1.address(), HASH_SIZE);

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal

    step2prev_first(mem, publicInputs, 0);

    //#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step2prev_first(mem, publicInputs, i);
        // CalculateExpsAll::step2prev_i(mem, const_n, (Goldilocks3::Element *)challenges.address(), i);
    }
    // CalculateExpsAll::step2prev_last(mem, const_n, (Goldilocks3::Element *)challenges.address(), N - 1);
    step2prev_first(mem, publicInputs, N - 1);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        Polinomial fPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]);
        Polinomial tPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]);
        Polinomial h1 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);
        Polinomial h2 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);

        std::cout << fPol.toString(3) << std::endl;
        std::cout << tPol.toString(3) << std::endl;

        calculateH1H2(h1, h2, fPol, tPol);
        std::cout << h1.toString(3) << std::endl;
        std::cout << h2.toString(3) << std::endl;
    }

    std::cout << "Merkelizing 2...." << std::endl;
    Goldilocks::Element *p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
    Goldilocks::Element *p_cm2_n = &mem[starkInfo.mapOffsets.section[eSection::cm2_n]];
    ntt.extendPol(p_cm2_2ns, p_cm2_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm2_n]);

    uint64_t numElementsTree2 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm2_n] + starkInfo.mapSectionsN3.section[eSection::cm2_n] * FIELD_EXTENSION, NExtended);

    Polinomial tree2(numElementsTree2, 1);
    Polinomial root2(HASH_SIZE, 1);

    PoseidonGoldilocks::merkletree(tree2.address(), p_cm2_2ns, starkInfo.mapSectionsN.section[eSection::cm2_n], NExtended);

    MerklehashGoldilocks::root(root2.address(), tree2.address(), tree2.length());
    std::cout << "MerkleTree root 2: [ " << root2.toString(4) << " ]" << std::endl;
    transcript.put(root2.address(), HASH_SIZE);

    ///////////
    // 3.- Compute Z polynomialsxx
    ///////////
    transcript.getField(challenges[2]); // gamma
    transcript.getField(challenges[3]); // betta

    step3prev_first(mem, publicInputs, 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step3prev_first(mem, publicInputs, i);

        // CalculateExpsAll::step3prev_i((Goldilocks::Element *)pAddress, const_n, challenges, x_n, i);
    }
    step3prev_first(mem, publicInputs, N - 1);

    // CalculateExpsAll::step3prev_last((Goldilocks::Element *)pAddress, const_n, challenges, x_n, N - 1);

    for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
    {
        std::cout << "Calculating z for plookup " << i << std::endl;

        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].denId]);

        std::cout << pNum.toString(3) << std::endl;
        std::cout << pDen.toString(3) << std::endl;

        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);

        calculateZ(z, pNum, pDen);

        std::cout << z.toString(3) << std::endl;
    }
    for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
    {
        std::cout << "Calculating z for permutation check  " << i << std::endl;

        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].denId]);

        std::cout << pNum.toString(3) << std::endl;
        std::cout << pDen.toString(3) << std::endl;

        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);

        calculateZ(z, pNum, pDen);

        std::cout << z.toString(3) << std::endl;
    }
    for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
    {
        std::cout << "Calculating z for connection  " << i << std::endl;

        Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].numId]);
        Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].denId]);

        std::cout << pNum.toString(3) << std::endl;
        std::cout << pDen.toString(3) << std::endl;

        Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited++]);

        calculateZ(z, pNum, pDen);

        std::cout << z.toString(3) << std::endl;
    }
    std::cout << "Merkelizing 3...." << std::endl;

    Goldilocks::Element *p_cm3_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
    Goldilocks::Element *p_cm3_n = &mem[starkInfo.mapOffsets.section[eSection::cm3_n]];
    ntt.extendPol(p_cm3_2ns, p_cm3_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm3_n]);

    uint64_t numElementsTree3 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN1.section[eSection::cm3_n] + starkInfo.mapSectionsN3.section[eSection::cm3_n] * FIELD_EXTENSION, NExtended);

    Polinomial tree3(numElementsTree3, 1);
    Polinomial root3(HASH_SIZE, 1);

    PoseidonGoldilocks::merkletree(tree3.address(), p_cm3_2ns, starkInfo.mapSectionsN.section[eSection::cm3_n], NExtended);

    MerklehashGoldilocks::root(root3.address(), tree3.address(), tree3.length());
    std::cout << "MerkleTree root 3: [ " << root3.toString(4) << " ]" << std::endl;
    transcript.put(root3.address(), HASH_SIZE);

    ///////////
    // 4. Compute C Polynomial
    ///////////
    transcript.getField(challenges[4]); // gamma

    step4_first(mem, publicInputs, 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step4_first(mem, publicInputs, i);

        // CalculateExpsAll::step3prev_i((Goldilocks::Element *)pAddress, const_n, challenges, x_n, i);
    }
    step4_first(mem, publicInputs, N - 1);

    // CalculateExpsAll::step3prev_last((Goldilocks::Element *)pAddress, const_n, challenges, x_n, N - 1);

    Goldilocks::Element *p_exps_withq_2ns = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_2ns]];
    Goldilocks::Element *p_exps_withq_n = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_n]];
    ntt.extendPol(p_exps_withq_2ns, p_exps_withq_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::exps_withq_n]);

    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    uint64_t next = 1 << extendBits;

    for (uint64_t i = 0; i < next; i++)
    {
        step42ns_first(mem, publicInputs, i);
    }
#pragma omp parallel for
    for (uint64_t i = next; i < NExtended - next; i++)
    {
        step42ns_i(mem, publicInputs, i);
    }

    for (uint64_t i = NExtended - next; i < NExtended; i++)
    {
        step42ns_last(mem, publicInputs, i);
    }

    std::cout << "Merkelizing 4...." << std::endl;
    Goldilocks::Element *p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];

    uint64_t numElementsTree4 = MerklehashGoldilocks::getTreeNumElements(starkInfo.mapSectionsN.section[eSection::q_2ns], NExtended);
    Polinomial tree4(numElementsTree4, 1);
    Polinomial root4(HASH_SIZE, 1);

    PoseidonGoldilocks::merkletree(tree4.address(), p_q_2ns, starkInfo.mapSectionsN.section[eSection::q_2ns], NExtended);

    MerklehashGoldilocks::root(root4.address(), tree4.address(), tree4.length());
    std::cout << "MerkleTree root 4: [ " << root4.toString(4) << " ]" << std::endl;
    transcript.put(root4.address(), HASH_SIZE);

    ///////////
    // 5. Compute FRI Polynomial
    ///////////
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

    std::cout << LEv.toString(5) << std::endl;

    std::cout << LpEv.toString(5) << std::endl;

    for (uint64_t i = 0; i < starkInfo.evMap.size(); i++)
    {
        EvMap ev = starkInfo.evMap[i];

        Polinomial acc(1, FIELD_EXTENSION);
        Polinomial tmp(1, FIELD_EXTENSION);
        if (ev.type == EvMap::eType::_const)
        {
            Polinomial p(&((Goldilocks::Element *)pConstPols2ns->address())[ev.id], pConstPols2ns->degree(), 1, pConstPols2ns->numPols());

            std::cout << p.toString(10) << std::endl;

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
    std::cout << evals.toString(51) << std::endl;

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

    std::memcpy(&fproof.publics[0], &publicInputs[0], starkInfo.nPublics * sizeof(Goldilocks::Element));

    nlohmann::ordered_json jProof = fproof.proofs.proof2json();

    ofstream ofstark(starkFile);
    ofstark << std::setw(4) << jProof.dump() << endl;
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
}

inline void StarkMock::batchInverse(Polinomial &res, Polinomial &src)
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

void StarkMock::calculateZ(Polinomial &z, Polinomial &num, Polinomial &den)
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
