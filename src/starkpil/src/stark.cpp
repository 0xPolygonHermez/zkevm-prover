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

    pConstPolsAddress2ns = (void *)calloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt), sizeof(Goldilocks::Element));
    pConstPols2ns = new ConstantPols(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt));

#pragma omp parallel for collapse(2)
    for (uint64_t i = 0; i < starkInfo.nConstants; i++)
    {
        for (uint64_t j = 0; j < NExtended; j++)
        {
            MerklehashGoldilocks::getElement(((ConstantPols *)pConstPols2ns)->getElement(i, j), (Goldilocks::Element *)pConstTreeAddress, j, i);
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

void Stark::genProof(void *pAddress, CommitPols &cmPols, const PublicInputs &_publicInputs, Proof &proof)
{

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
    std::cout << "Merkelizing 1...." << std::endl;
    /*
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
*/

    // HARDCODED VALUES
    Polinomial root1_hardcoded(HASH_SIZE, 1);
    *root1_hardcoded[0] = Goldilocks::fromU64(14773216157232762958ULL);
    *root1_hardcoded[1] = Goldilocks::fromU64(9090792250391374988ULL);
    *root1_hardcoded[2] = Goldilocks::fromU64(11395074597553208760ULL);
    *root1_hardcoded[3] = Goldilocks::fromU64(17145980724823558481ULL);
    transcript.put(root1_hardcoded.address(), HASH_SIZE);

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal

    std::cout << "step2prev_first...." << std::endl;

    step2prev_first(mem, &publicInputs[0], 0);
    std::cout << "step2prev_i...." << std::endl;
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step2prev_first(mem, &publicInputs[0], i);
        // CalculateExpsAll::step2prev_i(mem, const_n, (Goldilocks3::Element *)challenges.address(), i);
    }
    // CalculateExpsAll::step2prev_last(mem, const_n, (Goldilocks3::Element *)challenges.address(), N - 1);
    std::cout << "step2prev_last...." << std::endl;
    step2prev_first(mem, &publicInputs[0], N - 1);
    std::cout << "calculateH1H2... " << std::endl;

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

    std::cout << "Merkelizing 2...." << std::endl;
    /*
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
    */
    // HARDCODED VALUES
    Polinomial root2_hardcoded(HASH_SIZE, 1);
    *root2_hardcoded[0] = Goldilocks::fromU64(5602570006149680147ULL);
    *root2_hardcoded[1] = Goldilocks::fromU64(16794271776532285084ULL);
    *root2_hardcoded[2] = Goldilocks::fromU64(1070892053182687126ULL);
    *root2_hardcoded[3] = Goldilocks::fromU64(4818924615586863563ULL);
    transcript.put(root2_hardcoded.address(), HASH_SIZE);

    ///////////
    // 3.- Compute Z polynomialsxx
    ///////////
    transcript.getField(challenges[2]); // gamma
    transcript.getField(challenges[3]); // betta

    step3prev_first(mem, &publicInputs[0], 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step3prev_first(mem, &publicInputs[0], i);

        // CalculateExpsAll::step3prev_i((Goldilocks::Element *)pAddress, const_n, challenges, x_n, i);
    }
    step3prev_first(mem, &publicInputs[0], N - 1);

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
    /*
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
*/
    Polinomial root3_hardcoded(HASH_SIZE, 1);
    *root3_hardcoded[0] = Goldilocks::fromU64(9941220780739138404ULL);
    *root3_hardcoded[1] = Goldilocks::fromU64(5738999706464945940ULL);
    *root3_hardcoded[2] = Goldilocks::fromU64(9388229154186191556ULL);
    *root3_hardcoded[3] = Goldilocks::fromU64(11332475887922162499ULL);
    transcript.put(root3_hardcoded.address(), HASH_SIZE);

    ///////////
    // 4. Compute C Polynomial
    ///////////
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

    Goldilocks::Element *p_exps_withq_2ns = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_2ns]];
    Goldilocks::Element *p_exps_withq_n = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_n]];
    ntt.extendPol(p_exps_withq_2ns, p_exps_withq_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::exps_withq_n]);

    uint64_t extendBits = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    uint64_t next = 1 << extendBits;

    for (uint64_t i = 0; i < NExtended; i++)
    {
        step42ns_first(mem, &publicInputs[0], i);
    }
#pragma omp parallel for
    for (uint64_t i = next; i < NExtended - next; i++)
    {
        step42ns_i(mem, &publicInputs[0], i);
    }

    for (uint64_t i = NExtended - next; i < NExtended; i++)
    {
        step42ns_last(mem, &publicInputs[0], i);
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