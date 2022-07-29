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

void Stark::genProof(void *pAddress, CommitPols &cmPols, const PublicInputs &publicInputs, Proof &proof)
{
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
        return Goldilocks::toU64(a[1]) < Goldilocks::toU64(b[1]);
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
        vector<Goldilocks::Element> key = Goldilocks3::toVector((Goldilocks3::Element *)tPol[i]);
        std::pair<vector<Goldilocks::Element>, uint64_t> pr(key, i);
        idx_t.insert(pr);
        s.insert(pr);
    }

    for (uint64_t i = 0; i < fPol.degree(); i++)
    {
        vector<Goldilocks::Element> key = Goldilocks3::toVector((Goldilocks3::Element *)fPol[i]);

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
        Goldilocks::Element *h = it_sorted->second.data();

        if ((i & 1) == 0)
        {
            Goldilocks3::copy((Goldilocks3::Element *)h1[i / 2], (Goldilocks3::Element *)h);
        }
        else
        {
            Goldilocks3::copy((Goldilocks3::Element *)h2[i / 2], (Goldilocks3::Element *)h);
        }
    }
};