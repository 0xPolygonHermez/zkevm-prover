#include "stark_test.hpp"
#include "timer.hpp"
#include "utils.hpp"

#include "ntt_goldilocks.hpp"

#include "calculateExps_all.hpp"

#define NUM_CHALLENGES 8

void StarkTest(void)
{
#include "public_inputs_all.hpp"
#define starkInfo_File "all.starkinfo.json"
#define commited_file "all.commit"
#define constant_file "all.const"
#define constant_tree_file "all.consttree"
    // Load config & test vectors
    Config cfg;
    cfg.starkInfoFile = starkInfo_File;
    cfg.constPolsFile = constant_file;
    cfg.mapConstPolsFile = false;
    cfg.runProverServer = true;
    cfg.constantsTreeFile = "all.consttree";
    StarkInfo starkInfo(cfg);
    StarkTestMock stark(cfg);

    void *pAddress = malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    void *pCommitedAddress = mapFile(commited_file, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element), false);
    std::memcpy(pAddress, pCommitedAddress, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element));

    CommitPolsAll cmP(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);
    PublicInputsAll publics;
    publics.A = 1;
    publics.B = 2;
    publics.C = 74469561660084004;

    void *pConstantAddress = NULL;
    pConstantAddress = mapFile(constant_file, starkInfo.nConstants * (1 << starkInfo.starkStruct.nBits) * sizeof(Goldilocks::Element), false);
    ConstantPolsAll const_n(pConstantAddress, (1 << starkInfo.starkStruct.nBits));

    Proof proof;

    stark.genProof(pAddress, cmP, const_n, publics, proof);
}

void StarkTestMock::calculateH1H2(Polinomial &h1, Polinomial &h2, Polinomial &fPol, Polinomial &tPol)
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
}

StarkTestMock::StarkTestMock(const Config &config) : config(config),
                                                     starkInfo(config),
                                                     zi(config.generateProof() ? starkInfo.starkStruct.nBits : 0,
                                                        config.generateProof() ? starkInfo.starkStruct.nBitsExt : 0),
                                                     numCommited(starkInfo.nCm1),
                                                     N(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                     NExtended(config.generateProof() ? 1 << starkInfo.starkStruct.nBitsExt : 0),
                                                     ntt(config.generateProof() ? 1 << starkInfo.starkStruct.nBits : 0),
                                                     x_n(config.generateProof() ? N : 0, config.generateProof() ? 1 : 0),
                                                     x_2ns(config.generateProof() ? NExtended : 0, config.generateProof() ? 1 : 0),
                                                     challenges(config.generateProof() ? NUM_CHALLENGES : 0, config.generateProof() ? FIELD_EXTENSION : 0)

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
        pConstPolsAddress = mapFile(config.constPolsFile, ConstantPolsAll::pilSize(), false);
        cout << "Stark::Stark() successfully mapped " << ConstantPolsAll::pilSize() << " bytes from constant file " << config.constPolsFile << endl;
    }
    else
    {
        pConstPolsAddress = copyFile(config.constPolsFile, ConstantPolsAll::pilSize());
        cout << "Stark::Stark() successfully copied " << ConstantPolsAll::pilSize() << " bytes from constant file " << config.constPolsFile << endl;
    }
    pConstPols = new ConstantPolsAll(pConstPolsAddress, ConstantPolsAll::pilDegree());
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
    pConstPols2ns = new ConstantPolsAll(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt));

#pragma omp parallel for collapse(2)
    for (uint64_t i = 0; i < starkInfo.nConstants; i++)
    {
        for (uint64_t j = 0; j < NExtended; j++)
        {
            MerklehashGoldilocks::getElement(((ConstantPolsAll *)pConstPols2ns)->getElement(i, j), (Goldilocks::Element *)pConstTreeAddress, j, i);
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

StarkTestMock::~StarkTestMock()
{
    if (!config.generateProof())
        return;

    delete pConstPols;
    if (config.mapConstPolsFile)
    {
        unmapFile(pConstPolsAddress, ConstantPolsAll::pilSize());
    }
    else
    {
        free(pConstPolsAddress);
    }

    // free(pConstPolsAddress2ns);
    // delete pConstPols2ns;
}
void StarkTestMock::genProof(void *pAddress, CommitPolsAll &cmPols, ConstantPolsAll &const_n, const PublicInputsAll &publicInputs, Proof &proof)
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
    CalculateExpsAll::step2prev_first(mem, const_n, (Goldilocks3::Element *)challenges.address(), 0);

#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        CalculateExpsAll::step2prev_i(mem, const_n, (Goldilocks3::Element *)challenges.address(), i);
    }
    CalculateExpsAll::step2prev_last(mem, const_n, (Goldilocks3::Element *)challenges.address(), N - 1);

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
}