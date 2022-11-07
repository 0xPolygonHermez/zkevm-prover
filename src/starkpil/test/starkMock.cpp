#include <iomanip>
#include "starkMock.hpp"

StarkMock::StarkMock(const Config &config) : config(config),
                                             starkInfo(config, config.starkInfoFile),
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
void StarkMock::genProof(void *pAddress, FRIProof &proof)
{
    // Initialize vars
    Transcript transcript;
    std::memset(challenges.address(), 0, challenges.size());
    std::memset(xDivXSubXi.address(), 0, xDivXSubXi.size());
    std::memset(xDivXSubWXi.address(), 0, xDivXSubWXi.size());
    std::memset(evals.address(), 0, evals.size());

    ///////////
    // 1.- Calculate p_cm1_2ns
    ///////////
    TimerStart(STARK_STEP_1);

    Goldilocks::Element publicInputs[8];

    publicInputs[0] = Goldilocks::fromString("3248459814");
    publicInputs[1] = Goldilocks::fromString("1620587195");
    publicInputs[2] = Goldilocks::fromString("3678822139");
    publicInputs[3] = Goldilocks::fromString("1824295850");
    publicInputs[4] = Goldilocks::fromString("366027599");
    publicInputs[5] = Goldilocks::fromString("1355324045");
    publicInputs[6] = Goldilocks::fromString("1531026716");
    publicInputs[7] = Goldilocks::fromString("1017354875");
    publicInputs[8] = Goldilocks::fromString("0");
    publicInputs[9] = Goldilocks::fromString("0");
    publicInputs[10] = Goldilocks::fromString("0");
    publicInputs[11] = Goldilocks::fromString("0");
    publicInputs[12] = Goldilocks::fromString("0");
    publicInputs[13] = Goldilocks::fromString("0");
    publicInputs[14] = Goldilocks::fromString("0");
    publicInputs[15] = Goldilocks::fromString("0");
    publicInputs[16] = Goldilocks::fromString("0");
    publicInputs[17] = Goldilocks::fromString("1000");
    publicInputs[18] = Goldilocks::fromString("510351649");
    publicInputs[19] = Goldilocks::fromString("2243740642");
    publicInputs[20] = Goldilocks::fromString("121390774");
    publicInputs[21] = Goldilocks::fromString("3088140970");
    publicInputs[22] = Goldilocks::fromString("2387924872");
    publicInputs[23] = Goldilocks::fromString("2930644697");
    publicInputs[24] = Goldilocks::fromString("923028121");
    publicInputs[25] = Goldilocks::fromString("2301051566");
    publicInputs[26] = Goldilocks::fromString("537003291");
    publicInputs[27] = Goldilocks::fromString("344094503");
    publicInputs[28] = Goldilocks::fromString("251860201");
    publicInputs[29] = Goldilocks::fromString("686198245");
    publicInputs[30] = Goldilocks::fromString("3667240819");
    publicInputs[31] = Goldilocks::fromString("1437754387");
    publicInputs[32] = Goldilocks::fromString("2701071742");
    publicInputs[33] = Goldilocks::fromString("568001667");
    publicInputs[34] = Goldilocks::fromString("0");
    publicInputs[35] = Goldilocks::fromString("0");
    publicInputs[36] = Goldilocks::fromString("0");
    publicInputs[37] = Goldilocks::fromString("0");
    publicInputs[38] = Goldilocks::fromString("0");
    publicInputs[39] = Goldilocks::fromString("0");
    publicInputs[40] = Goldilocks::fromString("0");
    publicInputs[41] = Goldilocks::fromString("0");
    publicInputs[42] = Goldilocks::fromString("1");

    CommitPols cmPols(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);

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

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    TimerStart(STARK_STEP_2);
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal

    std::cout << "Challenges:\n"
              << challenges.toString(2) << std::endl;

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
    TimerStart(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE);

    if (starkInfo.puCtx.size() != 0)
    {

        Polinomial aux0 = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[0].fExpId]);
        u_int64_t stride_pol0 = aux0.degree() * FIELD_EXTENSION + 8;
        uint64_t tot_pols0 = 4 * starkInfo.puCtx.size();
        uint64_t tot_size0 = stride_pol0 * tot_pols0 * (u_int64_t)sizeof(Goldilocks::Element);

        Polinomial *newpols0 = new Polinomial[tot_pols0];
        Goldilocks::Element *buffpols0 = (Goldilocks::Element *)malloc(tot_size0);
        if (buffpols0 == NULL || newpols0 == NULL)
        {
            cout << "memory problems!" << endl;
            exit(1);
        }

        //#pragma omp parallel for
        for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
        {
            Polinomial fPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].fExpId]);
            Polinomial tPol = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].tExpId]);
            Polinomial h1 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2]);
            Polinomial h2 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2 + 1]);

            uint64_t indx = i * 4;
            newpols0[indx].potConstruct(&(buffpols0[indx * stride_pol0]), fPol.degree(), fPol.dim(), fPol.dim());
            Polinomial::copy(newpols0[indx], fPol);
            indx++;
            newpols0[indx].potConstruct(&(buffpols0[indx * stride_pol0]), tPol.degree(), tPol.dim(), tPol.dim());
            Polinomial::copy(newpols0[indx], tPol);
            indx++;

            newpols0[indx].potConstruct(&(buffpols0[indx * stride_pol0]), h1.degree(), h1.dim(), h1.dim());
            indx++;

            newpols0[indx].potConstruct(&(buffpols0[indx * stride_pol0]), h2.degree(), h2.dim(), h2.dim());
        }
        TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE);

        TimerStart(STARK_STEP_2_CALCULATEH1H2);
#pragma omp parallel for num_threads(8)
        for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
        {
            int indx1 = 4 * i;
            Polinomial::calculateH1H2_(newpols0[indx1 + 2], newpols0[indx1 + 3], newpols0[indx1], newpols0[indx1 + 1], i);
        }
        TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2);

        TimerStart(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE_2);
        for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
        {
            int indx1 = 4 * i + 2;
            int indx2 = 4 * i + 3;
            Polinomial h1 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2]);
            Polinomial h2 = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i * 2 + 1]);
            Polinomial::copy(h1, newpols0[indx1]);
            Polinomial::copy(h2, newpols0[indx2]);
        }
        delete[] newpols0;
        free(buffpols0);
        TimerStopAndLog(STARK_STEP_2_CALCULATEH1H2_TRANSPOSE_2);
    }
    numCommited = numCommited + starkInfo.puCtx.size() * 2;

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
    TimerStart(STARK_STEP_3_CALCULATE_Z_TRANSPOSE);

    if (starkInfo.puCtx.size() != 0)
    {
        Polinomial aux = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[0].numId]);
        uint64_t stride_pol_ = aux.degree() * FIELD_EXTENSION + 8; // assuming all polinomials have same degree
        uint64_t tot_pols = 3 * (starkInfo.puCtx.size() + starkInfo.peCtx.size() + starkInfo.ciCtx.size());
        uint64_t tot_size_ = stride_pol_ * tot_pols * (u_int64_t)sizeof(Goldilocks::Element);
        Polinomial *newpols_ = (Polinomial *)malloc(tot_pols * sizeof(Polinomial));
        Goldilocks::Element *buffpols_ = (Goldilocks::Element *)malloc(tot_size_);
        if (buffpols_ == NULL || newpols_ == NULL)
        {
            cout << "memory problems!" << endl;
            exit(1);
        }
        //#pragma omp parallel for (better without)
        for (uint64_t i = 0; i < starkInfo.puCtx.size(); i++)
        {
            Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].numId]);
            Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.puCtx[i].denId]);
            Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
            u_int64_t indx = i * 3;
            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), pNum.degree(), pNum.dim(), pNum.dim());
            Polinomial::copy(newpols_[indx], pNum);
            indx++;
            assert(pNum.degree() <= aux.degree());

            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), pDen.degree(), pDen.dim(), pDen.dim());
            Polinomial::copy(newpols_[indx], pDen);
            indx++;
            assert(pDen.degree() <= aux.degree());

            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), z.degree(), z.dim(), z.dim());
            assert(z.degree() <= aux.degree());
        }

        numCommited += starkInfo.puCtx.size();
        u_int64_t offset = 3 * starkInfo.puCtx.size();
        for (uint64_t i = 0; i < starkInfo.peCtx.size(); i++)
        {
            Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].numId]);
            Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.peCtx[i].denId]);
            Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
            u_int64_t indx = 3 * i + offset;
            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), pNum.degree(), pNum.dim(), pNum.dim());
            Polinomial::copy(newpols_[indx], pNum);
            indx++;
            assert(pNum.degree() <= aux.degree());

            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), pDen.degree(), pDen.dim(), pDen.dim());
            Polinomial::copy(newpols_[indx], pDen);
            indx++;
            assert(pDen.degree() <= aux.degree());

            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), z.degree(), z.dim(), z.dim());
            assert(z.degree() <= aux.degree());
        }
        numCommited += starkInfo.peCtx.size();
        offset += 3 * starkInfo.peCtx.size();
        for (uint64_t i = 0; i < starkInfo.ciCtx.size(); i++)
        {

            Polinomial pNum = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].numId]);
            Polinomial pDen = starkInfo.getPolinomial(mem, starkInfo.exps_n[starkInfo.ciCtx[i].denId]);
            Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
            u_int64_t indx = 3 * i + offset;

            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), pNum.degree(), pNum.dim(), pNum.dim());
            Polinomial::copy(newpols_[indx], pNum);
            indx++;
            assert(pNum.degree() <= aux.degree());

            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), pDen.degree(), pDen.dim(), pDen.dim());
            Polinomial::copy(newpols_[indx], pDen);
            indx++;
            assert(pDen.degree() <= aux.degree());

            newpols_[indx].potConstruct(&(buffpols_[indx * stride_pol_]), z.degree(), z.dim(), z.dim());
            assert(z.degree() <= aux.degree());
        }
        numCommited += starkInfo.ciCtx.size();
        numCommited -= starkInfo.ciCtx.size() + starkInfo.peCtx.size() + starkInfo.puCtx.size();

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
        for (uint64_t i = 0; i < numpols; i++)
        {
            int indx1 = 3 * i;
            Polinomial z = starkInfo.getPolinomial(mem, starkInfo.cm_n[numCommited + i]);
            Polinomial::copy(z, newpols_[indx1 + 2]);
        }
        free(newpols_);
        free(buffpols_);
        TimerStopAndLog(STARK_STEP_3_CALCULATE_Z_TRANSPOSE_2);
    }

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
#endif
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

    Polinomial friPol = starkInfo.getPolinomial(mem, starkInfo.exps_2ns[starkInfo.friExpId]);
    FRIProve::prove(proof, trees, transcript, friPol, starkInfo.starkStruct.nBitsExt, starkInfo);

    proof.proofs.setEvals(evals.address());

    std::memcpy(&proof.proofs.root1[0], root1.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root2[0], root2.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root3[0], root3.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root4[0], root4.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    TimerStopAndLog(STARK_STEP_FRI);
}
