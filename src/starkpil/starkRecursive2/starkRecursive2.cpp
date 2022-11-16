#include "starkRecursive2.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include "ntt_goldilocks.hpp"

#define NUM_CHALLENGES 8

StarkRecursive2::StarkRecursive2(const Config &config) : config(config),
                                                         starkInfo(config, config.recursive2StarkInfo),
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
    TimerStart(LOAD_RECURSIVE_2_CONST_POLS_TO_MEMORY);
    pConstPolsAddress = NULL;
    if (config.recursive2ConstPols.size() == 0)
    {
        cerr << "Error: StarkRecursive2::StarkRecursive2() received an empty config.recursive2ConstPols" << endl;
        exit(-1);
    }

    if (config.mapConstPolsFile)
    {
        pConstPolsAddress = mapFile(config.recursive2ConstPols, ConstantPolsRecursive2::pilSize(), false);
        cout << "StarkRecursive2::StarkRecursive2() successfully mapped " << ConstantPolsRecursive2::pilSize() << " bytes from constant file " << config.recursive2ConstPols << endl;
    }
    else
    {
        pConstPolsAddress = copyFile(config.recursive2ConstPols, ConstantPolsRecursive2::pilSize());
        cout << "StarkRecursive2::StarkRecursive2() successfully copied " << ConstantPolsRecursive2::pilSize() << " bytes from constant file " << config.recursive2ConstPols << endl;
    }
    pConstPols = new ConstantPolsRecursive2(pConstPolsAddress, ConstantPolsRecursive2::pilDegree());
    TimerStopAndLog(LOAD_RECURSIVE_2_CONST_POLS_TO_MEMORY);

    // Map constants tree file to memory

    TimerStart(LOAD_RECURSIVE_2_CONST_TREE_TO_MEMORY);
    pConstTreeAddress = NULL;
    if (config.recursive2ConstantsTree.size() == 0)
    {
        cerr << "Error: StarkRecursive2::StarkRecursive2() received an empty config.recursive2ConstantsTree" << endl;
        exit(-1);
    }

    if (config.mapConstantsTreeFile)
    {
        pConstTreeAddress = mapFile(config.recursive2ConstantsTree, starkInfo.getConstTreeSizeInBytes(), false);
        cout << "StarkRecursive2::StarkRecursive2() successfully mapped " << starkInfo.getConstTreeSizeInBytes() << " bytes from constant tree file " << config.recursive2ConstantsTree << endl;
    }
    else
    {
        pConstTreeAddress = copyFile(config.recursive2ConstantsTree, starkInfo.getConstTreeSizeInBytes());
        cout << "StarkRecursive2::StarkRecursive2() successfully copied " << starkInfo.getConstTreeSizeInBytes() << " bytes from constant file " << config.recursive2ConstantsTree << endl;
    }
    TimerStopAndLog(LOAD_RECURSIVE_2_CONST_TREE_TO_MEMORY);

    // Initialize and allocate ConstantPols2ns
    TimerStart(LOAD_RECURSIVE_2_CONST_POLS_2NS_TO_MEMORY);
    pConstPolsAddress2ns = (void *)calloc(starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt), sizeof(Goldilocks::Element));
    pConstPols2ns = new ConstantPolsRecursive2(pConstPolsAddress2ns, (1 << starkInfo.starkStruct.nBitsExt));
    std::memcpy(pConstPolsAddress2ns, (uint8_t *)pConstTreeAddress + 2 * sizeof(Goldilocks::Element), starkInfo.nConstants * (1 << starkInfo.starkStruct.nBitsExt) * sizeof(Goldilocks::Element));

    TimerStopAndLog(LOAD_RECURSIVE_2_CONST_POLS_2NS_TO_MEMORY);

    // TODO x_n and x_2ns could be precomputed
    TimerStart(COMPUTE_X_N_AND_X_2_NS);
    Goldilocks::Element xx = Goldilocks::one();
    for (uint64_t i = 0; i < N; i++)
    {
        *x_n[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBits));
    }
    xx = Goldilocks::shift();
    for (uint64_t i = 0; i < NExtended; i++)
    {
        *x_2ns[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBitsExt));
    }
    for (uint i = 0; i < 5; i++)
    {
        treesGL[i] = new MerkleTreeGL();
    }
    TimerStopAndLog(COMPUTE_X_N_AND_X_2_NS);
}

StarkRecursive2::~StarkRecursive2()
{
    if (!config.generateProof())
        return;

    delete pConstPols;
    delete pConstPols2ns;
    free(pConstPolsAddress2ns);

    if (config.mapConstPolsFile)
    {
        unmapFile(pConstPolsAddress, ConstantPolsRecursive2::pilSize());
    }
    else
    {
        free(pConstPolsAddress);
    }
    if (config.mapConstantsTreeFile)
    {
        unmapFile(pConstTreeAddress, ConstantPolsRecursive2::pilSize());
    }
    else
    {
        free(pConstTreeAddress);
    }

    for (uint i = 0; i < 5; i++)
    {
        delete treesGL[i];
    }
}

void StarkRecursive2::genProof(void *pAddress, FRIProof &proof, Goldilocks::Element publicInputs[43])
{
    // Reset
    reset();

    // Initialize vars
    Transcript transcript;
    CommitPolsRecursive2 cmPols(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);

    ///////////
    // 1.- Calculate p_cm1_2ns
    ///////////
    TimerStart(STARK_RECURSIVE_2_STEP_1);

    Goldilocks::Element *mem = (Goldilocks::Element *)pAddress;

    TimerStart(STARK_RECURSIVE_2_STEP_1_LDE_AND_MERKLETREE);

    Goldilocks::Element *p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];
    Goldilocks::Element *p_cm1_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm1_2ns]];
    Goldilocks::Element *p_cm1_n = &mem[starkInfo.mapOffsets.section[eSection::cm1_n]];
    TimerStart(STARK_RECURSIVE_2_STEP_1_LDE);
    ntt.extendPol(p_cm1_2ns, p_cm1_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm1_n], p_q_2ns);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_1_LDE);
    TimerStart(STARK_RECURSIVE_2_STEP_1_MERKLETREE);
    Polinomial root0(HASH_SIZE, 1);
    treesGL[0] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN1.section[eSection::cm1_n] + starkInfo.mapSectionsN3.section[eSection::cm1_n] * FIELD_EXTENSION, p_cm1_2ns);
    treesGL[0]->merkelize();
    treesGL[0]->getRoot(root0.address());
    std::cout << "MerkleTree rootGL 1: [ " << root0.toString(4) << " ]" << std::endl;
    transcript.put(root0.address(), HASH_SIZE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_1_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_1_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_1);

    ///////////
    // 2.- Caluculate plookups h1 and h2
    ///////////
    TimerStart(STARK_RECURSIVE_2_STEP_2);
    transcript.getField(challenges[0]); // u
    transcript.getField(challenges[1]); // defVal

    TimerStart(STARK_RECURSIVE_2_STEP_2_CALCULATE_EXPS);

    step2prev_first(mem, &publicInputs[0], 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step2prev_first(mem, &publicInputs[0], i);
        // CalculateExpsAll::step2prev_i(mem, const_n, (Goldilocks3::Element *)challenges.address(), i);
    }
    // CalculateExpsAll::step2prev_last(mem, const_n, (Goldilocks3::Element *)challenges.address(), N - 1);
    step2prev_first(mem, &publicInputs[0], N - 1);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_2_CALCULATE_EXPS);
    TimerStart(STARK_RECURSIVE_2_STEP_2_CALCULATEH1H2);
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
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_2_CALCULATEH1H2);

    TimerStart(STARK_RECURSIVE_2_STEP_2_LDE_AND_MERKLETREE);
    Goldilocks::Element *p_cm2_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm2_2ns]];
    Goldilocks::Element *p_cm2_n = &mem[starkInfo.mapOffsets.section[eSection::cm2_n]];
    TimerStart(STARK_RECURSIVE_2_STEP_2_LDE);
    ntt.extendPol(p_cm2_2ns, p_cm2_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm2_n], p_q_2ns);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_2_LDE);
    TimerStart(STARK_RECURSIVE_2_STEP_2_MERKLETREE);
    Polinomial root1(HASH_SIZE, 1);
    treesGL[1] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN.section[eSection::cm2_n], p_cm2_2ns);
    treesGL[1]->merkelize();
    treesGL[1]->getRoot(root1.address());
    std::cout << "MerkleTree rootGL 1: [ " << root1.toString(4) << " ]" << std::endl;
    transcript.put(root1.address(), HASH_SIZE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_2_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_2_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_2);

    ///////////
    // 3.- Compute Z polynomials
    ///////////

    TimerStart(STARK_RECURSIVE_2_STEP_3);
    transcript.getField(challenges[2]); // gamma
    transcript.getField(challenges[3]); // betta

    TimerStart(STARK_RECURSIVE_2_STEP_3_CALCULATE_EXPS);
    step3prev_first(mem, &publicInputs[0], 0);
#pragma omp parallel for
    for (uint64_t i = 1; i < N - 1; i++)
    {
        step3prev_first(mem, &publicInputs[0], i);
        // CalculateExpsAll::step3prev_i((Goldilocks::Element *)pAddress, const_n, challenges, x_n, i);
    }
    step3prev_first(mem, &publicInputs[0], N - 1);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_3_CALCULATE_EXPS);
    TimerStart(STARK_RECURSIVE_2_STEP_3_CALCULATE_Z);

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
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_3_CALCULATE_Z);

    TimerStart(STARK_RECURSIVE_2_STEP_3_LDE_AND_MERKLETREE);
    TimerStart(STARK_RECURSIVE_2_STEP_3_LDE);

    Goldilocks::Element *p_cm3_2ns = &mem[starkInfo.mapOffsets.section[eSection::cm3_2ns]];
    Goldilocks::Element *p_cm3_n = &mem[starkInfo.mapOffsets.section[eSection::cm3_n]];
    ntt.extendPol(p_cm3_2ns, p_cm3_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::cm3_n], p_q_2ns);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_3_LDE);
    TimerStart(STARK_RECURSIVE_2_STEP_3_MERKLETREE);
    Polinomial root2(HASH_SIZE, 1);
    treesGL[2] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN.section[eSection::cm3_n], p_cm3_2ns);
    treesGL[2]->merkelize();
    treesGL[2]->getRoot(root2.address());
    std::cout << "MerkleTree rootGL 2: [ " << root2.toString(4) << " ]" << std::endl;
    transcript.put(root2.address(), HASH_SIZE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_3_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_3_LDE_AND_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_3);

    ///////////
    // 4. Compute C Polynomial
    ///////////

    TimerStart(STARK_RECURSIVE_2_STEP_4);
    TimerStart(STARK_RECURSIVE_2_STEP_4_CALCULATE_EXPS);

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
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_4_CALCULATE_EXPS);
    TimerStart(STARK_RECURSIVE_2_STEP_4_LDE);
    Goldilocks::Element *p_exps_withq_2ns = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_2ns]];
    Goldilocks::Element *p_exps_withq_n = &mem[starkInfo.mapOffsets.section[eSection::exps_withq_n]];
    ntt.extendPol(p_exps_withq_2ns, p_exps_withq_n, NExtended, N, starkInfo.mapSectionsN.section[eSection::exps_withq_n], p_q_2ns);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_4_LDE);
    TimerStart(STARK_RECURSIVE_2_STEP_4_CALCULATE_EXPS_2NS);
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

    TimerStopAndLog(STARK_RECURSIVE_2_STEP_4_CALCULATE_EXPS_2NS);
    TimerStart(STARK_RECURSIVE_2_STEP_4_MERKLETREE);
    Polinomial root3(HASH_SIZE, 1);
    treesGL[3] = new MerkleTreeGL(NExtended, starkInfo.mapSectionsN.section[eSection::q_2ns], p_q_2ns);
    treesGL[3]->merkelize();
    treesGL[3]->getRoot(root3.address());
    std::cout << "MerkleTree rootGL 3: [ " << root3.toString(4) << " ]" << std::endl;
    transcript.put(root3.address(), HASH_SIZE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_4_MERKLETREE);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_4);

    ///////////
    // 5. Compute FRI Polynomial
    ///////////
    TimerStart(STARK_RECURSIVE_2_STEP_5);
    TimerStart(STARK_RECURSIVE_2_STEP_5_LEv_LpEv);

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
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_5_LEv_LpEv);

    TimerStart(STARK_RECURSIVE_2_STEP_5_EVMAP);
    /* sort polinomials depending on its type

        Subsets:
            0. const
            1. cm , dim=1
            2. qs , dim=1  //1 and 2 to be joined
            3. cm , dim=3
            4. qs, dim=3   //3 and 4 to be joined
     */

    uint64_t size_eval = starkInfo.evMap.size();
    uint64_t *sorted_evMap = (uint64_t *)malloc(5 * size_eval * sizeof(uint64_t));
    uint64_t counters[5] = {0, 0, 0, 0, 0};

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
    uint64_t offset1 = size_eval + counters[1];
    uint64_t offset2 = 2 * size_eval;
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
    uint64_t num_threads = omp_get_max_threads();
    Goldilocks::Element **evals_acc = (Goldilocks::Element **)malloc(num_threads * sizeof(Goldilocks::Element *));
    for (uint64_t i = 0; i < num_threads; ++i)
    {
        evals_acc[i] = (Goldilocks::Element *)malloc(size_eval * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }

#pragma omp parallel
    {
        uint64_t thread_idx = omp_get_thread_num();
        for (uint64_t i = 0; i < size_eval * FIELD_EXTENSION; ++i)
        {
            evals_acc[thread_idx][i] = Goldilocks::zero();
        }

#pragma omp for
        for (uint64_t k = 0; k < N; k++)
        {
            for (uint64_t i = 0; i < counters[0]; i++)
            {
                uint64_t indx = sorted_evMap[i];
                EvMap ev = starkInfo.evMap[indx];
                Polinomial tmp(1, FIELD_EXTENSION);
                Polinomial acc(1, FIELD_EXTENSION);

                Polinomial p(&((Goldilocks::Element *)pConstPols2ns->address())[ev.id], pConstPols2ns->degree(), 1, pConstPols2ns->numPols());

                Polinomial::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                for (uint64_t j = 0; j < FIELD_EXTENSION; ++j)
                {
                    evals_acc[thread_idx][indx * FIELD_EXTENSION + j] = evals_acc[thread_idx][indx * FIELD_EXTENSION + j] + tmp[0][j];
                }
            }
            for (uint64_t i = 0; i < counters[1]; i++)
            {
                uint64_t indx = sorted_evMap[size_eval + i];
                EvMap ev = starkInfo.evMap[indx];
                Polinomial tmp(1, FIELD_EXTENSION);

                Polinomial p;
                p = (ev.type == EvMap::eType::cm) ? starkInfo.getPolinomial(mem, starkInfo.cm_2ns[ev.id]) : starkInfo.getPolinomial(mem, starkInfo.qs[ev.id]);

                Polinomial ::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                for (uint64_t j = 0; j < FIELD_EXTENSION; ++j)
                {
                    evals_acc[thread_idx][indx * FIELD_EXTENSION + j] = evals_acc[thread_idx][indx * FIELD_EXTENSION + j] + tmp[0][j];
                }
            }
            for (uint64_t i = 0; i < counters[3]; i++)
            {
                uint64_t indx = sorted_evMap[3 * size_eval + i];
                EvMap ev = starkInfo.evMap[indx];
                Polinomial tmp(1, FIELD_EXTENSION);

                Polinomial p;
                p = (ev.type == EvMap::eType::cm) ? starkInfo.getPolinomial(mem, starkInfo.cm_2ns[ev.id]) : starkInfo.getPolinomial(mem, starkInfo.qs[ev.id]);

                Polinomial ::mulElement(tmp, 0, ev.prime ? LpEv : LEv, k, p, k << extendBits);
                for (uint64_t j = 0; j < FIELD_EXTENSION; ++j)
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
            uint64_t offset = i * FIELD_EXTENSION;
            for (uint64_t k = 0; k < num_threads; ++k)
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
    for (uint64_t i = 0; i < num_threads; ++i)
    {
        free(evals_acc[i]);
    }
    free(evals_acc);

    TimerStopAndLog(STARK_RECURSIVE_2_STEP_5_EVMAP);
    TimerStart(STARK_RECURSIVE_2_STEP_5_XDIVXSUB);

    // Calculate xDivXSubXi, xDivXSubWXi
    Polinomial xi(1, FIELD_EXTENSION);
    Polinomial wxi(1, FIELD_EXTENSION);

    Polinomial::copyElement(xi, 0, challenges, 7);
    Polinomial::mulElement(wxi, 0, challenges, 7, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits));

    Polinomial x(1, FIELD_EXTENSION); // [0,0,0]
    *x[0] = Goldilocks::shift();      // [49,0,0]

    for (uint64_t k = 0; k < (N << extendBits); k++)
    {
        // xDivXSubXi[k] = x - xi
        // xDivXSubWXi[k] = x - wxi
        // x = x * w[starkInfo.starkStruct.nBits + extendBits]
        Polinomial::subElement(xDivXSubXi, k, x, 0, xi, 0);
        Polinomial::subElement(xDivXSubWXi, k, x, 0, wxi, 0);
        Polinomial::mulElement(x, 0, x, 0, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
    }

    Polinomial::batchInverse(xDivXSubXi, xDivXSubXi);
    Polinomial::batchInverse(xDivXSubWXi, xDivXSubWXi);

    Polinomial x1(1, FIELD_EXTENSION); // [0,0,0]
    *x1[0] = Goldilocks::shift();      // [49,0,0]

    for (uint64_t k = 0; k < (N << extendBits); k++)
    {
        Polinomial::mulElement(xDivXSubXi, k, xDivXSubXi, k, x1, 0);
        Polinomial::mulElement(xDivXSubWXi, k, xDivXSubWXi, k, x1, 0);
        Polinomial::mulElement(x1, 0, x1, 0, (Goldilocks::Element &)Goldilocks::w(starkInfo.starkStruct.nBits + extendBits));
    }
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_5_XDIVXSUB);
    TimerStart(STARK_RECURSIVE_2_STEP_5_CALCULATE_EXPS);

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
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_5_CALCULATE_EXPS);
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_5);
    TimerStart(STARK_RECURSIVE_2_STEP_FRI);

    treesGL[4] = new MerkleTreeGL((Goldilocks::Element *)pConstTreeAddress);

    Polinomial friPol = starkInfo.getPolinomial(mem, starkInfo.exps_2ns[starkInfo.friExpId]);
    FRIProve::prove(proof, treesGL, transcript, friPol, starkInfo.starkStruct.nBitsExt, starkInfo);

    proof.proofs.setEvals(evals.address());

    std::memcpy(&proof.proofs.root1[0], root0.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root2[0], root1.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root3[0], root2.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    std::memcpy(&proof.proofs.root4[0], root3.address(), HASH_SIZE * sizeof(Goldilocks::Element));
    TimerStopAndLog(STARK_RECURSIVE_2_STEP_FRI);
}
