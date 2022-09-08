#include "stark_test.hpp"
#include "starkMock.hpp"

#define NUM_CHALLENGES 8

void StarkTest(void)
{
    json publicStarkJson;
    Goldilocks::Element publics[8];
    
    publicStarkJson[0] = "2043100198";
    publicStarkJson[1] = "2909753411";
    publicStarkJson[2] = "2146825699";
    publicStarkJson[3] = "3866023039";
    publicStarkJson[4] = "1719628537";
    publicStarkJson[5] = "3739677152";
    publicStarkJson[6] = "1596594856";
    publicStarkJson[7] = "3497182697";

    publics[0] = Goldilocks::fromString("2043100198");
    publics[1] = Goldilocks::fromString("2909753411");
    publics[2] = Goldilocks::fromString("2146825699");
    publics[3] = Goldilocks::fromString("3866023039");
    publics[4] = Goldilocks::fromString("1719628537");
    publics[5] = Goldilocks::fromString("3739677152");
    publics[6] = Goldilocks::fromString("1596594856");
    publics[7] = Goldilocks::fromString("3497182697");

    // Load config & test vectors
    Config config;
    config.starkInfoFile = starkInfo_File;
    config.constPolsFile = constant_file;
    config.mapConstPolsFile = false;
    config.runProverServer = true;
    config.constantsTreeFile = constant_tree_file;
    config.witnessFile = "basic.witness.wtns";
    config.verifierFile = "basic.verifier.dat";
    config.execFile = "basic.c12.exec";
    config.starkInfoC12File = "basic.c12.starkinfo.json";
    config.constPolsC12File = "basic.c12.const";
    config.constantsTreeC12File = "basic.c12.consttree";
    // config.starkFile = starkFile;
    StarkInfo starkInfo(config, config.starkInfoFile);
    StarkMock stark(config);

    void *pAddress = malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    void *pCommitedAddress = mapFile(commited_file, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element), false);
    std::memcpy(pAddress, pCommitedAddress, starkInfo.nCm1 * starkInfo.mapDeg.section[eSection::cm1_n] * sizeof(Goldilocks::Element));

    CommitPolsBasic cmP(pAddress, starkInfo.mapDeg.section[eSection::cm1_n]);

    void *pConstantAddress = NULL;
    pConstantAddress = mapFile(constant_file, starkInfo.nConstants * (1 << starkInfo.starkStruct.nBits) * sizeof(Goldilocks::Element), false);
    ConstantPolsBasic const_n(pConstantAddress, (1 << starkInfo.starkStruct.nBits));

    /*************************************/
    /*  Generate  stark proof            */
    /*************************************/
    TimerStart(STARK_PROOF);
    uint64_t polBits = starkInfo.starkStruct.steps[starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkInfo.starkStruct.steps.size(), starkInfo.evMap.size(), starkInfo.nPublics);
    stark.genProof(pAddress, fproof);
    TimerStopAndLog(STARK_PROOF);

    TimerStart(STARK_JSON_GENERATION);

    nlohmann::ordered_json jProof = fproof.proofs.proof2json();
    jProof["publics"] = publicStarkJson;
    ofstream ofstark(starkFile);
    ofstark << setw(4) << jProof.dump() << endl;
    ofstark.close();

    nlohmann::json zkin = proof2zkinStark(jProof);
    zkin["publics"] = publicStarkJson;
    ofstream ofzkin(config.starkZkIn);
    ofzkin << setw(4) << zkin.dump() << endl;
    ofzkin.close();

    TimerStopAndLog(STARK_JSON_GENERATION);

    /************/
    /* Verifier */
    /************/

    TimerStart(CIRCOM_LOAD_CIRCUIT);
    testCircom::Circom_Circuit *circuit = testCircom::loadCircuit(config.verifierFile);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);

    TimerStart(CIRCOM_LOAD_JSON);
    testCircom::Circom_CalcWit *ctx = new testCircom::Circom_CalcWit(circuit);
    testCircom::loadJsonImpl(ctx, zkin);
    if (ctx->getRemaingInputsToBeSet() != 0)
    {
        cerr << "Error: Not all inputs have been set. Only " << testCircom::get_main_input_signal_no() - ctx->getRemaingInputsToBeSet() << " out of " << testCircom::get_main_input_signal_no() << endl;
        exitProcess();
    }
    TimerStopAndLog(CIRCOM_LOAD_JSON);

    // If present, save witness file
    if (config.witnessFile.size() > 0)
    {
        TimerStart(CIRCOM_WRITE_BIN_WITNESS);
        testCircom::writeBinWitness(ctx, config.witnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
        TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
    }

    /*****************************************/
    /* Compute witness and c12 commited pols */
    /*****************************************/
    TimerStart(C12_WITNESS_AND_COMMITED_POLS);

    ExecFile execFile(config.execFile);
    uint64_t sizeWitness = testCircom::get_size_of_witness();
    Goldilocks::Element *tmp = new Goldilocks::Element[execFile.nAdds + sizeWitness];

#pragma omp parallel for
    for (uint64_t i = 0; i < sizeWitness; i++)
    {
        FrGElement aux;
        ctx->getWitness(i, &aux);
        FrG_toLongNormal(&aux, &aux);
        tmp[i] = Goldilocks::fromU64(aux.longVal[0]);
    }

    for (uint64_t i = 0; i < execFile.nAdds; i++)
    {
        FrG_toLongNormal(&execFile.p_adds[i * 4], &execFile.p_adds[i * 4]);
        FrG_toLongNormal(&execFile.p_adds[i * 4 + 1], &execFile.p_adds[i * 4 + 1]);
        FrG_toLongNormal(&execFile.p_adds[i * 4 + 2], &execFile.p_adds[i * 4 + 2]);
        FrG_toLongNormal(&execFile.p_adds[i * 4 + 3], &execFile.p_adds[i * 4 + 3]);

        uint64_t idx_1 = execFile.p_adds[i * 4].longVal[0];
        uint64_t idx_2 = execFile.p_adds[i * 4 + 1].longVal[0];

        Goldilocks::Element c = tmp[idx_1] * Goldilocks::fromU64(execFile.p_adds[i * 4 + 2].longVal[0]);
        Goldilocks::Element d = tmp[idx_2] * Goldilocks::fromU64(execFile.p_adds[i * 4 + 3].longVal[0]);
        tmp[sizeWitness + i] = c + d;
    }

    uint64_t Nbits = log2(execFile.nSMap - 1) + 1;
    uint64_t N = 1 << Nbits;

    StarkInfo starkInfoC12(config, config.starkInfoC12File);
    StarkC12Mock starkC12(config);
    uint64_t polsSizeC12 = starkC12.getTotalPolsSize();

    void *pAddressC12 = calloc(polsSizeC12, 1);
    CommitPolsBasicC12 cmPols12(pAddressC12, CommitPolsBasicC12::pilDegree());

#pragma omp parallel for
    for (uint i = 0; i < execFile.nSMap; i++)
    {
        for (uint j = 0; j < 12; j++)
        {
            FrGElement aux;
            FrG_toLongNormal(&aux, &execFile.p_sMap[12 * i + j]);
            uint64_t idx_1 = aux.longVal[0];
            if (idx_1 != 0)
            {
                cmPols12.Compressor.a[j][i] = tmp[idx_1];
            }
            else
            {
                cmPols12.Compressor.a[j][i] = Goldilocks::zero();
            }
        }
    }
    for (uint i = execFile.nSMap; i < N; i++)
    {
        for (uint j = 0; j < 12; j++)
        {
            cmPols12.Compressor.a[j][i] = Goldilocks::zero();
        }
    }
    delete[] tmp;
    TimerStopAndLog(C12_WITNESS_AND_COMMITED_POLS);

    /*****************************************/
    /* Generate C12 stark proof              */
    /*****************************************/
    TimerStart(STARK_C12_PROOF);
    uint64_t polBitsC12 = starkInfoC12.starkStruct.steps[starkInfoC12.starkStruct.steps.size() - 1].nBits;
    FRIProofC12 fproofC12((1 << polBitsC12), FIELD_EXTENSION, starkInfoC12.starkStruct.steps.size(), starkInfoC12.evMap.size(), starkInfoC12.nPublics);

    // Generate the proof
    starkC12.genProof(pAddressC12, fproofC12, publics);
    TimerStopAndLog(STARK_C12_PROOF);
}