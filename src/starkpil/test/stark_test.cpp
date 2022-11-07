#include "stark_test.hpp"
#include "starkMock.hpp"
#include <algorithm> // std::min
#include "scalar.hpp"
#include <openssl/sha.h>

#define NUM_CHALLENGES 8

void StarkTest(void)
{
    // Load config & test vectors
    Config config;
    config.starkInfoFile = starkInfo_File;
    config.constPolsFile = constant_file;
    config.mapConstPolsFile = false;
    config.runProverServer = true;
    config.mapConstantsTreeFile = false;
    config.constantsTreeFile = constant_tree_file;
    config.witnessFile = "basic.witness.wtns";
    config.verifierFile = "basic.verifier.dat";
    config.execC12aFile = "basic.c12a.exec";
    config.execC12bFile = "basic.c12b.exec";
    config.starkInfoC12aFile = "basic.c12a.starkinfo.json";
    config.starkInfoC12bFile = "basic.c12b.starkinfo.json";

    config.constPolsC12aFile = "basic.c12a.const";
    config.constPolsC12bFile = "basic.c12b.const";
    config.constantsTreeC12aFile = "basic.c12a.consttree";
    config.constantsTreeC12bFile = "basic.c12b.consttree";
    config.starkVerifierFile = "basic.g16.0001.zkey";
    config.starkZkInC12a = "basic.c12a.zkin.proof.json";
    config.starkZkInC12b = "basic.c12b.zkin.proof.json";
    config.publicStarkFile = "basic.public.json";
    config.starkZkIn = "basic.zkin.proof.json";

    std::unique_ptr<Groth16::Prover<AltBn128::Engine>> groth16Prover;
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    std::unique_ptr<ZKeyUtils::Header> zkeyHeader;
    mpz_t altBbn128r;
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);
    json publicJson = json::array();

    mpz_t address;
    mpz_t publicshash;
    json publicStarkJson;
    RawFr::Element publicsHash;
    string freeInStrings16[8];

    zkey = BinFileUtils::openExisting(config.starkVerifierFile, "zkey", 1);
    zkeyHeader = ZKeyUtils::loadHeader(zkey.get());

    if (mpz_cmp(zkeyHeader->rPrime, altBbn128r) != 0)
    {
        throw std::invalid_argument("zkey curve not supported");
    }

    groth16Prover = Groth16::makeProver<AltBn128::Engine>(
        zkeyHeader->nVars,
        zkeyHeader->nPublic,
        zkeyHeader->domainSize,
        zkeyHeader->nCoefs,
        zkeyHeader->vk_alpha1,
        zkeyHeader->vk_beta1,
        zkeyHeader->vk_beta2,
        zkeyHeader->vk_delta1,
        zkeyHeader->vk_delta2,
        zkey->getSectionData(4), // Coefs
        zkey->getSectionData(5), // pointsA
        zkey->getSectionData(6), // pointsB1
        zkey->getSectionData(7), // pointsB2
        zkey->getSectionData(8), // pointsC
        zkey->getSectionData(9)  // pointsH1
    );

    Goldilocks::Element publics[43];

    publicStarkJson[0] = "3248459814";
    publicStarkJson[1] = "1620587195";
    publicStarkJson[2] = "3678822139";
    publicStarkJson[3] = "1824295850";
    publicStarkJson[4] = "366027599";
    publicStarkJson[5] = "1355324045";
    publicStarkJson[6] = "1531026716";
    publicStarkJson[7] = "1017354875";
    publicStarkJson[8] = "0";
    publicStarkJson[9] = "0";
    publicStarkJson[10] = "0";
    publicStarkJson[11] = "0";
    publicStarkJson[12] = "0";
    publicStarkJson[13] = "0";
    publicStarkJson[14] = "0";
    publicStarkJson[15] = "0";
    publicStarkJson[16] = "0";
    publicStarkJson[17] = "1000";
    publicStarkJson[18] = "510351649";
    publicStarkJson[19] = "2243740642";
    publicStarkJson[20] = "121390774";
    publicStarkJson[21] = "3088140970";
    publicStarkJson[22] = "2387924872";
    publicStarkJson[23] = "2930644697";
    publicStarkJson[24] = "923028121";
    publicStarkJson[25] = "2301051566";
    publicStarkJson[26] = "537003291";
    publicStarkJson[27] = "344094503";
    publicStarkJson[28] = "251860201";
    publicStarkJson[29] = "686198245";
    publicStarkJson[30] = "3667240819";
    publicStarkJson[31] = "1437754387";
    publicStarkJson[32] = "2701071742";
    publicStarkJson[33] = "568001667";
    publicStarkJson[34] = "0";
    publicStarkJson[35] = "0";
    publicStarkJson[36] = "0";
    publicStarkJson[37] = "0";
    publicStarkJson[38] = "0";
    publicStarkJson[39] = "0";
    publicStarkJson[40] = "0";
    publicStarkJson[41] = "0";
    publicStarkJson[42] = "1";

    publics[0] = Goldilocks::fromString("3248459814");
    publics[1] = Goldilocks::fromString("1620587195");
    publics[2] = Goldilocks::fromString("3678822139");
    publics[3] = Goldilocks::fromString("1824295850");
    publics[4] = Goldilocks::fromString("366027599");
    publics[5] = Goldilocks::fromString("1355324045");
    publics[6] = Goldilocks::fromString("1531026716");
    publics[7] = Goldilocks::fromString("1017354875");
    publics[8] = Goldilocks::fromString("0");
    publics[9] = Goldilocks::fromString("0");
    publics[10] = Goldilocks::fromString("0");
    publics[11] = Goldilocks::fromString("0");
    publics[12] = Goldilocks::fromString("0");
    publics[13] = Goldilocks::fromString("0");
    publics[14] = Goldilocks::fromString("0");
    publics[15] = Goldilocks::fromString("0");
    publics[16] = Goldilocks::fromString("0");
    publics[17] = Goldilocks::fromString("1000");
    publics[18] = Goldilocks::fromString("510351649");
    publics[19] = Goldilocks::fromString("2243740642");
    publics[20] = Goldilocks::fromString("121390774");
    publics[21] = Goldilocks::fromString("3088140970");
    publics[22] = Goldilocks::fromString("2387924872");
    publics[23] = Goldilocks::fromString("2930644697");
    publics[24] = Goldilocks::fromString("923028121");
    publics[25] = Goldilocks::fromString("2301051566");
    publics[26] = Goldilocks::fromString("537003291");
    publics[27] = Goldilocks::fromString("344094503");
    publics[28] = Goldilocks::fromString("251860201");
    publics[29] = Goldilocks::fromString("686198245");
    publics[30] = Goldilocks::fromString("3667240819");
    publics[31] = Goldilocks::fromString("1437754387");
    publics[32] = Goldilocks::fromString("2701071742");
    publics[33] = Goldilocks::fromString("568001667");
    publics[34] = Goldilocks::fromString("0");
    publics[35] = Goldilocks::fromString("0");
    publics[36] = Goldilocks::fromString("0");
    publics[37] = Goldilocks::fromString("0");
    publics[38] = Goldilocks::fromString("0");
    publics[39] = Goldilocks::fromString("0");
    publics[40] = Goldilocks::fromString("0");
    publics[41] = Goldilocks::fromString("0");
    publics[42] = Goldilocks::fromString("1");

    mpz_init_set_str(address, "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266", 0);
    std::string strAddress = mpz_get_str(0, 16, address);
    std::string strAddress10 = mpz_get_str(0, 10, address);

    std::string buffer = "";
    buffer = buffer + std::string(40 - std::min(40, (int)strAddress.length()), '0') + strAddress;

    std::string aux;
    for (uint i = 0; i < 8; i++)
    {
        buffer = buffer + std::string(16 - std::min(16, (int)freeInStrings16[i].length()), '0') + freeInStrings16[i];
    }

    mpz_init_set_str(publicshash, sha256(buffer).c_str(), 16);
    std::string publicsHashString = mpz_get_str(0, 10, publicshash);
    RawFr::field.fromString(publicsHash, publicsHashString);

    // Save public file
    RawFr::Element input;
    RawFr::field.fromString(input, "2043100198");

    publicJson[0] = "14918438705377636817563619860509474434188349281706594260803853913155748736842";
    json2file(publicJson, config.publicStarkFile);

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
    MockCircom::Circom_Circuit *circuit = MockCircom::loadCircuit(config.verifierFile);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);

    TimerStart(CIRCOM_LOAD_JSON);
    MockCircom::Circom_CalcWit *ctx = new MockCircom::Circom_CalcWit(circuit);
    MockCircom::loadJsonImpl(ctx, zkin);
    if (ctx->getRemaingInputsToBeSet() != 0)
    {
        cerr << "Error: Not all inputs have been set. Only " << MockCircom::get_main_input_signal_no() - ctx->getRemaingInputsToBeSet() << " out of " << MockCircom::get_main_input_signal_no() << endl;
        exitProcess();
    }
    TimerStopAndLog(CIRCOM_LOAD_JSON);

    // If present, save witness file
    if (config.witnessFile.size() > 0)
    {
        TimerStart(CIRCOM_WRITE_BIN_WITNESS);
        MockCircom::writeBinWitness(ctx, config.witnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
        TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
    }

    /*****************************************/
    /* Compute witness and c12a commited pols */
    /*****************************************/
    TimerStart(C12_WITNESS_AND_COMMITED_POLS);

    ExecFile execC12aFile(config.execC12aFile);
    uint64_t sizeWitness = MockCircom::get_size_of_witness();
    Goldilocks::Element *tmp = new Goldilocks::Element[execC12aFile.nAdds + sizeWitness];

#pragma omp parallel for
    for (uint64_t i = 0; i < sizeWitness; i++)
    {
        FrGElement aux;
        ctx->getWitness(i, &aux);
        FrG_toLongNormal(&aux, &aux);
        tmp[i] = Goldilocks::fromU64(aux.longVal[0]);
    }

    for (uint64_t i = 0; i < execC12aFile.nAdds; i++)
    {
        FrG_toLongNormal(&execC12aFile.p_adds[i * 4], &execC12aFile.p_adds[i * 4]);
        FrG_toLongNormal(&execC12aFile.p_adds[i * 4 + 1], &execC12aFile.p_adds[i * 4 + 1]);
        FrG_toLongNormal(&execC12aFile.p_adds[i * 4 + 2], &execC12aFile.p_adds[i * 4 + 2]);
        FrG_toLongNormal(&execC12aFile.p_adds[i * 4 + 3], &execC12aFile.p_adds[i * 4 + 3]);

        uint64_t idx_1 = execC12aFile.p_adds[i * 4].longVal[0];
        uint64_t idx_2 = execC12aFile.p_adds[i * 4 + 1].longVal[0];

        Goldilocks::Element c = tmp[idx_1] * Goldilocks::fromU64(execC12aFile.p_adds[i * 4 + 2].longVal[0]);
        Goldilocks::Element d = tmp[idx_2] * Goldilocks::fromU64(execC12aFile.p_adds[i * 4 + 3].longVal[0]);
        tmp[sizeWitness + i] = c + d;
    }

    uint64_t Nbits = log2(execC12aFile.nSMap - 1) + 1;
    uint64_t N = 1 << Nbits;

    StarkInfo starkInfoC12a(config, config.starkInfoC12aFile);
    StarkC12aMock starkC12a(config);
    uint64_t polsSizeC12a = starkC12a.getTotalPolsSize();

    void *pAddressC12a = calloc(polsSizeC12a, 1);
    CommitPolsBasicC12a cmPols12(pAddressC12a, CommitPolsBasicC12a::pilDegree());

#pragma omp parallel for
    for (uint i = 0; i < execC12aFile.nSMap; i++)
    {
        for (uint j = 0; j < 12; j++)
        {
            FrGElement aux;
            FrG_toLongNormal(&aux, &execC12aFile.p_sMap[12 * i + j]);
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
    for (uint i = execC12aFile.nSMap; i < N; i++)
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
    uint64_t polBitsC12 = starkInfoC12a.starkStruct.steps[starkInfoC12a.starkStruct.steps.size() - 1].nBits;
    FRIProof fproof_c12a((1 << polBitsC12), FIELD_EXTENSION, starkInfoC12a.starkStruct.steps.size(), starkInfoC12a.evMap.size(), starkInfoC12a.nPublics);

    // Generate the proof
    starkC12a.genProof(pAddressC12a, fproof_c12a, publics);
    TimerStopAndLog(STARK_C12_PROOF);

    nlohmann::ordered_json jProofC12a = fproof_c12a.proofs.proof2json();
    nlohmann::ordered_json zkinC12a = proof2zkinStark(jProofC12a);
    zkinC12a["publics"] = publicStarkJson;
    ofstream ofzkin2(config.starkZkInC12a);
    ofzkin2 << setw(4) << zkinC12a.dump() << endl;
    ofzkin2.close();

    jProofC12a["publics"] = publicStarkJson;
    ofstream ofstark2(starkFileC12a);
    ofstark2 << setw(4) << jProofC12a.dump() << endl;
    ofstark.close();

    return;
    /************/
    /* Verifier */
    /************/
    TimerStart(CIRCOM_LOAD_CIRCUIT_C12a);
    MockCircomC12a::Circom_Circuit *circuitC12a = MockCircomC12a::loadCircuit("basic.c12a.verifier.dat");
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_C12a);

    TimerStart(CIRCOM_C12a_LOAD_JSON);

    MockCircomC12a::Circom_CalcWit *ctxC12a = new MockCircomC12a::Circom_CalcWit(circuitC12a);
    json zkinC12ajson = json::parse(zkinC12a.dump().c_str());

    MockCircomC12a::loadJsonImpl(ctxC12a, zkinC12ajson);
    if (ctxC12a->getRemaingInputsToBeSet() != 0)
    {
        cerr << "Error: Not all inputs have been set. Only " << MockCircomC12a::get_main_input_signal_no() - ctxC12a->getRemaingInputsToBeSet() << " out of " << MockCircomC12a::get_main_input_signal_no() << endl;
        exitProcess();
    }
    TimerStopAndLog(CIRCOM_C12a_LOAD_JSON);

    // If present, save witness file
    if (config.witnessFile.size() > 0)
    {
        TimerStart(CIRCOM_WRITE_BIN_WITNESS);
        MockCircomC12a::writeBinWitness(ctxC12a, "basic.c12a.witness.wtns"); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
        TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
    }

    /******************************************/
    /* Compute witness and C12b commited pols */
    /******************************************/
    TimerStart(C12b_WITNESS_AND_COMMITED_POLS);

    ExecFile execC12bFile(config.execC12bFile);
    uint64_t sizeWitnessc12a = MockCircomC12a::get_size_of_witness();
    Goldilocks::Element *tmpc12a = new Goldilocks::Element[execC12bFile.nAdds + sizeWitnessc12a];

#pragma omp parallel for
    for (uint64_t i = 0; i < sizeWitnessc12a; i++)
    {
        FrGElement aux;
        ctxC12a->getWitness(i, &aux);
        FrG_toLongNormal(&aux, &aux);
        tmpc12a[i] = Goldilocks::fromU64(aux.longVal[0]);
    }

    for (uint64_t i = 0; i < execC12bFile.nAdds; i++)
    {
        FrG_toLongNormal(&execC12bFile.p_adds[i * 4], &execC12bFile.p_adds[i * 4]);
        FrG_toLongNormal(&execC12bFile.p_adds[i * 4 + 1], &execC12bFile.p_adds[i * 4 + 1]);
        FrG_toLongNormal(&execC12bFile.p_adds[i * 4 + 2], &execC12bFile.p_adds[i * 4 + 2]);
        FrG_toLongNormal(&execC12bFile.p_adds[i * 4 + 3], &execC12bFile.p_adds[i * 4 + 3]);

        uint64_t idx_1 = execC12bFile.p_adds[i * 4].longVal[0];
        uint64_t idx_2 = execC12bFile.p_adds[i * 4 + 1].longVal[0];

        Goldilocks::Element c = tmpc12a[idx_1] * Goldilocks::fromU64(execC12bFile.p_adds[i * 4 + 2].longVal[0]);
        Goldilocks::Element d = tmpc12a[idx_2] * Goldilocks::fromU64(execC12bFile.p_adds[i * 4 + 3].longVal[0]);
        tmpc12a[sizeWitnessc12a + i] = c + d;
    }

    uint64_t NbitsC12a = log2(execC12bFile.nSMap - 1) + 1;
    uint64_t NC12a = 1 << NbitsC12a;

    StarkInfo starkInfoC12b(config, config.starkInfoC12bFile);
    StarkC12bMock starkC12b(config);
    uint64_t polsSizeC12b = starkC12b.getTotalPolsSize();

    void *pAddressC12b = calloc(polsSizeC12b, 1);
    CommitPolsBasicC12b cmPols12b(pAddressC12b, CommitPolsBasicC12b::pilDegree());

#pragma omp parallel for
    for (uint i = 0; i < execC12bFile.nSMap; i++)
    {
        for (uint j = 0; j < 12; j++)
        {
            FrGElement aux;
            FrG_toLongNormal(&aux, &execC12bFile.p_sMap[12 * i + j]);
            uint64_t idx_1 = aux.longVal[0];
            if (idx_1 != 0)
            {
                cmPols12b.Compressor.a[j][i] = tmpc12a[idx_1];
            }
            else
            {
                cmPols12b.Compressor.a[j][i] = Goldilocks::zero();
            }
        }
    }
    for (uint i = execC12bFile.nSMap; i < NC12a; i++)
    {
        for (uint j = 0; j < 12; j++)
        {
            cmPols12b.Compressor.a[j][i] = Goldilocks::zero();
        }
    }
    delete[] tmpc12a;
    TimerStopAndLog(C12b_WITNESS_AND_COMMITED_POLS);

    /*****************************************/
    /* Generate C12 stark proof              */
    /*****************************************/
    TimerStart(STARK_C12b_PROOF);
    uint64_t polBitsC12b = starkInfoC12b.starkStruct.steps[starkInfoC12b.starkStruct.steps.size() - 1].nBits;
    FRIProofC12 fproof_c12b((1 << polBitsC12b), FIELD_EXTENSION, starkInfoC12b.starkStruct.steps.size(), starkInfoC12b.evMap.size(), starkInfoC12b.nPublics);

    // Generate the proof
    starkC12b.genProof(pAddressC12b, fproof_c12b, publics);
    TimerStopAndLog(STARK_C12b_PROOF);

    nlohmann::ordered_json jProofC12b = fproof_c12b.proofs.proof2json();
    nlohmann::ordered_json zkinC12b = proof2zkinStark(jProofC12b);
    zkinC12b["publics"] = publicStarkJson;
    zkinC12b["proverAddr"] = strAddress10;
    ofstream ofzkin2b(config.starkZkInC12b);
    ofzkin2b << setw(4) << zkinC12b.dump() << endl;
    ofzkin2b.close();

    /************/
    /* Verifier */
    /************/
    TimerStart(CIRCOM_LOAD_CIRCUIT_C12b);
    MockCircomC12b::Circom_Circuit *circuitC12b = MockCircomC12b::loadCircuit("basic.c12b.verifier.dat");
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_C12b);

    TimerStart(CIRCOM_C12b_LOAD_JSON);
    MockCircomC12b::Circom_CalcWit *ctxC12b = new MockCircomC12b::Circom_CalcWit(circuitC12b);
    json zkinC12bjson = json::parse(zkinC12b.dump().c_str());

    MockCircomC12b::loadJsonImpl(ctxC12b, zkinC12bjson);
    if (ctxC12b->getRemaingInputsToBeSet() != 0)
    {
        cerr << "Error: Not all inputs have been set. Only " << MockCircomC12b::get_main_input_signal_no() - ctxC12b->getRemaingInputsToBeSet() << " out of " << MockCircomC12b::get_main_input_signal_no() << endl;
        exitProcess();
    }
    TimerStopAndLog(CIRCOM_C12b_LOAD_JSON);

    // If present, save witness file
    if (config.witnessFile.size() > 0)
    {
        TimerStart(CIRCOM_WRITE_BIN_WITNESS);
        MockCircomC12b::writeBinWitness(ctxC12b, "basic.c12b.witness.wtns"); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
        TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
    }
    TimerStart(CIRCOM_GET_BIN_WITNESS);
    AltBn128::FrElement *pWitnessC12b = NULL;
    uint64_t witnessSizeb = 0;
    MockCircomC12b::getBinWitness(ctxC12b, pWitnessC12b, witnessSizeb);
    TimerStopAndLog(CIRCOM_GET_BIN_WITNESS);

    // Generate Groth16 via rapid SNARK
    TimerStart(RAPID_SNARK);
    json jsonProof;
    try
    {
        auto proof = groth16Prover->prove(pWitnessC12b);
        jsonProof = proof->toJson();
    }
    catch (std::exception &e)
    {
        cerr << "Error: Prover::Prove() got exception in rapid SNARK:" << e.what() << '\n';
        exitProcess();
    }
    TimerStopAndLog(RAPID_SNARK);

    // Save proof.json to disk
    json2file(jsonProof, "basic.proof.json");

    /***********/
    /* Cleanup */
    /***********/
    delete ctx;
    delete ctxC12a;
    MockCircom::freeCircuit(circuit);
    MockCircomC12a::freeCircuit(circuitC12a);

    free(pAddressC12a);
    free(pAddressC12b);
    free(pWitnessC12b);
}