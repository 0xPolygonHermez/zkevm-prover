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
    config.constantsTreeFile = constant_tree_file;
    config.witnessFile = "basic.witness.wtns";
    config.verifierFile = "basic.verifier.dat";
    config.execFile = "basic.c12.exec";
    config.starkInfoC12File = "basic.c12.starkinfo.json";
    config.constPolsC12File = "basic.c12.const";
    config.constantsTreeC12File = "basic.c12.consttree";
    config.starkVerifierFile = "basic.g16.0001.zkey";
    config.starkZkInC12 = "basic.c12.zkin.proof.json";
    config.publicStarkFile = "basic.public.json";

    std::unique_ptr<Groth16::Prover<AltBn128::Engine>> groth16Prover;
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    std::unique_ptr<ZKeyUtils::Header> zkeyHeader;
    mpz_t altBbn128r;
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);
    json publicJson = json::array();
    ;
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
    /* Compute witness and c12 commited pols */
    /*****************************************/
    TimerStart(C12_WITNESS_AND_COMMITED_POLS);

    ExecFile execFile(config.execFile);
    uint64_t sizeWitness = MockCircom::get_size_of_witness();
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

    nlohmann::ordered_json jProofC12 = fproofC12.proofs.proof2json();
    nlohmann::ordered_json zkinC12 = proof2zkinStark(jProofC12);
    zkinC12["publics"] = publicStarkJson;
    zkinC12["proverAddr"] = strAddress10;
    ofstream ofzkin2(config.starkZkInC12);
    ofzkin2 << setw(4) << zkinC12.dump() << endl;
    ofzkin2.close();

    /************/
    /* Verifier */
    /************/
    TimerStart(CIRCOM_LOAD_CIRCUIT_C12);
    MockCircomC12::Circom_Circuit *circuitC12 = MockCircomC12::loadCircuit("basic.c12.verifier.dat");
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT_C12);

    TimerStart(CIRCOM_C12_LOAD_JSON);
    MockCircomC12::Circom_CalcWit *ctxC12 = new MockCircomC12::Circom_CalcWit(circuitC12);
    json zkinC12json = json::parse(zkinC12.dump().c_str());

    MockCircomC12::loadJsonImpl(ctxC12, zkinC12json);
    if (ctxC12->getRemaingInputsToBeSet() != 0)
    {
        cerr << "Error: Not all inputs have been set. Only " << MockCircomC12::get_main_input_signal_no() - ctxC12->getRemaingInputsToBeSet() << " out of " << MockCircomC12::get_main_input_signal_no() << endl;
        exitProcess();
    }
    TimerStopAndLog(CIRCOM_C12_LOAD_JSON);

    // If present, save witness file
    if (config.witnessFile.size() > 0)
    {
        TimerStart(CIRCOM_WRITE_BIN_WITNESS);
        MockCircomC12::writeBinWitness(ctxC12, "basic.c12.witness.wtns"); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
        TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
    }
    TimerStart(CIRCOM_GET_BIN_WITNESS);
    AltBn128::FrElement *pWitnessC12 = NULL;
    uint64_t witnessSize = 0;
    MockCircomC12::getBinWitness(ctxC12, pWitnessC12, witnessSize);
    TimerStopAndLog(CIRCOM_GET_BIN_WITNESS);

    // Generate Groth16 via rapid SNARK
    TimerStart(RAPID_SNARK);
    json jsonProof;
    try
    {
        auto proof = groth16Prover->prove(pWitnessC12);
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
    delete ctxC12;
    MockCircom::freeCircuit(circuit);
    MockCircomC12::freeCircuit(circuitC12);

    free(pAddressC12);
    free(pWitnessC12);
}