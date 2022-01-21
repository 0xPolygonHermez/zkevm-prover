#include <fstream>
#include <iomanip>
#include "prover.hpp"
#include "utils.hpp"
#include "mem.hpp"
#include "batchmachine_executor.hpp"
#include "proof2zkin.hpp"
#include "verifier_cpp/main.hpp"
#include "rapidsnark/rapidsnark_prover.hpp"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"

using namespace std;

Prover::Prover( RawFr &fr,
            const Rom &romData,
            const Script &script,
            const Pil &pil,
            const Pols &constPols,
            const string &cmPolsOutputFile,
            const string &constTreePolsInputFile,
            const string &inputFile,
            const string &starkFile,
            const string &verifierFile,
            const string &witnessFile,
            const string &starkVerifierFile,
            const string &proofFile,
            const DatabaseConfig &databaseConfig ) :
        fr(fr),
        romData(romData),
        executor(fr, romData, databaseConfig),
        script(script),
        pil(pil),
        constPols(constPols),
        cmPolsOutputFile(cmPolsOutputFile),
        constTreePolsInputFile(constTreePolsInputFile),
        inputFile(inputFile),
        starkFile(starkFile),
        verifierFile(verifierFile),
        witnessFile(witnessFile),
        starkVerifierFile(starkVerifierFile),
        proofFile(proofFile)
{
    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);
    
    try {
        auto zkey = BinFileUtils::openExisting(starkVerifierFile, "zkey", 1); // TODO: Should we delete this?
        auto zkeyHeader = ZKeyUtils::loadHeader(zkey.get()); // TODO: Should we delete this?

        //std::string proofStr;
        if (mpz_cmp(zkeyHeader->rPrime, altBbn128r) != 0) {
            throw std::invalid_argument( "zkey curve not supported" );
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
            zkey->getSectionData(4),    // Coefs
            zkey->getSectionData(5),    // pointsA
            zkey->getSectionData(6),    // pointsB1
            zkey->getSectionData(7),    // pointsB2
            zkey->getSectionData(8),    // pointsC
            zkey->getSectionData(9)     // pointsH1
        );

    } catch (std::exception& e) {
        cerr << "Error: Prover::Prover() got an exception: " << e.what() << '\n';
        exit(-1);
    }

    Pols2Refs(fr, constPols, constRefs);
}

Prover::~Prover ()
{
    mpz_clear(altBbn128r);
}

void Prover::prove (const Input &input, Proof &proof)
{
    TimerStart(PROVER_PROVE);

    /************/
    /* Executor */
    /************/
    
    // Load committed polynomials into memory, mapped to a newly created output file, filled by executor
    Pols cmPols;
    cmPols.load(pil.cmPols);
    cmPols.mapToOutputFile(cmPolsOutputFile);

    // Execute the program
    TimerStart(EXECUTOR_EXECUTE);
    executor.execute(input, cmPols);
    TimerStopAndLog(EXECUTOR_EXECUTE);

    /***********************/
    /* STARK Batch Machine */
    /***********************/

    TimerStart(MEM_ALLOC);
    Mem mem;
    MemAlloc(mem, fr, script, cmPols, constRefs, constTreePolsInputFile);
    TimerStopAndLog(MEM_ALLOC);

    TimerStart(BATCH_MACHINE_EXECUTOR);
    json starkProof;
    BatchMachineExecutor bme(fr, script);
    bme.execute(mem, starkProof);

    TimerStopAndLog(BATCH_MACHINE_EXECUTOR);

#ifdef PROVER_SAVE_STARK_PROOF_TO_DISK
    TimerStart(SAVE_STARK_PROOF);
    ofstream ofstark(starkFile);
    ofstark << setw(4) << starkProof << endl;
    ofstark.close();
    TimerStopAndLog(SAVE_STARK_PROOF);
#endif

    /****************/
    /* Proof 2 zkIn */
    /****************/

    TimerStart(PROOF2ZKIN);
    json zkin;
    proof2zkin(starkProof, zkin);
    zkin["globalHash"] = fr.toString(cmPols.FREE0.pData[0], 16);
    TimerStopAndLog(PROOF2ZKIN);

#ifdef PROVER_SAVE_ZKIN_PROOF_TO_DISK
    TimerStart(SAVE_ZKIN_PROOF);
    string zkinFile = starkFile;
    zkinFile.erase(zkinFile.find_last_not_of(".json")+1);
    zkinFile += ".zkin.json";
    ofstream ofzkin(zkinFile);
    ofzkin << setw(4) << zkin << endl;
    ofzkin.close();
    TimerStopAndLog(SAVE_ZKIN_PROOF);
#endif

#ifdef PROVER_INJECT_ZKIN_JSON
    TimerStart(PROVER_INJECT_ZKIN_JSON);
    zkin.clear();
    std::ifstream zkinStream("/home/fractasy/git/zkproverc/testvectors/zkin.json");
    if (!zkinStream.good())
    {
        cerr << "Error: failed loading zkin.json file " << endl;
        exit(-1);
    }
    zkinStream >> zkin;
    zkinStream.close();
    TimerStopAndLog(PROVER_INJECT_ZKIN_JSON);
#endif

    /************/
    /* Verifier */
    /************/
    TimerStart(CIRCOM_LOAD_CIRCUIT);    
    Circom_Circuit *circuit = loadCircuit(verifierFile);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);
 
    TimerStart(CIRCOM_LOAD_JSON);
    Circom_CalcWit *ctx = new Circom_CalcWit(circuit);
    loadJsonImpl(ctx, zkin);
    if (ctx->getRemaingInputsToBeSet()!=0)
    {
        cerr << "Error: Not all inputs have been set. Only " << get_main_input_signal_no()-ctx->getRemaingInputsToBeSet() << " out of " << get_main_input_signal_no() << endl;
        exit(-1);
    }
    TimerStopAndLog(CIRCOM_LOAD_JSON);

#ifdef PROVER_SAVE_WITNESS_TO_DISK
    TimerStart(CIRCOM_WRITE_BIN_WITNESS);
    writeBinWitness(ctx, witnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
    TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);
#endif

    TimerStart(CIRCOM_GET_BIN_WITNESS);
    AltBn128::FrElement * pWitness = NULL;
    uint64_t witnessSize = 0;
    getBinWitness(ctx, pWitness, witnessSize);
    TimerStopAndLog(CIRCOM_GET_BIN_WITNESS);

#ifdef PROVER_USE_PROOF_GOOD_JSON
    // Load and parse a good proof JSON file, just for development and testing purposes
    string goodProofFile = "../testvectors/proof.good.json";
    std::ifstream goodProofStream(goodProofFile);
    if (!goodProofStream.good())
    {
        cerr << "Error: failed loading a good proof JSON file " << goodProofFile << endl;
        exit(-1);
    }
    json jsonProof;
    goodProofStream >> jsonProof;
    goodProofStream.close();
#else
    // Generate Groth16 via rapid SNARK
    TimerStart(RAPID_SNARK);
    json jsonProof;
    try
    {
        auto proof = groth16Prover->prove(pWitness); // TODO: Don't compile rapid snark files
        jsonProof = proof->toJson();
    }
    catch (std::exception& e)
    {
        cerr << "Error: Prover::Prove() got exception in rapid SNARK:" << e.what() << '\n';
        exit(-1);
    }
    TimerStopAndLog(RAPID_SNARK);
#endif

#ifdef PROVER_SAVE_PROOF_TO_DISK
    ofstream ofproof(proofFile);
    ofproof << setw(4) << jsonProof << endl;
    ofproof.close();
#endif

    // Populate Proof with the correct data
    PublicInputsExtended publicInputsExtended;
    publicInputsExtended.publicInputs = input.publicInputs;
    publicInputsExtended.inputHash = fr.toString(cmPols.FREE0.pData[0], 16);
    proof.load(jsonProof, publicInputsExtended);

    /***********/
    /* Cleanup */
    /***********/

    TimerStart(MEM_FREE);
    MemFree(mem);
    TimerStopAndLog(MEM_FREE);

    free(pWitness);
    cmPols.unmap();

    //cout << "Prover::prove() done" << endl;

    TimerStopAndLog(PROVER_PROVE);
}