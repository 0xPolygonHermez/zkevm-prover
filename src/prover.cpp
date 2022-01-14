#include <fstream>
#include <iomanip>
#include "prover.hpp"
#include "utils.hpp"
#include "mem.hpp"
#include "batchmachine_executor.hpp"
#include "proof2zkin.hpp"
#include "verifier_cpp/main.hpp"
#include "rapidsnark/rapidsnark_prover.hpp"

using namespace std;

void Prover::prove (const Input &input, Proof &proof)
{
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
    MemAlloc(mem, script);
    TimerStopAndLog(MEM_ALLOC);

    TimerStart(MEM_COPY_POLS);
    MemCopyPols(fr, mem, cmPols, constPols, constTreePolsInputFile);
    TimerStopAndLog(MEM_COPY_POLS);

    TimerStart(BM_EXECUTOR);
    json starkProof;
    BatchMachineExecutor bme(fr, script);
    bme.execute(mem, starkProof);

    TimerStopAndLog(BM_EXECUTOR);

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
    TimerStopAndLog(PROOF2ZKIN);

#ifdef PROVER_SAVE_ZKIN_PROOF_TO_DISK
    TimerStart(SAVE_ZKIN_PROOF);
    string zkinFile = starkFile + ".zkin.json";
    ofstream ofzkin(zkinFile);
    ofzkin << setw(4) << zkin << endl;
    ofzkin.close();
    TimerStopAndLog(SAVE_ZKIN_PROOF);
#endif

    /************/
    /* Verifier */
    /************/
    TimerStart(CIRCOM_LOAD_CIRCUIT);    
    Circom_Circuit *circuit = loadCircuit(verifierFile);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);
 
    TimerStart(CIRCOM_LOAD_JSON);
    Circom_CalcWit *ctx = new Circom_CalcWit(circuit);
    loadJson(ctx, /*zkinFile*/ "../testvectors/zkin.json"); // TODO: Delete when the generated json object is available
    //loadJsonImpl(ctx, zkin); // TODO: Uncomment when the generated json object is available
    if (ctx->getRemaingInputsToBeSet()!=0)
    {
        cerr << "Error: Not all inputs have been set. Only " << get_main_input_signal_no()-ctx->getRemaingInputsToBeSet() << " out of " << get_main_input_signal_no() << endl;
        exit(-1);
    }
    TimerStopAndLog(CIRCOM_LOAD_JSON);

    TimerStart(CIRCOM_WRITE_BIN_WITNESS);
    writeBinWitness(ctx, witnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format
    TimerStopAndLog(CIRCOM_WRITE_BIN_WITNESS);

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
    json jsonPublic;
    // TODO: Logger traces are not coming out in CONSOLE mode
    rapidsnark_prover(starkVerifierFile, witnessFile, jsonProof, jsonPublic);
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
    //publicInputsExtended.inputHash = ?; // TODO: How do we calculate this inputHash?
    proof.load(jsonProof, publicInputsExtended);

    /***********/
    /* Cleanup */
    /***********/

    TimerStart(MEM_UNCOPY_POLS);
    MemUncopyPols(fr, mem, cmPols, constPols, constTreePolsInputFile);
    TimerStopAndLog(MEM_UNCOPY_POLS);

    MemFree(mem);
    cmPols.unmap();

    cout << "Prover::prove() done" << endl;
}