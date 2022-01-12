#include <fstream>
#include <iomanip>
#include "prover.hpp"
#include "utils.hpp"
#include "mem.hpp"
#include "batchmachine_executor.hpp"
#include "proof2zkin.hpp"
#include "verifier_cpp/main.hpp"

using namespace std;

void Prover::prove (const Input &input)
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

#ifdef SAVE_STARK_PROOF_TO_DISK
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

#ifdef SAVE_ZKIN_PROOF_TO_DISK
    TimerStart(SAVE_ZKIN_PROOF);
    string zkinFile = starkFile + ".zkin.json";
    ofstream ofzkin(zkinFile);
    ofzkin << setw(4) << zkin << endl;
    ofzkin.close();
    TimerStop(SAVE_ZKIN_PROOF);
#endif



    /************/
    /* Verifier */
    /************/
    TimerStart(CIRCOM_LOAD_CIRCUIT);    
    Circom_Circuit *circuit = loadCircuit(verifierFile);
    TimerStopAndLog(CIRCOM_LOAD_CIRCUIT);
 
    TimerStart(CIRCOM_LOAD_JSON);
    Circom_CalcWit *ctx = new Circom_CalcWit(circuit);
    loadJson(ctx, /*zkinFile*/ "../testvectors/proof.json"); // TODO: Delete when the generated json object is available
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

    /*Generate Groth16 via rapid SNARK

     "Usage: prove <circuit.zkey> (Jordi to provide) <witness.wtns> (from circom) <proof.json> (output, small, to return via gRPC) <public.json> (output, not needed, contains public input)\n";
    */

    /***********/
    /* Cleanup */
    /***********/

    TimerStart(MEM_UNCOPY_POLS);
    MemUncopyPols(fr, mem, cmPols, constPols, constTreePolsInputFile);
    TimerStopAndLog(MEM_UNCOPY_POLS);

    MemFree(mem);
    cmPols.unmap();
}