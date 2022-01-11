#include "prover.hpp"
#include "utils.hpp"
#include "mem.hpp"
#include "batchmachine_executor.hpp"

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

    TimerStart(MEM_COPY);
    MemCopyPols(fr, mem, cmPols, constPols, constTreePolsInputFile);
    TimerStopAndLog(MEM_COPY);

    TimerStart(BM_EXECUTOR);
    json starkProof;
    BatchMachineExecutor bme(fr, script);
    bme.execute(mem, starkProof);
    TimerStopAndLog(BM_EXECUTOR);

    /****************/
    /* Proof 2 zkIn */
    /****************/

    TimerStart(PROOF2ZKIN);

    json zkin;
    //proof2zkin(starkProof, zkin);

    //ofstream o("zkin.json");
    //o << setw(4) << zkin << endl;
    //o.close();

    TimerStopAndLog(PROOF2ZKIN);

    /************/
    /* Verifier */
    /************/

    // TODO: Should we save zkin to file and use it as input file for the verifier?
    /*
    Circom_Circuit *circuit = loadCircuit("zkin.json"); // proof.json
    Circom_CalcWit *ctx = new Circom_CalcWit(circuit);
 
    loadJson(ctx, pInputFile);
    if (ctx->getRemaingInputsToBeSet()!=0) {
        cerr << "Error: Not all inputs have been set. Only " << get_main_input_signal_no()-ctx->getRemaingInputsToBeSet() << " out of " << get_main_input_signal_no() << endl;
        exit(-1);
    }

    writeBinWitness(ctx, pWitnessFile); // No need to write the file to disk, 12-13M fe, in binary, in wtns format

    Generate Groth16 via rapid SNARK

     "Usage: prove <circuit.zkey> (Jordi to provide) <witness.wtns> (from circom) <proof.json> (output, small, to return via gRPC) <public.json> (output, not needed, contains public input)\n";
    */

    /***********/
    /* Cleanup */
    /***********/

    MemFree(mem);
    cmPols.unmap();
}