#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include "ffiasm/fr.hpp"
#include "executor.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "stark_struct.hpp"
#include "pil.hpp"
#include "script.hpp"
#include "mem.hpp"
#include "batchmachine_executor.hpp"
#include "proof2zkin.hpp"
#include "calcwit.hpp"
#include "circom.hpp"
#include "verifier_cpp/main.hpp"

#ifdef RUN_GRPC_SERVER
#include "server.hpp"
#endif

using namespace std;
using json = nlohmann::json;


//fractasy@fractasy:~/git/grpc/cmake/build/third_party/protobuf$ ./protoc --proto_path=/home/fractasy/git/zkproverc/src/gRPC/proto --cpp_out=/home/fractasy/git/zkproverc/src/gRPC/gen /home/fractasy/git/zkproverc/src/gRPC/proto/zk-prover.proto 


int main (int argc, char** argv)
{
#ifdef RUN_GRPC_SERVER
    RawFr fr;
    ZkServer server(fr);
    server.run();
#else

    TimerStart(WHOLE_PROCESS);
    TimerStart(PARSE_JSON_FILES);

    /* Check executable input arguments:
       - Input JSON file must contain a set of transactions, and the old and mew states
       - ROM JSON file must contain the program instructions set
       - PIL JSON file must contain the circuit polynomials definition
       - Output JSON file will contain the proof
    */

    
    const char * pUsage = "Usage: zkprover <input.json> -r <rom.json> -p <main.pil.json> -o <commit.bin> -c <constants.bin> -t <constantstree.bin> -x <starkgen_bmscript.json> -s <stark.json> -v <verifier.dat> -w <witness.wtns>";
    const char * pInputFile = NULL;
    const char * pRomFile = "rom.json";
    const char * pPilFile = "zkevm.pil.json";
    const char * pOutputFile = "commit.bin";
    const char * pConstantsFile = "constants.bin";
    const char * pConstantsTreeFile = "constantstree.bin";
    const char * pScriptFile = "starkgen_bmscript.json";
    const char * pStarkFile = "stark.json";
    const char * pVerifierFile = "verifier.dat";
    const char * pWitnessFile = "witness.wtns";

    // Search for mandatory and optional arguments, if any
    for (int i=1; i<argc; i++)
    {
        // ROM JSON file arguments: "-r <rom.json>" or "-rom <rom.json>"
        if ( strcmp(argv[i],"-r")==0 || strcmp(argv[i],"-rom")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing ROM JSON file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pRomFile = argv[i];
            continue;
        }
        // PIL JSON file arguments: "-p <main.pil.json>" or "-pil <main.pil.json>"
        else if ( strcmp(argv[i],"-p")==0 || strcmp(argv[i],"-pil")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing PIL JSON file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pPilFile = argv[i];
            continue;
        }
        // Output JSON file arguments: "-o <proof.json>" or "-output <proof.json>"
        else if ( strcmp(argv[i],"-o")==0 || strcmp(argv[i],"-output")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing output JSON file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pOutputFile = argv[i];
            continue;
        }
        // Constants JSON file arguments: "-c <constants.json>" or "-constants <constants.json>"
        else if ( strcmp(argv[i],"-c")==0 || strcmp(argv[i],"-constants")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing constants JSON file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pConstantsFile = argv[i];
            continue;
        }
        // Constants tree JSON file arguments: "-t <constantstree.json>" or "-constantstree <constantstree.json>"
        else if ( strcmp(argv[i],"-t")==0 || strcmp(argv[i],"-constantstree")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing constants tree JSON file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pConstantsTreeFile = argv[i];
            continue;
        }
        // Script JSON file arguments: "-x <starkgen_bmscript.json>" or "-script <starkgen_bmscript.json>"
        else if ( strcmp(argv[i],"-x")==0 || strcmp(argv[i],"-script")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing script JSON file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pScriptFile = argv[i];
            continue;
        }
        // Stark tree JSON file arguments: "-s <stark.json>" or "-stark <stark.json>"
        else if ( strcmp(argv[i],"-s")==0 || strcmp(argv[i],"-stark")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing STARK JSON file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pStarkFile = argv[i];
            continue;
        }
        // Verifier binary file arguments: "-v <verifier.dat>" or "-verifier <verifier.dat>"
        else if ( strcmp(argv[i],"-v")==0 || strcmp(argv[i],"-verifier")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing verifier file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pVerifierFile = argv[i];
            continue;
        }
        // Witness binary file arguments: "-w <witness.wtns>" or "-witness <witness.wtns>"
        else if ( strcmp(argv[i],"-w")==0 || strcmp(argv[i],"-witness")==0 )
        {
            i++;
            if ( i >= argc )
            {
                cerr << "Error: Missing witness file name" << endl;
                cout << pUsage << endl;
                exit(-1);
            }
            pWitnessFile = argv[i];
            continue;
        }
        else if (pInputFile == NULL)
        {
            pInputFile = argv[1];
            continue;
        }
        else
        {
            cerr << "Error: Unrecognized argument: " << argv[i] << endl;
            cout << pUsage << endl;
            exit(-1);
        }
    }

    // Check that at least we got the input JSON file argument
    if ( pInputFile == NULL )
    {
        cerr << "Error: You need to specify an input file name" << endl;
        cout << pUsage << endl;
        exit(-1);
    }
    
    // Log parsed arguments and/or default file names
    cout << "Input file=" << pInputFile << endl;
    cout << "ROM file=" << pRomFile << endl;
    cout << "PIL file=" << pPilFile << endl;
    cout << "Output file=" << pOutputFile << endl;
    cout << "Constants file=" << pConstantsFile << endl;
    cout << "Constants tree file=" << pConstantsTreeFile << endl;
    cout << "Script file=" << pScriptFile << endl;
    cout << "STARK file=" << pStarkFile << endl;
    cout << "Verifier file=" << pVerifierFile << endl;
    cout << "Witness file=" << pWitnessFile << endl;

    //zkprover::Proof pr;
    //zkprover::State st;

    

    
    //zkProverProto.ZKProver()
    


    // Load and parse input JSON file
    std::ifstream inputStream(pInputFile);
    if (!inputStream.good())
    {
        cerr << "Error: failed loading input JSON file " << pInputFile << endl;
        exit(-1);
    }
    json inputJson;
    inputStream >> inputJson;
    inputStream.close();

    // Load and parse ROM JSON file
    std::ifstream romStream(pRomFile);
    if (!romStream.good())
    {
        cerr << "Error: failed loading ROM JSON file " << pRomFile << endl;
        exit(-1);
    }
    json romJson;
    romStream >> romJson;
    romStream.close();

    // Load and parse PIL JSON file
    std::ifstream pilStream(pPilFile);
    if (!pilStream.good())
    {
        cerr << "Error: failed loading PIL JSON file " << pPilFile << endl;
        exit(-1);
    }
    json pilJson;
    pilStream >> pilJson;
    pilStream.close();

    // Load and parse script JSON file
    std::ifstream scriptStream(pScriptFile);
    if (!scriptStream.good())
    {
        cerr << "Error: failed loading script JSON file " << pScriptFile << endl;
        exit(-1);
    }
    json scriptJson;
    scriptStream >> scriptJson;
    scriptStream.close(); 
    
    // Output and input file names
    string outputFile(pOutputFile);
    string constantsFile(pConstantsFile);
    string constantsTreeFile(pConstantsTreeFile);

    TimerStop(PARSE_JSON_FILES);

    // This raw FR library has been compiled to implement the curve BN128
    RawFr fr;

#ifdef DEBUG
    batchInverseTest(fr);
#endif

    /*************************/
    /* Parse input pols data */
    /*************************/

    TimerStart(LOAD_POLS_TO_MEMORY);
        
    // Load PIL JSON file content into memory */
    Pil pil;
    pil.parse(pilJson);

    // Load committed polynomials into memory, mapped to a newly created output file, filled by executor
    Pols cmPols;
    cmPols.load(pil.cmPols);
    cmPols.mapToOutputFile(outputFile);

    // Load constant polynomials into memory, and map them to an existing input file containing their values
    Pols constPols;
    constPols.load(pil.constPols);
    constPols.mapToInputFile(constantsFile);

    TimerStop(LOAD_POLS_TO_MEMORY);
    TimerLog(LOAD_POLS_TO_MEMORY);

    /************/
    /* EXECUTOR */
    /************/

    TimerStart(ROM_LOAD);

    // Instantiate the ROM
    Rom romData;
    romData.load(romJson);
    
    TimerStop(ROM_LOAD);

    // Instantiate and load the executor
    Executor executor(fr, romData);
    
    // Call execute
    TimerStart(INPUT_LOAD);
    Input input(fr);
    input.load(inputJson);
    TimerStopAndLog(INPUT_LOAD);

    TimerStart(EXECUTOR_EXECUTE);

    // Call execute    
    executor.execute(input, cmPols);
    
    TimerStop(EXECUTOR_EXECUTE);

    TimerStart(ROM_UNLOAD);
    
    // Unload the executor
    romData.unload();
    
    TimerStop(ROM_UNLOAD);

    TimerStop(WHOLE_PROCESS);

    TimerLog(PARSE_JSON_FILES);
    TimerLog(ROM_LOAD);
    TimerLog(EXECUTOR_EXECUTE);
    TimerLog(ROM_UNLOAD);
    TimerLog(WHOLE_PROCESS);

    /***********************/
    /* STARK Batch Machine */
    /***********************/
    TimerStart(SCRIPT_PARSE);

    Script script;
    script.parse(scriptJson);

    TimerStopAndLog(SCRIPT_PARSE);

    TimerStart(MEM_ALLOC);

    Mem mem;
    MemAlloc(mem, script);
    
    TimerStopAndLog(MEM_ALLOC);

    TimerStart(MEM_COPY);
    
    MemCopyPols(fr, mem, cmPols, constPols); // TODO: copy also constTree
    
    TimerStopAndLog(MEM_COPY);

    TimerStart(BM_EXECUTOR);
    
    json proof;
    batchMachineExecutor(fr, mem, script, proof);
    
    TimerStopAndLog(BM_EXECUTOR);

    /****************/
    /* Proof 2 zkIn */
    /****************/

    TimerStart(PROOF2ZKIN);

    json zkin;
    //proof2zkin(proof, zkin);

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

    writeBinWitness(ctx, pWitnessFile);
    */
    /***********/
    /* Cleanup */
    /***********/

    MemFree(mem);
    cmPols.unmap();
    constPols.unmap();

    cout << "Done" << endl;

#endif
}