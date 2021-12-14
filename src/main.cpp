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

using namespace std;
using json = nlohmann::json;

int main (int argc, char** argv)
{
    TimerStart(WHOLE_PROCESS);
    TimerStart(PARSE_JSON_FILES);

    /* Check executable input arguments:
       - Input JSON file must contain a set of transactions, and the old and mew states
       - ROM JSON file must contain the program instructions set
       - PIL JSON file must contain the circuit polynomials definition
       - Output JSON file will contain the proof
    */

    
    const char * pUsage = "Usage: zkprover <input.json> -r <rom.json> -p <main.pil.json> -o <commit.bin> -c <constants.bin> -t <constantstree.bin> -s <stark.json>";
    const char * pInputFile = NULL;
    const char * pRomFile = "rom.json";
    const char * pPilFile = "zkevm.pil.json";
    const char * pOutputFile = "commit.bin";
    const char * pConstantsFile = "constants.bin";
    const char * pConstantsTreeFile = "constantstree.bin";
    const char * pStarkFile = "stark.json";

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
    cout << "STARK file=" << pStarkFile << endl;

    // Load and parse input JSON file
    std::ifstream inputStream(pInputFile);
    if (!inputStream.good())
    {
        cerr << "Error: failed loading input JSON file " << pInputFile << endl;
        exit(-1);
    }
    json inputFile;
    inputStream >> inputFile;
    inputStream.close();

    // Load and parse ROM JSON file
    std::ifstream romStream(pRomFile);
    if (!romStream.good())
    {
        cerr << "Error: failed loading ROM JSON file " << pRomFile << endl;
        exit(-1);
    }
    json romFile;
    romStream >> romFile;
    romStream.close();

    // Load and parse PIL JSON file
    std::ifstream pilStream(pPilFile);
    if (!pilStream.good())
    {
        cerr << "Error: failed loading PIL JSON file " << pPilFile << endl;
        exit(-1);
    }
    json pilFile;
    pilStream >> pilFile;
    pilStream.close(); 
    
    // Output and input file names
    string outputFile(pOutputFile);
    string constantsFile(pConstantsFile);
    string constantsTreeFile(pConstantsTreeFile);

    TimerStop(PARSE_JSON_FILES);

    // This raw FR library has been compiled to implement the curve BN128
    RawFr fr;

    /*************************/
    /* Parse input pols data */
    /*************************/

    TimerStart(LOAD_POLS_TO_MEMORY);
        
    // Load PIL JSON file content into memory */
    vector<PolJsonData> cmPolsJsonData;
    vector<PolJsonData> constPolsJsonData;
    Pols::parse(pilFile, cmPolsJsonData, constPolsJsonData);

    // Load committed polynomials into memory, mapped to a newly created output file
    Pols cmPols;
    cmPols.load(cmPolsJsonData);
    cmPols.mapToOutputFile(outputFile);

    // Load constant polynomials into memory, and map them to an existing input file containing their values
    Pols constPols;
    constPols.load(constPolsJsonData);
    constPols.mapToInputFile(constantsFile);

    TimerStop(LOAD_POLS_TO_MEMORY);
    TimerLog(LOAD_POLS_TO_MEMORY);

    /************/
    /* EXECUTOR */
    /************/

    TimerStart(EXECUTOR_LOAD);
        
    // Instantiate and load the executor
    Executor executor(fr);
    executor.load(romFile);
    
    TimerStop(EXECUTOR_LOAD);

    TimerStart(EXECUTOR_EXECUTE);
    
    // Call execute
    executor.execute(inputFile, pilFile, outputFile, cmPols);
    
    TimerStop(EXECUTOR_EXECUTE);

    TimerStart(EXECUTOR_UNLOAD);
    
    // Unload the executor
    executor.unload();
    
    TimerStop(EXECUTOR_UNLOAD);

    TimerStop(WHOLE_PROCESS);

    TimerLog(PARSE_JSON_FILES);
    TimerLog(EXECUTOR_LOAD);
    TimerLog(EXECUTOR_EXECUTE);
    TimerLog(EXECUTOR_UNLOAD);
    TimerLog(WHOLE_PROCESS);

    /*******************/
    /* STARK GENERATOR */
    /*******************/

        //const M = new Merkle(16, poseidon, poseidon.F);

    //const groupSize = 1 << (Nbits+extendBits - starkStruct[0].nBits);
    //const nGroups = 1 << starkStruct[0].nBits;
    uint64_t groupSize = 1 << (NBITS + EXTENDED_BITS - starkStruct[0].nBits);
    uint64_t nGroups = 1 << starkStruct[0].nBits;

    cout << "Done" << endl;
    
    //const MGPC = new MerkleGroupMultipol(M, nGroups, groupSize, pil.nConstants);
    //const MGP = new MerkleGroupMultipol(M, 2**16, 2, pols.length);
    //const constTree = await importMerkleGroupMultipol(constantTreeFile, MGPC);

    /*const starkProof = await starkGen(pols, polsConst, constTree, pil, {
        N: N,
        starkStruct: starkStruct
    });*/

    /*const starkProofJ = stringifyFElements(F, starkProof);

    await fs.promises.writeFile(outputFile, JSON.stringify(starkProofJ, null, 1), "utf8");

    console.log("Stark generated correctly");*/

    cmPols.unmap();
    constPols.unmap();

}