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
#include "prover.hpp"
#include "server.hpp"

using namespace std;
using json = nlohmann::json;

// bServerMode configures the behavior based on the presence or not of the input.json program argument
// $ zkProver input.json -> bServerMode = false, the program will only process the provided input.json file
// $ zkProver -> bServerMode = true, the program will be a gRPC server waiting for incoming GenProof requests
bool bServerMode = true;

// fractasy@fractasy:~//grpc/cmake/build/third_pgitarty/protobuf$ ./protoc --proto_path=/home/fractasy/git/zkproverc/src/gRPC/proto --cpp_out=/home/fractasy/git/zkproverc/src/gRPC/gen /home/fractasy/git/zkproverc/src/gRPC/proto/zk-prover.proto

int main(int argc, char **argv)
{
    TimerStart(WHOLE_PROCESS);
    TimerStart(PARSE_JSON_FILES);

    /* Check executable input arguments:
       - Input JSON file must contain a set of transactions, and the old and mew states
       - ROM JSON file must contain the program instructions set
       - PIL JSON file must contain the circuit polynomials definition
       - Output JSON file will contain the proof
    */

    const char *pInputFile = NULL;
    string romFile = "rom.json";
    string pilFile = "zkevm.pil.json";
    string cmPolsFile = "commit.bin";
    string constPolsFile = "constants.bin";
    string constantsTreeFile = "constantstree.bin";
    string scriptFile = "starkgen_bmscript.json";
    string starkFile = "stark.json";
    string verifierFile = "verifier.dat";
    string witnessFile = "witness.wtns";
    string starkVerifierFile = "starkverifier_0001.zkey";
    string proofFile = "proof.json";
    DatabaseConfig databaseConfig;
    databaseConfig.bUseServer = false;
    databaseConfig.host = DATABASE_HOST;
    databaseConfig.port = DATABASE_PORT;
    databaseConfig.user = DATABASE_USER;
    databaseConfig.password = DATABASE_PASSWORD;
    databaseConfig.databaseName = DATABASE_NAME;
    databaseConfig.tableName = DATABASE_TABLE_NAME;
    string usage =
        "Usage: zkprover <input.json> -r <" + romFile + "> " +
        "-p <" + pilFile + "> " +
        "-o <" + cmPolsFile + "> " +
        "-c <" + constPolsFile + "> " +
        "-t <" + constantsTreeFile + "> " +
        "-x <" + scriptFile + "> " +
        "-s <" + starkFile + "> " +
        "-v <" + verifierFile + "> " +
        "-w <" + witnessFile + "> " +
        "-k <" + starkVerifierFile + "> " +
        "-f <" + proofFile + "> " +
        "-dbuseserver <" + to_string(databaseConfig.bUseServer) + "> " +
        "-dbhost <" + databaseConfig.host + "> " +
        "-dbport <" + to_string(databaseConfig.port) + "> " +
        "-dbuser <" + databaseConfig.user + "> " +
        "-dbpwd <" + databaseConfig.password + "> " +
        "-dbdatabasename <" + databaseConfig.databaseName + "> " +
        "-dbtablename <" + databaseConfig.tableName + "> ";

    // Search for mandatory and optional arguments, if any
    for (int i = 1; i < argc; i++)
    {
        // ROM JSON file arguments: "-r <rom.json>" or "-rom <rom.json>"
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "-rom") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing ROM JSON file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            romFile = argv[i];
            continue;
        }
        // PIL JSON file arguments: "-p <main.pil.json>" or "-pil <main.pil.json>"
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "-pil") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing PIL JSON file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            pilFile = argv[i];
            continue;
        }
        // Output JSON file arguments: "-o <proof.json>" or "-output <proof.json>"
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "-output") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing output JSON file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            cmPolsFile = argv[i];
            continue;
        }
        // Constants JSON file arguments: "-c <constants.json>" or "-constants <constants.json>"
        else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "-constants") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing constants JSON file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            constPolsFile = argv[i];
            continue;
        }
        // Constants tree JSON file arguments: "-t <constantstree.json>" or "-constantstree <constantstree.json>"
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "-constantstree") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing constants tree JSON file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            constantsTreeFile = argv[i];
            continue;
        }
        // Script JSON file arguments: "-x <starkgen_bmscript.json>" or "-script <starkgen_bmscript.json>"
        else if (strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "-script") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing script JSON file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            scriptFile = argv[i];
            continue;
        }
        // Stark tree JSON file arguments: "-s <stark.json>" or "-stark <stark.json>"
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "-stark") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing STARK JSON file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            starkFile = argv[i];
            continue;
        }
        // Verifier binary file arguments: "-v <verifier.dat>" or "-verifier <verifier.dat>"
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "-verifier") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing verifier file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            verifierFile = argv[i];
            continue;
        }
        // Witness binary file arguments: "-w <witness.wtns>" or "-witness <witness.wtns>"
        else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "-witness") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing witness file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            witnessFile = argv[i];
            continue;
        }
        // STARK verifier binary file arguments: "-k <starkverifier_0001.zkey>" or "-witness <starkverifier_0001.zkey>"
        else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "-starkverifier") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing STARK verifier file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            starkVerifierFile = argv[i];
            continue;
        }
        // Proof JSON binary file arguments: "-f <proof.json>" or "-proof <proof.json>"
        else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "-proof") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing proof file name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            proofFile = argv[i];
            continue;
        }
        // Database use server argument
        else if (strcmp(argv[i], "-dbuseserver") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing use server" << endl;
                cout << usage << endl;
                exit(-1);
            }
            if (strcmp(argv[i],"true") == 0)
            { 
                databaseConfig.bUseServer = true;
            }
            continue;
        }
        // Database host argument
        else if (strcmp(argv[i], "-dbhost") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing database host" << endl;
                cout << usage << endl;
                exit(-1);
            }
            databaseConfig.host = argv[i];
            databaseConfig.bUseServer = true;
            continue;
        }
        // Database port argument
        else if (strcmp(argv[i], "-dbport") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing database port" << endl;
                cout << usage << endl;
                exit(-1);
            }
            databaseConfig.port = atoi(argv[i]);
            databaseConfig.bUseServer = true;
            continue;
        }
        // Database user argument
        else if (strcmp(argv[i], "-dbuser") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing database user" << endl;
                cout << usage << endl;
                exit(-1);
            }
            databaseConfig.user = argv[i];
            databaseConfig.bUseServer = true;
            continue;
        }
        // Database user argument
        else if (strcmp(argv[i], "-dbpwd") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing database password" << endl;
                cout << usage << endl;
                exit(-1);
            }
            databaseConfig.password = argv[i];
            databaseConfig.bUseServer = true;
            continue;
        }
        // Database database name argument
        else if (strcmp(argv[i], "-dbdatabasename") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing database name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            databaseConfig.databaseName = argv[i];
            databaseConfig.bUseServer = true;
            continue;
        }
        // Database table name argument
        else if (strcmp(argv[i], "-dbtablename") == 0)
        {
            i++;
            if (i >= argc)
            {
                cerr << "Error: Missing database table name" << endl;
                cout << usage << endl;
                exit(-1);
            }
            databaseConfig.tableName = argv[i];
            databaseConfig.bUseServer = true;
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
            cout << usage << endl;
            exit(-1);
        }
    }

    // If we got the input JSON file argument, disable the server mode and just prove that file
    // Otherwise, input data will be provided via service client
    if (pInputFile != NULL)
    {
        bServerMode = false;
    }

    // Log parsed arguments and/or default file names
    cout << "Input file=" << (bServerMode?"NULL":pInputFile) << endl;
    cout << "ROM file=" << romFile << endl;
    cout << "PIL file=" << pilFile << endl;
    cout << "Output file=" << cmPolsFile << endl;
    cout << "Constants file=" << constPolsFile << endl;
    cout << "Constants tree file=" << constantsTreeFile << endl;
    cout << "Script file=" << scriptFile << endl;
    cout << "STARK file=" << starkFile << endl;
    cout << "Verifier file=" << verifierFile << endl;
    cout << "Witness file=" << witnessFile << endl;
    cout << "STARK verifier file=" << starkVerifierFile << endl;
    cout << "Proof file=" << proofFile << endl;

    // Load and parse input JSON file
    json inputJson;
    if (!bServerMode)
    {
        std::ifstream inputStream(pInputFile);
        if (!inputStream.good())
        {
            cerr << "Error: failed loading input JSON file " << pInputFile << endl;
            exit(-1);
        }
        inputStream >> inputJson;
        inputStream.close();
    }

    // Input file names
    string inputFile(bServerMode?"NULL":pInputFile);

    // Load and parse ROM JSON file
    std::ifstream romStream(romFile);
    if (!romStream.good())
    {
        cerr << "Error: failed loading ROM JSON file " << romFile << endl;
        exit(-1);
    }
    json romJson;
    romStream >> romJson;
    romStream.close();

    // Load and parse PIL JSON file
    std::ifstream pilStream(pilFile);
    if (!pilStream.good())
    {
        cerr << "Error: failed loading PIL JSON file " << pilFile << endl;
        exit(-1);
    }
    json pilJson;
    pilStream >> pilJson;
    pilStream.close();

    // Load and parse script JSON file
    std::ifstream scriptStream(scriptFile);
    if (!scriptStream.good())
    {
        cerr << "Error: failed loading script JSON file " << scriptFile << endl;
        exit(-1);
    }
    json scriptJson;
    scriptStream >> scriptJson;
    scriptStream.close();

    TimerStopAndLog(PARSE_JSON_FILES);

    // This raw FR library has been compiled to implement the curve BN128
    RawFr fr;

#ifdef DEBUG
    BatchMachineExecutor::batchInverseTest(fr);
#endif

    /*************************/
    /* Parse input pols data */
    /*************************/

    TimerStart(LOAD_POLS_TO_MEMORY);

    // Load PIL JSON file content into memory */
    Pil pil;
    pil.parse(pilJson);

    // Load constant polynomials into memory, and map them to an existing input file containing their values
    Pols constPols;
    constPols.load(pil.constPols);
    constPols.mapToInputFile(constPolsFile);

    TimerStopAndLog(LOAD_POLS_TO_MEMORY);

    // Instantiate the ROM
    TimerStart(ROM_LOAD);
    Rom romData;
    romData.load(romJson);
    TimerStopAndLog(ROM_LOAD);

    // Parse Input JSON file
    Input input(fr);
    if (!bServerMode)
    {
        TimerStart(INPUT_LOAD);
        input.load(inputJson);
        TimerStopAndLog(INPUT_LOAD);
    }

    // Parse script JSON file
    TimerStart(SCRIPT_PARSE);
    Script script;
    script.parse(scriptJson);
    TimerStopAndLog(SCRIPT_PARSE);

    // Create the prover
    Prover prover(  fr,
                    romData,
                    script,
                    pil,
                    constPols,
                    cmPolsFile,
                    constantsTreeFile,
                    inputFile,
                    starkFile,
                    verifierFile,
                    witnessFile,
                    starkVerifierFile,
                    proofFile,
                    databaseConfig );

    if (bServerMode)
    {
        // Create server instance, passing all constant data
        ZkServer server(fr, prover);

        // Run the server
        server.run(); // Internally, it calls prover.prove() for every input data received, in order to generate the proof and return it to the client
    }
    else
    {
        // Call the prover
        TimerStart(PROVE);
        Proof proof;
        prover.prove(input, proof);
        TimerStopAndLog(PROVE);
    }

    // Unload the ROM data
    TimerStart(ROM_UNLOAD);
    romData.unload();
    TimerStopAndLog(ROM_UNLOAD);

    TimerStopAndLog(WHOLE_PROCESS);

    cout << "Done" << endl;
}