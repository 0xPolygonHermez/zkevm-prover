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

using namespace std;
using json = nlohmann::json;

int main (int argc, char** argv)
{
    TimeStart(Main);
    TimeStart(Main_JSON_Parse);

    /* Check executable input arguments:
       - Input JSON file must contain a set of transactions, and the old and mew states
       - ROM JSON file must contain the program instructions set
       - PIL JSON file must contain the circuit polynomials definition
       - Output JSON file will contain the proof
    */
    
    const char * pUsage      = "Usage: zkprover <input.json> -r <rom.json> -p <main.pil.json> -o <pols>";
    const char * pInputFile  = NULL;
    const char * pRomFile    = "rom.json";
    const char * pPilFile    = "zkevm.pil.json";
    const char * pOutputFile = "pols";

    // Search for mandatory and optional arguments, if any
    for (int i=1; i<argc; i++)
    {
        // ROM JSON file arguments: "-r <rom.json>" or "-rom <rom.json>"
        if ( strcmp(argv[i],"r")==0 || strcmp(argv[i],"rom")==0 )
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
        else if ( strcmp(argv[i],"p")==0 || strcmp(argv[i],"pil")==0 )
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
        else if ( strcmp(argv[i],"o")==0 || strcmp(argv[i],"output")==0 )
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
        else if (pInputFile == NULL)
        {
            pInputFile = argv[1];
            continue;
        }
        else
        {
            cerr << "Error: Unrecognized argument" << endl;
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
    
    // Output file name
    string outputFile(pOutputFile);

    TimeStop(Main_JSON_Parse);

    // This raw FR library has been compiled to implement the curve BN128
    RawFr fr;

    // Call execute
    execute(fr, inputFile, romFile, pilFile, outputFile);

    TimeStop(Main);

}