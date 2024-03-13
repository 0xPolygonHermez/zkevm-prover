#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <sys/stat.h>
#include <alt_bn128.hpp>
#include "../rapidsnark/fflonk_setup.hpp"

using namespace std;

int main (int argc, char **argv)
{
    cout << "Fflonk setup" << endl;

    Logger::getInstance()->enableConsoleLogging();
    Logger::getInstance()->updateLogLevel(LOG_LEVEL_DEBUG);

    // Check arguments list
    if (argc != 4)
    {
        cerr << "Error: expected 3 arguments but got " << argc - 1 << " Usage: fflonkSetup <r1csFile> <ptauFile> <zkeyFile>" << endl;
        return -1;
    }

    // Get file names
    string r1csFilename = argv[1];
    string ptauFilename = argv[2];
    string zkeyFilename = argv[3];

    // Get file sizes
    struct stat fileStat;
    int iResult = stat(r1csFilename.c_str(), &fileStat);
    if (iResult != 0)
    {
        cerr << "Error: could not find r1cs file " << r1csFilename << endl;
        return -1;
    }
    uint64_t r1csFileSize = fileStat.st_size;

    iResult = stat(ptauFilename.c_str(), &fileStat);
    if (iResult != 0)
    {
        cerr << "Error: could not find ptau file " << ptauFilename << endl;
        return -1;
    }
    uint64_t ptauFileSize = fileStat.st_size;

    // Check file sizes
    if (r1csFileSize == 0)
    {
        cerr << "Error: first file " << r1csFilename << " size = 0" << endl;
        return -1;
    }
    if (ptauFileSize == 0)
    {
        cerr << "Error: second file " << ptauFilename << " size = 0" << endl;
        return -1;
    }

    auto fflonkSetup = new Fflonk::FflonkSetup(AltBn128::Engine::engine);
    fflonkSetup->generateZkey(r1csFilename, ptauFilename, zkeyFilename);

    return 0;
}