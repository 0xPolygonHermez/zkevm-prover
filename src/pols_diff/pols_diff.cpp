#include <iostream>
#include <fstream>
#include <string>
#include <gmpxx.h>
#include <cstdio>
#include <sys/stat.h>
#include "../config/definitions.hpp"
#include "../main_sm/fork_7/pols_generated/commit_pols.hpp"

using namespace std;

// Fork namespace
const string forkNamespace = PROVER_FORK_NAMESPACE_STRING;

#define min(a,b) ((a)<(b)?(a):(b))

int main (int argc, char **argv)
{
    cout << "Pols diff" << endl;

    // Check arguments list
    if (argc != 3)
    {
        cerr << "Error: expected 2 arguments but got " << argc - 1 << " Usage: polsDiff <commitPolsFile1> <commitPolsFile2>" << endl;
        return -1;
    }

    // Get file names
    string firstFileName = argv[1];
    string secondFileName = argv[2];

    // Get file sizes
    struct stat fileStat;
    int iResult = stat( firstFileName.c_str(), &fileStat);
    if (iResult != 0)
    {
        cerr << "Error: could not find file " << firstFileName << endl;
        return -1;
    }
    uint64_t firstFileSize = fileStat.st_size;
    iResult = stat( secondFileName.c_str(), &fileStat);
    if (iResult != 0)
    {
        cerr << "Error: could not find file " << secondFileName << endl;
        return -1;
    }
    uint64_t secondFileSize = fileStat.st_size;

    // Check file sizes
    if (firstFileSize == 0)
    {
        cerr << "Error: first file " << firstFileName << " size = 0" << endl;
        return -1;
    }
    if (secondFileSize == 0)
    {
        cerr << "Error: second file " << secondFileSize << " size = 0" << endl;
        return -1;
    }
    if (firstFileSize != secondFileSize)
    {
        cerr << "Error: file sizes do not match: firstFileSize=" << firstFileSize << " secondFileSize=" << secondFileSize << endl;
        return -1;
    }

    FILE * file1;
    file1 = fopen(firstFileName.c_str(),"rb");
    if (file1 == NULL)
    {
        cerr << "Error calling fopen() of firstFileName=" << firstFileName << endl;
        return -1;
    }

    FILE * file2;
    file2 = fopen(secondFileName.c_str(),"rb");
    if (file1 == NULL)
    {
        cerr << "Error calling fopen() of secondFileName=" << secondFileName << endl;
        return -1;
    }

#define FILE_BUFFER_SIZE 1024
    uint8_t buffer1[FILE_BUFFER_SIZE];
    uint8_t buffer2[FILE_BUFFER_SIZE];
    uint64_t readBytes = 0;
    while (readBytes < firstFileSize)
    {
        uint64_t numberOfBytesToRead = min(firstFileSize - readBytes, FILE_BUFFER_SIZE);
        size_t result = fread(buffer1, numberOfBytesToRead, 1, file1);
        if (result != 1)
        {
            cerr << "Error calling fread() of firstFileName=" << firstFileName << " result=" << result << endl;
            return -1;
        }
        result = fread(buffer2, numberOfBytesToRead, 1, file2);
        if (result != 1)
        {
            cerr << "Error calling fread() of secondFileName=" << secondFileName << " result=" << result << endl;
            return -1;
        }
        for (uint64_t i=0; i<numberOfBytesToRead; i++)
        {
            if (buffer1[i] != buffer2[i])
            {
                uint64_t bytePosition = readBytes + i;
                uint64_t polNumber = bytePosition % (fork_7::CommitPols::numPols() * 8);
                uint64_t evaluation = bytePosition / (fork_7::CommitPols::numPols() * 8);
                cout << "pos=" << bytePosition << " file1=" << uint64_t(buffer1[i]) << " file2=" << uint64_t(buffer2[i]) << " " << fork_7::address2CommitPolName(polNumber) << " eval=" << evaluation << endl;
            }
        }
        readBytes += numberOfBytesToRead;
    }

    return 0;
}