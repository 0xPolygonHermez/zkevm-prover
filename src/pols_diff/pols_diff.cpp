#include <iostream>
#include <fstream>
#include <string>
#include <gmpxx.h>
#include <cstdio>
#include <sys/stat.h>
#include "../config/definitions.hpp"
#include "../main_sm/fork_9/pols_generated/commit_pols.hpp"

using namespace std;
using namespace fork_9;

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
    if (file2 == NULL)
    {
        cerr << "Error calling fopen() of secondFileName=" << secondFileName << endl;
        fclose(file1);
        return -1;
    }

#define FILE_BUFFER_SIZE 1024
    uint8_t buffer1[FILE_BUFFER_SIZE] = { 0 };
    uint8_t buffer2[FILE_BUFFER_SIZE] = { 0 };
    uint64_t readBytes = 0;
    while (readBytes < firstFileSize)
    {
        uint64_t numberOfBytesToRead = min(firstFileSize - readBytes, FILE_BUFFER_SIZE);
        size_t result = fread(buffer1, numberOfBytesToRead, 1, file1);
        if (result != 1)
        {
            cerr << "Error calling fread() of firstFileName=" << firstFileName << " result=" << result << endl;
            fclose(file1);
            fclose(file2);
            return -1;
        }
        result = fread(buffer2, numberOfBytesToRead, 1, file2);
        if (result != 1)
        {
            cerr << "Error calling fread() of secondFileName=" << secondFileName << " result=" << result << endl;
            fclose(file1);
            fclose(file2);
            return -1;
        }
        for (uint64_t i=0; i<numberOfBytesToRead; i++)
        {
            if (buffer1[i] != buffer2[i])
            {
                uint64_t bytePosition = readBytes + i;
                uint64_t polNumber = bytePosition % (CommitPols::numPols() * 8);
                uint64_t evaluation = bytePosition / (CommitPols::numPols() * 8);

                // Read zkPC for this evaluation in file 1
                CommitPols commitPols(0, CommitPols::pilDegree());
                uint64_t zkPCOffset = (uint64_t)(commitPols.Main.zkPC.address() + evaluation*CommitPols::numPols());
                fpos_t pos1;
                fgetpos(file1, &pos1);
                fseek(file1, zkPCOffset, SEEK_SET);
                uint64_t zkPC;
                fread(&zkPC, 8, 1, file1);
                fsetpos(file1, &pos1);
                
                cout << "pos=" << bytePosition << " file1=" << uint64_t(buffer1[i]) << " file2=" << uint64_t(buffer2[i]) << " " << address2CommitPolName(polNumber) << " eval=" << evaluation << " zkPCOffset=" << zkPCOffset << " zkPC1=" << zkPC << endl;
            }
        }
        readBytes += numberOfBytesToRead;
    }

    fclose(file1);
    fclose(file2);

    return 0;
}