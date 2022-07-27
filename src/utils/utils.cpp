#include <fstream>
#include <iostream>
#include <iomanip>
#include <uuid/uuid.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include "utils.hpp"
#include "scalar.hpp"
#include <openssl/md5.h>
#include <execinfo.h>

using namespace std;

void printRegs(Context &ctx)
{
    cout << "Registers:" << endl;
    printReg(ctx, "A7", ctx.pols.A7[*ctx.pStep]);
    printReg(ctx, "A6", ctx.pols.A6[*ctx.pStep]);
    printReg(ctx, "A5", ctx.pols.A5[*ctx.pStep]);
    printReg(ctx, "A4", ctx.pols.A4[*ctx.pStep]);
    printReg(ctx, "A3", ctx.pols.A3[*ctx.pStep]);
    printReg(ctx, "A2", ctx.pols.A2[*ctx.pStep]);
    printReg(ctx, "A1", ctx.pols.A1[*ctx.pStep]);
    printReg(ctx, "A0", ctx.pols.A0[*ctx.pStep]);
    printReg(ctx, "B7", ctx.pols.B7[*ctx.pStep]);
    printReg(ctx, "B6", ctx.pols.B6[*ctx.pStep]);
    printReg(ctx, "B5", ctx.pols.B5[*ctx.pStep]);
    printReg(ctx, "B4", ctx.pols.B4[*ctx.pStep]);
    printReg(ctx, "B3", ctx.pols.B3[*ctx.pStep]);
    printReg(ctx, "B2", ctx.pols.B2[*ctx.pStep]);
    printReg(ctx, "B1", ctx.pols.B1[*ctx.pStep]);
    printReg(ctx, "B0", ctx.pols.B0[*ctx.pStep]);
    printReg(ctx, "C7", ctx.pols.C7[*ctx.pStep]);
    printReg(ctx, "C6", ctx.pols.C6[*ctx.pStep]);
    printReg(ctx, "C5", ctx.pols.C5[*ctx.pStep]);
    printReg(ctx, "C4", ctx.pols.C4[*ctx.pStep]);
    printReg(ctx, "C3", ctx.pols.C3[*ctx.pStep]);
    printReg(ctx, "C2", ctx.pols.C2[*ctx.pStep]);
    printReg(ctx, "C1", ctx.pols.C1[*ctx.pStep]);
    printReg(ctx, "C0", ctx.pols.C0[*ctx.pStep]);
    printReg(ctx, "D7", ctx.pols.D7[*ctx.pStep]);
    printReg(ctx, "D6", ctx.pols.D6[*ctx.pStep]);
    printReg(ctx, "D5", ctx.pols.D5[*ctx.pStep]);
    printReg(ctx, "D4", ctx.pols.D4[*ctx.pStep]);
    printReg(ctx, "D3", ctx.pols.D3[*ctx.pStep]);
    printReg(ctx, "D2", ctx.pols.D2[*ctx.pStep]);
    printReg(ctx, "D1", ctx.pols.D1[*ctx.pStep]);
    printReg(ctx, "D0", ctx.pols.D0[*ctx.pStep]);
    printReg(ctx, "E7", ctx.pols.E7[*ctx.pStep]);
    printReg(ctx, "E6", ctx.pols.E6[*ctx.pStep]);
    printReg(ctx, "E5", ctx.pols.E5[*ctx.pStep]);
    printReg(ctx, "E4", ctx.pols.E4[*ctx.pStep]);
    printReg(ctx, "E3", ctx.pols.E3[*ctx.pStep]);
    printReg(ctx, "E2", ctx.pols.E2[*ctx.pStep]);
    printReg(ctx, "E1", ctx.pols.E1[*ctx.pStep]);
    printReg(ctx, "E0", ctx.pols.E0[*ctx.pStep]);
    printReg(ctx, "SR7", ctx.pols.SR7[*ctx.pStep]);
    printReg(ctx, "SR6", ctx.pols.SR6[*ctx.pStep]);
    printReg(ctx, "SR5", ctx.pols.SR5[*ctx.pStep]);
    printReg(ctx, "SR4", ctx.pols.SR4[*ctx.pStep]);
    printReg(ctx, "SR3", ctx.pols.SR3[*ctx.pStep]);
    printReg(ctx, "SR2", ctx.pols.SR2[*ctx.pStep]);
    printReg(ctx, "SR1", ctx.pols.SR1[*ctx.pStep]);
    printReg(ctx, "SR0", ctx.pols.SR0[*ctx.pStep]);
    printReg(ctx, "CTX", ctx.pols.CTX[*ctx.pStep]);
    printReg(ctx, "SP", ctx.pols.SP[*ctx.pStep]);
    printReg(ctx, "PC", ctx.pols.PC[*ctx.pStep]);
    printReg(ctx, "MAXMEM", ctx.pols.MAXMEM[*ctx.pStep]);
    printReg(ctx, "GAS", ctx.pols.GAS[*ctx.pStep]);
    printReg(ctx, "zkPC", ctx.pols.zkPC[*ctx.pStep]);
    Goldilocks::Element step;
    step = ctx.fr.fromU64(*ctx.pStep);
    printReg(ctx, "STEP", step, false, true);
#ifdef LOG_FILENAME
    cout << "File: " << ctx.fileName << " Line: " << ctx.line << endl;
#endif
}

void printVars(Context &ctx)
{
    cout << "Variables:" << endl;
    uint64_t i = 0;
    for (map<string, mpz_class>::iterator it = ctx.vars.begin(); it != ctx.vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << it->second.get_str(16) << endl;
        i++;
    }
}

string printFea(Context &ctx, Fea &fea)
{
    return       ctx.fr.toString(fea.fe7, 16) +
           ":" + ctx.fr.toString(fea.fe6, 16) +
           ":" + ctx.fr.toString(fea.fe5, 16) +
           ":" + ctx.fr.toString(fea.fe4, 16) +
           ":" + ctx.fr.toString(fea.fe3, 16) +
           ":" + ctx.fr.toString(fea.fe2, 16) +
           ":" + ctx.fr.toString(fea.fe1, 16) +
           ":" + ctx.fr.toString(fea.fe0, 16);
}

void printMem(Context &ctx)
{
    cout << "Memory:" << endl;
    uint64_t i = 0;
    for (map<uint64_t, Fea>::iterator it = ctx.mem.begin(); it != ctx.mem.end(); it++)
    {
        mpz_class addr(it->first);
        cout << "i: " << i << " address:" << addr.get_str(16) << " ";
        cout << printFea(ctx, it->second);
        cout << endl;
        i++;
    }
}

#ifdef USE_LOCAL_STORAGE
void printStorage(Context &ctx)
{
    uint64_t i = 0;
    for (map<Goldilocks::Element, mpz_class, CompareFe>::iterator it = ctx.sto.begin(); it != ctx.sto.end(); it++)
    {
        Goldilocks::Element fe = it->first;
        mpz_class scalar = it->second;
        cout << "Storage: " << i << " fe: " << ctx.fr.toString(fe, 16) << " scalar: " << scalar.get_str(16) << endl;
    }
}
#endif

void printReg(Context &ctx, string name, Goldilocks::Element &fe, bool h, bool bShort)
{
    cout << "    Register: " << name << " Value: " << ctx.fr.toString(fe, 16) << endl;
}

void printU64(Context &ctx, string name, uint64_t v)
{
    cout << "    U64: " << name << ":" << v << endl;
}

void printU32(Context &ctx, string name, uint32_t v)
{
    cout << "    U32: " << name << ":" << v << endl;
}

void printU16(Context &ctx, string name, uint16_t v)
{
    cout << "    U16: " << name << ":" << v << endl;
}

void printBa(uint8_t * pData, uint64_t dataSize, string name)
{
    cout << name << " = ";
    for (uint64_t k=0; k<dataSize; k++)
    {
        cout << byte2string(pData[k]) << ":";
    }
    cout << endl;
}

void printBits(uint8_t * pData, uint64_t dataSize, string name)
{
    cout << name << " = ";
    for (uint64_t k=0; k<dataSize/8; k++)
    {
        uint8_t byte;
        bits2byte(pData+k*8, byte);
        cout << byte2string(byte) << ":";
    }
    cout << endl;
}

void printCallStack (void)
{
    void *callStack[100];
    size_t callStackSize = backtrace(callStack, 100);
    char **callStackSymbols = backtrace_symbols(callStack, callStackSize);
    cout << "Call stack:" << endl;
    for (uint64_t i=0; i<callStackSize; i++)
    {
        cout << i << ": call=" << callStackSymbols[i] << endl;
    }
    free(callStackSymbols);
}

void exitProcess(void)
{
    printCallStack();
    exit(-1);
}

string getTimestamp (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char tmbuf[64], buf[256];
    strftime(tmbuf, sizeof tmbuf, "%Y%m%d_%H%M%S", gmtime(&tv.tv_sec));
    snprintf(buf, sizeof buf, "%s_%03ld", tmbuf, tv.tv_usec/1000);
    return buf;
}

string getUUID (void)
{
    char uuidString[37];
    uuid_t uuid;
    uuid_generate(uuid);
    uuid_unparse(uuid, uuidString);
    return uuidString;
}

void json2file(const json &j, const string &fileName)
{
    ofstream outputStream(fileName);
    if (!outputStream.good())
    {
        cerr << "Error: json2file() failed creating output JSON file " << fileName << endl;
        exit(-1);
    }
    outputStream << setw(4) << j << endl;
    outputStream.close();
}

void file2json(const string &fileName, json &j)
{
    std::ifstream inputStream(fileName);
    if (!inputStream.good())
    {
        cerr << "Error: file2json() failed loading input JSON file " << fileName << endl;
        exit(-1);
    }
    inputStream >> j;
    inputStream.close();
}

void * mapFileInternal (const string &fileName, uint64_t size, bool bOutput, bool bMapInputFile)
{
    // If input, check the file size is the same as the expected polsSize
    if (!bOutput)
    {
        struct stat sb;
        if ( lstat(fileName.c_str(), &sb) == -1)
        {
            cerr << "Error: mapFile() failed calling lstat() of file " << fileName << endl;
            exit(-1);
        }
        if ((uint64_t)sb.st_size != size)
        {
            cerr << "Error: mapFile() found size of file " << fileName << " to be " << sb.st_size << " B instead of " << size << " B" << endl;
            exit(-1);
        }
    }

    // Open the file withe the proper flags
    int oflags;
    if (bOutput) oflags = O_CREAT|O_RDWR|O_TRUNC;
    else         oflags = O_RDWR;
    int fd = open(fileName.c_str(), oflags, 0666);
    if (fd < 0)
    {
        cerr << "Error: mapFile() failed opening file: " << fileName << endl;
        exit(-1);
    }

    // If output, extend the file size to the required one
    if (bOutput)
    {
        // Seek the last byte of the file
        int result = lseek(fd, size-1, SEEK_SET);
        if (result == -1)
        {
            cerr << "Error: mapFile() failed calling lseek() of file: " << fileName << endl;
            exit(-1);
        }

        // Write a 0 at the last byte of the file, to set its size; content is all zeros
        result = write(fd, "", 1);
        if (result < 0)
        {
            cerr << "Error: mapFile() failed calling write() of file: " << fileName << endl;
            exit(-1);
        }
    }

    // Map the file into memory
    void * pAddress;
    pAddress = (uint8_t *)mmap( NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (pAddress == MAP_FAILED)
    {
        cerr << "Error: mapFile() failed calling mmap() of file: " << fileName << endl;
        exit(-1);
    }
    close(fd);

    
    // If mapped memory is wanted, then we are done
    if (bMapInputFile) return pAddress;

    // Allocate memory
    void * pMemAddress = malloc(size);
    if (pMemAddress == NULL)
    {
        cerr << "Error: mapFile() failed calling malloc() of size: " << size << endl;
        exit(-1);
    }

    // Copy file contents into memory
    memcpy(pMemAddress, pAddress, size);

    // Unmap file content from memory
    unmapFile(pAddress, size);

    return pMemAddress;
}

void * mapFile (const string &fileName, uint64_t size, bool bOutput)
{
    return mapFileInternal(fileName, size, bOutput, true);
}

void * copyFile (const string &fileName, uint64_t size)
{
    return mapFileInternal(fileName, size, false, false);
}

void unmapFile (void * pAddress, uint64_t size)
{
    int err = munmap(pAddress, size);
    if (err != 0)
    {
        cerr << "Error: unmapFile() failed calling munmap() of address=" << pAddress << " size=" << size << endl;
        exit(-1);
    }
}