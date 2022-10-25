#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
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
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/crypto.h>

using namespace std;
using namespace std::filesystem;

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
    for (unordered_map<string, mpz_class>::iterator it = ctx.vars.begin(); it != ctx.vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << it->second.get_str(16) << endl;
        i++;
    }
}

string printFea(Context &ctx, Fea &fea)
{
    return ctx.fr.toString(fea.fe7, 16) +
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
    for (unordered_map<uint64_t, Fea>::iterator it = ctx.mem.begin(); it != ctx.mem.end(); it++)
    {
        mpz_class addr(it->first);
        cout << "i: " << i << " address:" << addr.get_str(16) << " ";
        cout << printFea(ctx, it->second);
        cout << endl;
        i++;
    }
}

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

void printBa(uint8_t *pData, uint64_t dataSize, string name)
{
    cout << name << " = ";
    for (uint64_t k = 0; k < dataSize; k++)
    {
        cout << byte2string(pData[k]) << ":";
    }
    cout << endl;
}

void printBits(uint8_t *pData, uint64_t dataSize, string name)
{
    cout << name << " = ";
    for (uint64_t k = 0; k < dataSize / 8; k++)
    {
        uint8_t byte;
        bits2byte(pData + k * 8, byte);
        cout << byte2string(byte) << ":";
    }
    cout << endl;
}

void printCallStack(void)
{
    void *callStack[100];
    size_t callStackSize = backtrace(callStack, 100);
    char **callStackSymbols = backtrace_symbols(callStack, callStackSize);
    cout << "CALL STACK" << endl;
    for (uint64_t i = 0; i < callStackSize; i++)
    {
        cout << i << ": call=" << callStackSymbols[i] << endl;
    }
    cout << endl;
    free(callStackSymbols);
}

void getMemoryInfo(MemoryInfo &info)
{
    vector<string> labels{"MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "SwapCached:", "SwapTotal:", "SwapFree:"};

    ifstream meminfo = ifstream{"/proc/meminfo"};
    if (!meminfo.good())
    {
        cout << "Failed to get memory info" << endl;
    }

    string line, label;
    uint64_t value;
    while (getline(meminfo, line))
    {
        stringstream ss{line};
        ss >> label >> value;
        if (find(labels.begin(), labels.end(), label) != labels.end())
        {
            if (label == "MemTotal:") info.total = value;
            else if (label == "MemFree:") info.free = value;
            else if (label == "MemAvailable:") info.available = value;
            else if (label == "Buffers:") info.buffers = value;
            else if (label == "Cached:") info.cached = value;
            else if (label == "SwapCached:") info.swapCached = value;
            else if (label == "SwapTotal:") info.swapTotal = value;
            else if (label == "SwapFree:") info.swapFree = value;
        }
    }
    meminfo.close();
}

void printMemoryInfo(bool compact)
{
    cout << "MEMORY INFO" << endl;

    constexpr double factorMB = 1024;

    MemoryInfo info;
    getMemoryInfo(info);

    string endLine = (compact ? ", " : "\n");
    int tab = (compact ? 0 : 15);

    cout << left << setw(tab) << "MemTotal: " << right << setw(tab) << (info.total / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "MemFree: " << right << setw(tab) << (info.free / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "MemAvailable: " << right << setw(tab) << (info.available / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "Buffers: " << right << setw(tab) << (info.buffers / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "Cached: " << right << setw(tab) << (info.cached / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "SwapCached: " << right << setw(tab) << (info.swapCached / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "SwapTotal: " << right << setw(tab) << (info.swapTotal / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "SwapFree: " << right << setw(tab) << (info.swapFree / factorMB) << " MB";

    cout << endl;
}

void printProcessInfo()
{
    cout << "PROCESS INFO" << endl;

    ifstream stat("/proc/self/stat", ios_base::in);
    if (!stat.good())
    {
        cout << "Failed to get process stat info" << endl;
    }

    string comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string cutime, cstime, priority, nice;
    string itrealvalue, starttime;

    int pid;
    unsigned long utime, stime, vsize;
    long rss, numthreads;

    stat >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >> stime >> cutime >> cstime >> priority >> nice >> numthreads >> itrealvalue >> starttime >> vsize >> rss;

    stat.close();

    cout << left << setw(15) << "Pid: " << right << setw(15) << pid << endl;
    cout << left << setw(15) << "User time: " << right << setw(15) << (double)utime / sysconf(_SC_CLK_TCK) << " s" << endl;
    cout << left << setw(15) << "Kernel time: " << right << setw(15) << (double)stime / sysconf(_SC_CLK_TCK) << " s" << endl;
    cout << left << setw(15) << "Total time: " << right << setw(15) << (double)utime / sysconf(_SC_CLK_TCK) + (double)stime / sysconf(_SC_CLK_TCK) << " s" << endl;
    cout << left << setw(15) << "Num threads: " << right << setw(15) << numthreads << endl;
    cout << left << setw(15) << "Virtual mem: " << right << setw(15) << vsize / 1024 / 1024 << " MB" << endl;
    cout << endl;
}

string getTimestamp(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char tmbuf[64], buf[256];
    strftime(tmbuf, sizeof(tmbuf), "%Y%m%d_%H%M%S", gmtime(&tv.tv_sec));
    snprintf(buf, sizeof(buf), "%s_%06ld", tmbuf, tv.tv_usec);
    return buf;
}

string getUUID(void)
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
        exitProcess();
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
        exitProcess();
    }
    try
    {
        inputStream >> j;
    }
    catch (exception &e)
    {
        cerr << "Error: file2json() failed parsing input JSON file " << fileName << " exception=" << e.what() << endl;
        exitProcess();
    }
    inputStream.close();
}

void file2json(const string &fileName, ordered_json &j)
{
    std::ifstream inputStream(fileName);
    if (!inputStream.good())
    {
        cerr << "Error: file2json() failed loading input JSON file " << fileName << endl;
        exitProcess();
    }
    try
    {
        inputStream >> j;
    }
    catch (exception &e)
    {
        cerr << "Error: file2json() failed parsing input JSON file " << fileName << " exception=" << e.what() << endl;
        exitProcess();
    }
    inputStream.close();
}

bool fileExists (const string &fileName)
{
    struct stat fileStat;
    int iResult = stat( fileName.c_str(), &fileStat);
    return (iResult == 0);
}

void *mapFileInternal(const string &fileName, uint64_t size, bool bOutput, bool bMapInputFile)
{
    // If input, check the file size is the same as the expected polsSize
    if (!bOutput)
    {
        struct stat sb;
        if (lstat(fileName.c_str(), &sb) == -1)
        {
            cerr << "Error: mapFile() failed calling lstat() of file " << fileName << endl;
            exitProcess();
        }
        if ((uint64_t)sb.st_size != size)
        {
            cerr << "Error: mapFile() found size of file " << fileName << " to be " << sb.st_size << " B instead of " << size << " B" << endl;
            exitProcess();
        }
    }

    // Open the file withe the proper flags
    int oflags;
    if (bOutput)
        oflags = O_CREAT | O_RDWR | O_TRUNC;
    else
        oflags = O_RDWR;
    int fd = open(fileName.c_str(), oflags, 0666);
    if (fd < 0)
    {
        cerr << "Error: mapFile() failed opening file: " << fileName << endl;
        exitProcess();
    }

    // If output, extend the file size to the required one
    if (bOutput)
    {
        // Seek the last byte of the file
        int result = lseek(fd, size - 1, SEEK_SET);
        if (result == -1)
        {
            cerr << "Error: mapFile() failed calling lseek() of file: " << fileName << endl;
            exitProcess();
        }

        // Write a 0 at the last byte of the file, to set its size; content is all zeros
        result = write(fd, "", 1);
        if (result < 0)
        {
            cerr << "Error: mapFile() failed calling write() of file: " << fileName << endl;
            exitProcess();
        }
    }

    // Map the file into memory
    void *pAddress;
    pAddress = (uint8_t *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pAddress == MAP_FAILED)
    {
        cerr << "Error: mapFile() failed calling mmap() of file: " << fileName << endl;
        exitProcess();
    }
    close(fd);

    // If mapped memory is wanted, then we are done
    if (bMapInputFile)
        return pAddress;

    // Allocate memory
    void *pMemAddress = malloc(size);
    if (pMemAddress == NULL)
    {
        cerr << "Error: mapFile() failed calling malloc() of size: " << size << endl;
        exitProcess();
    }

    // Copy file contents into memory
    memcpy(pMemAddress, pAddress, size);

    // Unmap file content from memory
    unmapFile(pAddress, size);

    return pMemAddress;
}

void *mapFile(const string &fileName, uint64_t size, bool bOutput)
{
    return mapFileInternal(fileName, size, bOutput, true);
}

void *copyFile(const string &fileName, uint64_t size)
{
    return mapFileInternal(fileName, size, false, false);
}

void unmapFile(void *pAddress, uint64_t size)
{
    int err = munmap(pAddress, size);
    if (err != 0)
    {
        cerr << "Error: unmapFile() failed calling munmap() of address=" << pAddress << " size=" << size << endl;
        exitProcess();
    }
}

string sha256(string str)
{
    long len = 0;
    unsigned char *bin = OPENSSL_hexstr2buf(str.c_str(), &len);

    // digest the blob
    const EVP_MD *md_algo = EVP_sha256();
    unsigned int md_len = EVP_MD_size(md_algo);
    std::vector<unsigned char> md(md_len);
    EVP_Digest(bin, len, md.data(), &md_len, md_algo, nullptr);

    // free the input data.
    OPENSSL_free(bin);
    char mdString[SHA256_DIGEST_LENGTH * 2 + 1];
    int i;
    for (i = 0; i < SHA256_DIGEST_LENGTH; i++)
        sprintf(&mdString[i * 2], "%02x", (unsigned int)md[i]);
    return mdString;
}

vector<string> getFolderFiles (string folder, bool sorted)
{
    vector<string> vfiles;
    
    for (directory_entry p: directory_iterator(folder))
    {
        vfiles.push_back(p.path().filename());
    }
    
    // Sort files alphabetically
    if (sorted) sort(vfiles.begin(),vfiles.end());    

    return vfiles;
}

uint64_t getNumberOfCores (void)
{
    return omp_get_num_procs();
}

void string2File (const string & s, const string & fileName)
{
    std::ofstream outfile;
    outfile.open(fileName);
    outfile << s << endl;
    outfile.close();
}

/*
// Convert an octal string into an hex string
bool octal2hex (const string &octalString, string &hexString)
{
    hexString.clear();

    uint64_t octalStringSize = octalString.size();
    for (uint64_t i=0; i<octalStringSize; i++)
    {
        char c = octalString[i];
        if (c != '\\')
        {
            hexString += byte2string(c);
            continue;
        }
        if (octalStringSize - i < 3)
        {
            cerr << "Error: octal2hex() found an invalid octal sequence at position i=" << i << " rest=" << octalString.substr(i) << endl;
            return false;
        }
        i++;
        c = char2byte(octalString[i]);
        c = c << 3;
        i++;
        c += char2byte(octalString[i]);
        c = c << 3;
        i++;
        c += char2byte(octalString[i]);
    }
    return true;
}

// Convert a text with "octal_strings" in quotes, to a text with "hex_strings" in quotes
bool octalText2hexText (const string &octalText, string &hexText)
{
    hexText.clear();

    size_t currentPosition = 0;
    size_t stringBegin;
    size_t stringEnd;

    do
    {
        stringBegin = octalText.find('"', currentPosition);
        if (stringBegin == string::npos)
        {
            hexText = octalText.substr(currentPosition);
            break;
        }
        stringEnd = octalText.find('"', stringBegin + 1);
        if (stringEnd == string::npos)
        {
            cerr << "Error: octalText2hexText() could not find the ending \"" << endl;
            hexText = octalText; // Copy it as it is
            return false;
        }
        hexText += octalText.substr(currentPosition, stringBegin+1);
        string aux;
        octal2hex(octalText.substr(stringBegin+1, stringEnd), aux);
        hexText += aux;
        hexText += "\"";
        currentPosition = stringEnd + 1;
    } while (true);
    return true;
}
*/