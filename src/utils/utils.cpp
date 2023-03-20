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
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>

using namespace std;
using namespace std::filesystem;

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
    string endLine = (compact ? ", " : "\n");

    cout << "MEMORY INFO" << endLine;

    constexpr double factorMB = 1024;

    MemoryInfo info;
    getMemoryInfo(info);

    int tab = (compact ? 0 : 15);

    cout << left << setw(tab) << "MemTotal: " << right << setw(tab) << (info.total / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "MemFree: " << right << setw(tab) << (info.free / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "MemAvailable: " << right << setw(tab) << (info.available / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "Buffers: " << right << setw(tab) << (info.buffers / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "Cached: " << right << setw(tab) << (info.cached / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "SwapCached: " << right << setw(tab) << (info.swapCached / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "SwapTotal: " << right << setw(tab) << (info.swapTotal / factorMB) << " MB" << endLine;
    cout << left << setw(tab) << "SwapFree: " << right << setw(tab) << (info.swapFree / factorMB) << " MB" << endl;
}

void printProcessInfo(bool compact)
{
    string endLine = (compact ? ", " : "\n");

    cout << "PROCESS INFO" << endLine;

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

    int tab = (compact ? 0 : 15);

    cout << left << setw(tab) << "Pid: " << right << setw(tab) << pid << endLine;
    cout << left << setw(tab) << "User time: " << right << setw(tab) << (double)utime / sysconf(_SC_CLK_TCK) << " s" << endLine;
    cout << left << setw(tab) << "Kernel time: " << right << setw(tab) << (double)stime / sysconf(_SC_CLK_TCK) << " s" << endLine;
    cout << left << setw(tab) << "Total time: " << right << setw(tab) << (double)utime / sysconf(_SC_CLK_TCK) + (double)stime / sysconf(_SC_CLK_TCK) << " s" << endLine;
    cout << left << setw(tab) << "Num threads: " << right << setw(tab) << numthreads << endLine;
    cout << left << setw(tab) << "Virtual mem: " << right << setw(tab) << vsize / 1024 / 1024 << " MB" << endl;
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

void getTimestampWithSlashes(string &timestamp, string &folder, string &file)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char tmbuf[64], buf[256];
    strftime(tmbuf, sizeof(tmbuf), "%Y%m%d_%H%M%S", gmtime(&tv.tv_sec));
    snprintf(buf, sizeof(buf), "%s_%06ld", tmbuf, tv.tv_usec);
    timestamp = buf;
    strftime(tmbuf, sizeof(tmbuf), "%Y/%m/%d/%H", gmtime(&tv.tv_sec));
    folder = tmbuf;
    strftime(tmbuf, sizeof(tmbuf), "%M%S", gmtime(&tv.tv_sec));
    snprintf(buf, sizeof(buf), "%s_%06ld", tmbuf, tv.tv_usec);
    file = buf;
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

void ensureDirectoryExists (const string &fileName)
{
    string command = "[ -d " + fileName + " ] || mkdir -p " + fileName;
    int iResult = system(command.c_str());
    if (iResult != 0)
    {
        cerr << "Error: ensureDirectoryExists() system() returned: " << to_string(iResult) << endl;
        exitProcess();
    }
}

uint64_t getNumberOfFileDescriptors (void)
{
    auto iterator = std::filesystem::directory_iterator("/proc/self/fd");
    uint64_t result = 0;
    for (auto& i : iterator)
    {
        if (i.exists())
        {
            result++;
        }
        //cout << "getNumberOfFileDescriptors() i=" << i << " file=" << i.path() << endl;
    }
    return result;
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
        oflags = O_RDONLY;
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
    pAddress = (uint8_t *)mmap(NULL, size, bOutput ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED, fd, 0);
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

void string2file (const string & s, const string & fileName)
{
    std::ofstream outfile;
    outfile.open(fileName);
    outfile << s << endl;
    outfile.close();
}

void file2string (const string &fileName, string &s)
{
    std::ifstream infile;
    infile.open(fileName);
    std::stringstream ss;
    ss << infile.rdbuf();
    s = ss.str();
    infile.close();
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

// Get IP address
void getIPAddress (string &ipAddress)
{
    ipAddress.clear();

    struct ifaddrs* pIfaddrs = NULL;

    int iResult = getifaddrs(&pIfaddrs);
    if (iResult != 0)
    {
        cerr << "Error: getNetworkInfo() failed calling getifaddrs() iResult=" << iResult << "=" << strerror(iResult) << endl;
        return;
    }

    for ( struct ifaddrs* pEntry = pIfaddrs; pEntry != NULL; pEntry = pEntry->ifa_next)
    {
        // Skip localhost
        std::string name = std::string(pEntry->ifa_name);
        if (name == "lo")
        {
            continue;
        }
        sa_family_t address_family = pEntry->ifa_addr->sa_family;

        // Report IPv4 addresses
        if (address_family == AF_INET)
        {
            if (pEntry->ifa_addr != NULL)
            {
                char buffer[INET_ADDRSTRLEN] = {0};
                inet_ntop(address_family, &((struct sockaddr_in*)(pEntry->ifa_addr))->sin_addr, buffer, INET_ADDRSTRLEN);
                if (ipAddress != "")
                {
                    ipAddress += ",";
                }
                ipAddress += buffer;
            }
        }

        // Report IPv6 addresses
        /*else if (address_family == AF_INET6)
        {
            if ( pEntry->ifa_addr != nullptr )
            {
                char buffer[INET6_ADDRSTRLEN] = {0};
                inet_ntop(address_family, &((struct sockaddr_in6*)(pEntry->ifa_addr))->sin6_addr, buffer, INET6_ADDRSTRLEN);
                ipAddress += buffer;
                ipAddress += " ";
            }
        }*/
    }

    freeifaddrs(pIfaddrs);
}