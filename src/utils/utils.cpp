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
#include "zklog.hpp"

using namespace std;
using namespace std::filesystem;

void printBa(uint8_t *pData, uint64_t dataSize, string name)
{
    string s = name + " = ";
    for (uint64_t k = 0; k < dataSize; k++)
    {
        s += byte2string(pData[k]) + ":";
    }
    zklog.info(s);
}

void printBits(uint8_t *pData, uint64_t dataSize, string name)
{
    string s = name + " = ";
    for (uint64_t k = 0; k < dataSize / 8; k++)
    {
        uint8_t byte;
        bits2byte(pData + k * 8, byte);
        s += byte2string(byte) + ":";
    }
    zklog.info(s);
}

void printCallStack(void)
{
    void *callStack[100];
    size_t callStackSize = backtrace(callStack, 100);
    char **callStackSymbols = backtrace_symbols(callStack, callStackSize);
    zklog.info("CALL STACK");
    for (uint64_t i = 0; i < callStackSize; i++)
    {
        zklog.info(to_string(i) + ": call=" + callStackSymbols[i]);
    }
    free(callStackSymbols);
}

void getMemoryInfo(MemoryInfo &info)
{
    vector<string> labels{"MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "SwapCached:", "SwapTotal:", "SwapFree:"};

    ifstream meminfo = ifstream{"/proc/meminfo"};
    if (!meminfo.good())
    {
        zklog.error("Failed to get memory info");
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

void parseProcSelfStat (double &vm, double &rss)
{
    string aux;
    ifstream ifs("/proc/self/stat", ios_base::in);
    ifs >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> aux >> vm >> rss;
}

void printMemoryInfo(bool compact, const char * pMessage)
{
    string s;

    string endLine = (compact ? ", " : "\n");
    string tab = (compact ? "" : "    ");

    s = "MEMORY INFO " + (pMessage==NULL?"":string(pMessage)) + endLine;

    constexpr double factorMB = 1024;

    MemoryInfo info;
    getMemoryInfo(info);

    double vm, rss;
    parseProcSelfStat(vm, rss);
    vm /= 1024*1024;
    rss /= 1024*1024;

    s += tab + "MemTotal: "+ to_string(info.total / factorMB) + " MB" + endLine;
    s += tab + "MemFree: " + to_string(info.free / factorMB) + " MB" + endLine;
    s += tab + "MemAvailable: " + to_string(info.available / factorMB) + " MB" + endLine;
    s += tab + "Buffers: " + to_string(info.buffers / factorMB) + " MB" + endLine;
    s += tab + "Cached: " + to_string(info.cached / factorMB) + " MB" + endLine;
    s += tab + "SwapCached: " + to_string(info.swapCached / factorMB) + " MB" + endLine;
    s += tab + "SwapTotal: " + to_string(info.swapTotal / factorMB) + " MB" + endLine;
    s += tab + "SwapFree: " + to_string(info.swapFree / factorMB) + " MB" + endLine;
    s += tab + "VM: " + to_string(vm) + " MB" + endLine;
    s += tab + "RSS: " + to_string(rss) + " MB";

    zklog.info(s);
}

void printProcessInfo(bool compact)
{
    string endLine = (compact ? ", " : "\n");
    string tab = (compact ? "" : "    ");

    string s = "PROCESS INFO" + endLine;

    ifstream stat("/proc/self/stat", ios_base::in);
    if (!stat.good())
    {
        zklog.error("printProcessInfo() failed to get process stat info");
        return;
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

    s += tab + "Pid: " + to_string(pid) + endLine;
    s += tab + "User time: " + to_string((double)utime / sysconf(_SC_CLK_TCK)) + " s" + endLine;
    s += tab + "Kernel time: " + to_string((double)stime / sysconf(_SC_CLK_TCK)) + " s" + endLine;
    s += tab + "Total time: " + to_string((double)utime / sysconf(_SC_CLK_TCK) + (double)stime / sysconf(_SC_CLK_TCK)) + " s" + endLine;
    s += tab + "Num threads: " + to_string(numthreads) + endLine;
    s += tab + "Virtual mem: " + to_string(vsize / 1024 / 1024) + " MB";

    zklog.info(s);
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

string getTimestampWithPeriod(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char buf[256];
    snprintf(buf, sizeof(buf), "%ld.%06ld", tv.tv_sec, tv.tv_usec);
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
        zklog.error("json2file() failed creating output JSON file " + fileName);
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
        zklog.error("file2json() failed loading input JSON file " + fileName + "; does this file exist?");
        exitProcess();
    }
    try
    {
        inputStream >> j;
    }
    catch (exception &e)
    {
        zklog.error("file2json() failed parsing input JSON file " + fileName + " exception=" + e.what());
        exitProcess();
    }
    inputStream.close();
}

void file2json(const string &fileName, ordered_json &j)
{
    std::ifstream inputStream(fileName);
    if (!inputStream.good())
    {
        zklog.error("file2json() failed loading input JSON file " + fileName);
        exitProcess();
    }
    try
    {
        inputStream >> j;
    }
    catch (exception &e)
    {
        zklog.error("file2json() failed parsing input JSON file " + fileName + " exception=" + e.what());
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
        zklog.error("ensureDirectoryExists() system() returned: " + to_string(iResult));
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
        //zklog.info("getNumberOfFileDescriptors() i=" + to_string(i) + " file=" + i.path());
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
            zklog.error("mapFile() failed calling lstat() of file " + fileName);
            exitProcess();
        }
        if ((uint64_t)sb.st_size != size)
        {
            zklog.error("mapFile() found size of file " + fileName + " to be " + to_string(sb.st_size) + " B instead of " + to_string(size) + " B");
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
        zklog.error("mapFile() failed opening file: " + fileName);
        exitProcess();
    }

    // If output, extend the file size to the required one
    if (bOutput)
    {
        // Seek the last byte of the file
        int result = lseek(fd, size - 1, SEEK_SET);
        if (result == -1)
        {
            zklog.error("mapFile() failed calling lseek() of file: " + fileName);
            exitProcess();
        }

        // Write a 0 at the last byte of the file, to set its size; content is all zeros
        result = write(fd, "", 1);
        if (result < 0)
        {
            zklog.error("mapFile() failed calling write() of file: " + fileName);
            exitProcess();
        }
    }

    // Map the file into memory
    void *pAddress;
    pAddress = (uint8_t *)mmap(NULL, size, bOutput ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED, fd, 0);
    if (pAddress == MAP_FAILED)
    {
        zklog.error("mapFile() failed calling mmap() of file: " + fileName);
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
        zklog.error("mapFile() failed calling malloc() of size: " + to_string(size));
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
        zklog.error("unmapFile() failed calling munmap() of address=" + to_string(uint64_t(pAddress)) + " size=" + to_string(size));
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
            zklog.error("octal2hex() found an invalid octal sequence at position i=" + to_string(i) + " rest=" + octalString.substr(i));
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
            zklog.error("octalText2hexText() could not find the ending \"");
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
        zklog.error("getNetworkInfo() failed calling getifaddrs() iResult=" + to_string(iResult) + "=" + strerror(iResult));
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

        if (pEntry->ifa_addr != NULL)
        {
            sa_family_t address_family = pEntry->ifa_addr->sa_family;
            if (address_family == AF_INET) 
            {
                char buffer[INET_ADDRSTRLEN] = {0};
                inet_ntop(address_family, &((struct sockaddr_in*)(pEntry->ifa_addr))->sin_addr, buffer, INET_ADDRSTRLEN);
                if (ipAddress != "")
                {
                    ipAddress += ",";
                }
                ipAddress += buffer;    // Code for IPv4 address handling
            }
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
    }

    freeifaddrs(pIfaddrs);
}

void getStringIncrement(const string &oldString, const string &newString, uint64_t &offset, uint64_t &length)
{
    // If new string is shorter, return it all
    if (oldString.size() > newString.size())
    {
        offset = 0;
        length = newString.size();
        return;
    }
    
    // Find first different char, and assign it to offset
    int64_t i = 0;
    for (; i < (int64_t)oldString.size(); i++)
    {
        if (oldString[i] != newString[i])
        {
            break;
        }
    }
    if (i == (int64_t)oldString.size())
    {
        if (oldString.size() == newString.size()) // Identical strings
        {
            offset = 0;
            length = 0;
            return;
        }
        for (; i < (int64_t)newString.size(); i++)
        {
            if (newString[i] != 0)
            {
                break;
            }
        }
        if (i == (int64_t)newString.size()) // new string is all zeros
        {
            offset = 0;
            length = 0;
            return;
        }
    }
    offset = i;

    // If new string is longer, find last non-zero byte, if any
    if (newString.size() > oldString.size())
    {
        for (i = (int64_t)newString.size()-1; i >= (int64_t)oldString.size(); i--)
        {
            if (newString[i] != 0)
            {
                length = i + 1 - offset;
                return;
            }
        }     
    }


    // Find last different char, and calculate length
    for (i = (int64_t)oldString.size() - 1; i >= 0; i--)
    {
        if (oldString[i] != newString[i])
        {
            length = i + 1 - offset;
            return;
        }
    }

    length = 0;
}

string emptyString;
