#ifndef UTILS_HPP
#define UTILS_HPP

#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "input.hpp"
#include "proof_fflonk.hpp"
#include "definitions.hpp"

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

struct MemoryInfo {
    uint64_t total;
    uint64_t free;
    uint64_t available;
    uint64_t buffers;
    uint64_t cached;
    uint64_t swapCached;
    uint64_t swapTotal;
    uint64_t swapFree;
};

void printBa(uint8_t * pData, uint64_t dataSize, string name);
void printBits(uint8_t * pData, uint64_t dataSize, string name);

void getMemoryInfo(MemoryInfo &info);
void printMemoryInfo(bool compact = false, const char * pMessage = NULL);
void printProcessInfo(bool compact = false);
// Prints current call stack with function names (mangled)
void printCallStack (void);

// Returns timestamp in UTC, e.g. "20230110_173200_128863"
string getTimestamp(void);

// Returns timestamp in UTC, e.g. "1653327845.128863"
string getTimestampWithPeriod(void);

// Returns timestamp in UTC with slashes up to the hour, e.g. "2023/01/10/17/3200_128863"
void getTimestampWithSlashes(string &timestamp, string &folder, string &file);

// Returns a new UUID, e.g. "8796757a-827c-11ec-a8a3-0242ac120002"
string getUUID (void);

// Converts a json into/from a file
void json2file(const json &j, const string &fileName);
void file2json(const string &fileName, json &j);
void file2json(const string &fileName, ordered_json &j);

// Returns if file exists
bool fileExists (const string &fileName);

// Ensure directory exists
void ensureDirectoryExists (const string &fileName);

// Get number of open file descriptors
uint64_t getNumberOfFileDescriptors (void);

// Maps memory into a file
void * mapFile (const string &fileName, uint64_t size, bool bOutput);
void unmapFile (void * pAddress, uint64_t size);

// Copies file content into memory; use free after use
void * copyFile (const string &fileName, uint64_t size);

// Compute the sha256 hash of a string
string sha256(string str);

// Get files from a folder
vector<string> getFolderFiles (string folder, bool sorted);

// Gets the nubmer of cores in the system processor
uint64_t getNumberOfCores (void);

// Save a string into a file
void string2file (const string &s, const string &fileName);

// Copy a file content into a string
void file2string (const string &fileName, string &s);

/*
// Convert an octal string into an hex string
bool octal2hex (const string &octalString, string &hexString);

// Convert a text with "octal_strings" in quotes, to a text with "hex_strings" in quotes
bool octalText2hexText (const string &octalText, string &hexText);
*/

// Get IP address
void getIPAddress (string &ipAddress);

// Gets the incremental of a string (old) vs. another (old), i.e. the set of chars that are different
// If the new string is shorter than the old string, it returns the whole new string
void getStringIncrement(const string &oldString, const string &newString, uint64_t &offset, uint64_t &length);

extern string emptyString;

#endif