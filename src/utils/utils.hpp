#ifndef UTILS_HPP
#define UTILS_HPP

#include <sys/time.h>
#include "goldilocks_base_field.hpp"
#include "context.hpp"
#include "config.hpp"
#include "input.hpp"
#include "proof.hpp"
#include "definitions.hpp"

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

/*********/
/* Print */
/*********/

// These functions log information into the console

void printRegs(Context &ctx);
void printVars(Context &ctx);
void printMem(Context &ctx);

void printReg(Context &ctx, string name, Goldilocks::Element &V, bool h = false, bool bShort = false);
void printU64(Context &ctx, string name, uint64_t v);
void printU32(Context &ctx, string name, uint32_t v);
void printU16(Context &ctx, string name, uint16_t v);

string printFea(Context &ctx, Fea &fea);

void printBa(uint8_t * pData, uint64_t dataSize, string name);
void printBits(uint8_t * pData, uint64_t dataSize, string name);

void getMemoryInfo(MemoryInfo &info);
void printMemoryInfo(bool compact = false);
void printProcessInfo(bool compact = false);
// Prints current call stack with function names (mangled)
void printCallStack (void);

// zkmin and zkmax
#define zkmin(a, b) ((a >= b) ? b : a)
#define zkmax(a, b) ((a >= b) ? a : b)

// Returns timestamp in UTC, e.g. "2022-01-28_08:08:22_348"
string getTimestamp(void);

// Returns a new UUID, e.g. "8796757a-827c-11ec-a8a3-0242ac120002"
string getUUID (void);

// Converts a json into/from a file
void json2file(const json &j, const string &fileName);
void file2json(const string &fileName, json &j);

// Returns if file exists
bool fileExists (const string &fileName);

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
void string2File (const string & s, const string & fileName);

/*
// Convert an octal string into an hex string
bool octal2hex (const string &octalString, string &hexString);

// Convert a text with "octal_strings" in quotes, to a text with "hex_strings" in quotes
bool octalText2hexText (const string &octalText, string &hexText);
*/

#endif