#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

#define NEVALUATIONS 65536 //1<<16, i.e. 1<<NBITS
#define NPOLS 85 //Up to 512 in the future
#define NCONSTPOLS 57
#define ARITY 4
#define NBITS 16
#define EXTENDED_BITS 1
#define MERKLE_ARITY 16

//#define LOG_STEPS
//#define LOG_INX
//#define LOG_ADDR
//#define LOG_ASSERT
//#define LOG_SETX
//#define LOG_JMP
//#define LOG_STORAGE
//#define LOG_MEMORY
//#define LOG_HASH
//#define LOG_POLS
//#define LOG_VARIABLES // If defined, logs variable declaration, get and set actions
//#define LOG_FILENAME // If defined, logs ROM compilation file name and line number
#define LOG_TIME // If defined, logs time differences to measure performance
//#define LOG_TXS
#define LOG_SERVICE
//#define LOG_BME
//#define LOG_BME_HASH
//#define LOG_SCRIPT_OUTPUT
#define LOG_RPC_INPUT
#define LOG_RPC_OUTPUT

#define DEBUG
#ifdef DEBUG
#define zkassert(a) {if (!(a)) {cerr << "Error: assert failed: " << (#a) << endl; exit(-1);}}
#else
#define zkassert(a)
#endif

/* Executor defines */
//#define USE_LOCAL_STORAGE // If defined, use ctx.sto[], else, use smt.get()

/* Prover defines */
//#define PROVER_USE_PROOF_GOOD_JSON
//#define PROVER_INJECT_ZKIN_JSON

class Config
{
public:
    bool runServer;
    bool runServerMock;
    bool runClient;
    bool runFile;
    bool runKeccakScriptGenerator;
    bool runKeccakTest;
    uint16_t serverPort;
    uint16_t serverMockPort;
    uint16_t clientPort;
    string inputFile;
    string outputPath;
    string romFile;
    string pilFile;
    string cmPolsFile;
    string constPolsFile;
    string constantsTreeFile;
    string scriptFile;
    string starkFile;
    string verifierFile;
    string witnessFile;
    string starkVerifierFile;
    string publicFile;
    string proofFile;
    string keccakScriptFile;
    string dbHost;
    uint16_t dbPort;
    string dbUser;
    string dbPassword;
    string dbDatabaseName;
    string dbTableName;
    uint64_t cleanerPollingPeriod;
    uint64_t requestsPersistence;
    void load(json &config);
};

#endif