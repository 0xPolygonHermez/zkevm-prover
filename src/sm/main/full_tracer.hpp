#ifndef FULL_TRACER_HPP
#define FULL_TRACER_HPP

#include <gmpxx.h>
#include "rom_command.hpp"

// Tracer service to output the logs of a batch of transactions. A complete log is created with all the transactions embedded
// for each batch and also a log is created for each transaction separatedly. The events are triggered from the zkrom and handled
// from the zkprover

class Context;

class OpcodeContract
{
public:
    string address;
    string caller;
    string value;
    string input;
    string gas;
};

class Opcode
{
public:
    uint64_t gas;
    int64_t gasCost;
    string stateRoot;
    uint64_t depth;
    uint64_t pc;
    uint8_t op;
    string opcode;
    uint64_t refund;
    string error;
    OpcodeContract contract;
    vector<string> stack;
    vector<string> memory;
    map<string,string> storage;
    Opcode() : gas(0), gasCost(0) {};
};

class TxTraceContext
{
public:
    string type;
    string from;
    string to;
    string input;
    uint64_t gas;
    uint64_t gasUsed;
    string value;
    string output;
    uint64_t nonce;
    string gasPrice;
    uint64_t chainId;
    string oldStateRoot;
    string newStateRoot;
    uint64_t time; // In us
};

class TxTrace
{
public:
    string to;
    TxTraceContext context;
    vector<Opcode> steps;
    TxTrace() : to("0x00") {};
};

class FinalTrace
{
public:
    bool bInitialized;
    string batchHash;
    string oldStateRoot;
    string globalHash;
    uint64_t numBatch;
    uint64_t timestamp;
    string sequencerAddr;
    vector<TxTrace> txs;
    FinalTrace() : bInitialized(false) {};
};

class FullTracer
{
private:
    uint64_t depth;
    map<string, uint64_t> labels;
    map<uint64_t,map<string,string>> deltaStorage;
    FinalTrace finalTrace;
    map<uint64_t,string> txGAS;
    uint64_t txCount;
    uint64_t txTime; // in us
    vector<Opcode> info;
    vector<Opcode> trace;
    vector<vector<string>> fullStack;
    void onProcessTx (Context &ctx, const RomCommand &cmd);
    void onUpdateStorage (Context &ctx, const RomCommand &cmd);
    void onFinishTx (Context &ctx, const RomCommand &cmd);
    void onStartBatch (Context &ctx, const RomCommand &cmd);
    void onFinishBatch (Context &ctx, const RomCommand &cmd);
    void onOpcode (Context &ctx, const RomCommand &cmd);
    void getVarFromCtx(Context &ctx, bool global, string &varLabel, mpz_class &result);
    void getCalldataFromStack (Context &ctx, string &result);
    void getRegFromCtx(Context &ctx, string &reg, mpz_class &result);
    uint64_t findOffsetLabel (Context &ctx, string &label);
    uint64_t getCurrentTime (void);
public:
    FullTracer() : depth(1), txCount(0), txTime(0) {};
    void handleEvent (Context &ctx, const RomCommand &cmd);
};

#endif