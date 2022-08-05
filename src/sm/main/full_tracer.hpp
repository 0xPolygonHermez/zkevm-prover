#ifndef FULL_TRACER_HPP
#define FULL_TRACER_HPP

#include <gmpxx.h>
#include "rom_command.hpp"
#include "goldilocks_base_field.hpp"

// Tracer service to output the logs of a batch of transactions. A complete log is created with all the transactions embedded
// for each batch and also a log is created for each transaction separatedly. The events are triggered from the zkrom and handled
// from the zkprover

class Context;

class OpcodeContract
{
public:
    string address;
    string caller;
    uint64_t value;
    string data;
    uint64_t gas;
};

class Opcode
{
public:
    uint64_t remaining_gas;
    int64_t gasCost;
    string state_root;
    uint64_t depth;
    uint64_t pc;
    uint8_t op;
    string opcode;
    uint64_t refund;
    string error;
    OpcodeContract contract;
    vector<uint64_t> stack;
    string memory;
    uint64_t memory_size;
    map<string,string> storage;
    vector<string> return_data;
    Opcode() : remaining_gas(0), gasCost(0) {};
};

class Log
{
public:
    string address;
    uint64_t batch_number;
    string tx_hash;
    uint64_t tx_index;
    string batch_hash;
    uint64_t index;
    vector<string> data;
    vector<string> topics;
};

class TxTraceContext
{
public:
    string type;
    string from;
    string to;
    string data;
    uint64_t gas;
    uint64_t gas_used;
    uint64_t value;
    string batch;
    string output;
    uint64_t nonce;
    uint64_t gasPrice;
    uint64_t chainId;
    string old_state_root;
    //string newStateRoot;
    uint64_t execution_time; // In us
    string error;
    vector<Log> logs;
};

/*class TxTrace
{
public:
    string to;
    TxTraceContext context;
    vector<Opcode> steps;
    TxTrace() : to("0x00") {};
};*/

class CallTrace
{
public:
    TxTraceContext context;
    vector<Opcode> steps;
};

class Response
{
public:
    CallTrace call_trace;
    string tx_hash;
    string rlp_tx;
    uint64_t type;
    string return_value;
    uint64_t gas_left;
    uint64_t gas_used;
    uint64_t gas_refunded;
    string error;
    string create_address;
    string state_root;
    vector<Log> logs;
    vector<Opcode> execution_trace;
    bool unprocessed_transaction;
};

class FinalTrace
{
public:
    bool bInitialized;
    string batchHash;
    string old_state_root;
    string new_state_root;
    string new_local_exit_root;
    string globalHash;
    uint64_t numBatch;
    uint64_t timestamp;
    string sequencerAddr;
    uint64_t cumulative_gas_used;
    vector<Response> responses;
    FinalTrace() : bInitialized(false) {};
};

class FullTracer
{
public:
    Goldilocks &fr;
    uint64_t depth;
    uint64_t initGas;
    map<uint64_t,map<string,string>> deltaStorage;
    FinalTrace finalTrace;
    map<uint64_t,uint64_t> txGAS;
    uint64_t txCount;
    uint64_t txTime; // in us
    vector<Opcode> info; // Opcode step traces of the all the processed tx
    vector<vector<uint64_t>> fullStack;// Stack of the transaction
    uint64_t accBatchGas;
    map<uint64_t,map<uint64_t,Log>> logs;
    vector<Opcode> call_trace;
    vector<Opcode> execution_trace;
private:
    void onError (Context &ctx, const RomCommand &cmd);
    void onStoreLog (Context &ctx, const RomCommand &cmd);
    void onProcessTx (Context &ctx, const RomCommand &cmd);
    void onUpdateStorage (Context &ctx, const RomCommand &cmd);
    void onFinishTx (Context &ctx, const RomCommand &cmd);
    void onStartBatch (Context &ctx, const RomCommand &cmd);
    void onFinishBatch (Context &ctx, const RomCommand &cmd);
    void onOpcode (Context &ctx, const RomCommand &cmd);
    void getFromMemory(Context &ctx, mpz_class &offset, mpz_class &length, string &result);
    void getVarFromCtx(Context &ctx, bool global, const char * pVarLabel, mpz_class &result);
    void getCalldataFromStack (Context &ctx, uint64_t offset, uint64_t length, string &result);
    void getRegFromCtx(Context &ctx, string &reg, mpz_class &result);
    uint64_t findOffsetLabel (Context &ctx, const char * pLabel);
    uint64_t getCurrentTime (void);
    void getTransactionHash(string &to, uint64_t value, uint64_t nonce, uint64_t gasLimit, uint64_t gasPrice, string &data, mpz_class &r, mpz_class &s, uint64_t v, string &txHash, string &rlpTx);
public:
    FullTracer(Goldilocks &fr) : fr(fr), depth(1), txCount(0), txTime(0)
    {
        depth = 1;
        initGas = 0;
        txCount = 0;
        txTime = 0;
    };
    void handleEvent (Context &ctx, const RomCommand &cmd);
};

#endif