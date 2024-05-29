#ifndef FULL_TRACER_INTERFACE_HPP
#define FULL_TRACER_INTERFACE_HPP

#include <string>
#include <unordered_map>
#include "zkglobals.hpp"

using namespace std;

// Tracer service to output the logs of a batch of transactions. A complete log is created with all the transactions embedded
// for each batch and also a log is created for each transaction separatedly. The events are triggered from the zkrom and handled
// from the zkprover

class OpcodeContract
{
public:
    string address;
    string caller;
    mpz_class value;
    string data;
    uint64_t gas;
    string type;
    OpcodeContract() : value(0), gas(0) {};
};

class Opcode
{
public:
    uint64_t gas;
    int64_t gas_cost;
    string state_root;
    uint64_t depth;
    uint64_t pc;
    uint8_t op;
    const char * opcode;
    uint64_t gas_refund;
    string error;
    OpcodeContract contract;
    vector<mpz_class> stack;
    string memory;
    uint64_t memory_size;
    uint64_t memory_offset;
    unordered_map<string,string> storage;
    vector<string> return_data;
    struct timeval startTime;
    uint64_t duration;
    Opcode() : gas(0), gas_cost(0), depth(0), pc(0), op(0), opcode(NULL), gas_refund(0), memory_size(0), memory_offset(0), startTime({0,0}), duration(0) {};
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
    Log() : batch_number(0), tx_index(0), index(0) {};
};

class LogV2
{
public:
    string address;
    uint64_t block_number;
    string block_hash;
    string tx_hash;
    string tx_hash_l2;
    uint64_t tx_index;
    string batch_hash;
    uint64_t index;
    vector<string> data;
    vector<string> topics;
    LogV2() : block_number(0), tx_index(0), index(0) {};
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
    mpz_class value;
    string batch;
    string output;
    mpz_class gas_price;
    uint64_t chainId; // TODO: delete; only used by fork_0
    string old_state_root;
    uint64_t execution_time; // In us
    string error;
    vector<Log> logs;
    TxTraceContext() : gas(0), gas_used(0), execution_time(0) {};
};

class TxTraceContextV2
{
public:
    string type;
    string from;
    string to;
    string data;
    uint64_t gas;
    uint64_t gas_used;
    mpz_class value;
    string batch;
    string output;
    mpz_class gas_price;
    uint64_t chainId; // TODO: delete; only used by fork_0
    string old_state_root;
    uint64_t execution_time; // In us
    string error;
    vector<LogV2> logs;
    uint64_t txIndex;
    TxTraceContextV2() : gas(0), gas_used(0), execution_time(0), txIndex(0) {};
};

class FullTrace
{
public:
    TxTraceContext context;
    vector<Opcode> steps;
};

class FullTraceV2
{
public:
    TxTraceContextV2 context;
    vector<Opcode> steps;
};

class Response
{
public:
    FullTrace full_trace;
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
    string effective_gas_price;
    uint32_t effective_percentage;
    bool has_gasprice_opcode;
    bool has_balance_opcode;
    Response() : type(0), gas_left(0), gas_used(0), gas_refunded(0), effective_percentage(0), has_gasprice_opcode(false), has_balance_opcode(false) {};
};

class ResponseV2
{
public:
    FullTraceV2 full_trace;
    string tx_hash;
    string tx_hash_l2;
    string block_hash;
    uint64_t block_number;
    string rlp_tx;
    uint64_t type;
    string return_value;
    uint64_t gas_left;
    uint64_t gas_used;
    uint64_t gas_refunded;
    uint64_t cumulative_gas_used;
    string error;
    string create_address;
    string state_root;
    vector<LogV2> logs;
    string effective_gas_price;
    uint32_t effective_percentage;
    bool has_gasprice_opcode;
    bool has_balance_opcode;
    uint32_t status;
    ResponseV2() : block_number(0), type(0), gas_left(0), gas_used(0), gas_refunded(0), cumulative_gas_used(0), effective_percentage(0), has_gasprice_opcode(false), has_balance_opcode(false), status(0) {};
};

class FinalTrace
{
public:
    bool bInitialized;
    string new_state_root;
    string new_local_exit_root;
    string newAccInputHash;
    string new_acc_input_hash;
    uint64_t numBatch;
    uint64_t cumulative_gas_used;
    vector<Response> responses;
    string error;
    FinalTrace() : bInitialized(false), numBatch(0), cumulative_gas_used(0) {};
};

class Block
{
public:
    string parent_hash;
    string coinbase;
    uint64_t gas_limit;
    uint64_t gas_used;
    string block_hash;
    uint64_t block_number;
    string receipts_root;
    uint64_t timestamp;
    string ger;
    string block_info_root;
    string block_hash_l1;
    vector<ResponseV2> responses;
    vector<LogV2> logs;
    bool initialized;
    string error;
    uint64_t ctx;
    Block() : gas_limit(0), block_number(0), timestamp(0), initialized(false), ctx(0) {};
};

class FinalTraceV2
{
public:
    bool bInitialized;
    string new_state_root;
    string new_local_exit_root;
    string newAccInputHash;
    string new_acc_input_hash;
    uint64_t numBatch;
    uint64_t cumulative_gas_used;
    uint64_t gas_used;
    vector<Block> block_responses;
    bool invalid_batch;
    string error;
    FinalTraceV2() : bInitialized(false), numBatch(0), cumulative_gas_used(0), gas_used(0), invalid_batch(false) {};
};

class InfoReadWrite
{
public:
    string nonce;
    Goldilocks::Element nonceKey[4];
    string balance;
    Goldilocks::Element balanceKey[4];
    string sc_code;
    unordered_map<string, string> sc_storage;
    string sc_length;
    InfoReadWrite()
    {
        // Reset nonce key
        nonceKey[0] = fr.zero();
        nonceKey[1] = fr.zero();
        nonceKey[2] = fr.zero();
        nonceKey[3] = fr.zero();

        // Reset balance key
        balanceKey[0] = fr.zero();
        balanceKey[1] = fr.zero();
        balanceKey[2] = fr.zero();
        balanceKey[3] = fr.zero();
    }
};

class TxGAS
{
public:
    uint64_t forwarded;
    uint64_t remaining;
};

class ReturnFromCreate
{
public:
    bool enabled;
    uint64_t originCTX;
    uint64_t createCTX;
    vector<string> returnValue;
    ReturnFromCreate() : enabled(false), originCTX(0), createCTX(0) {};
};

class FullTracerInterface
{
public:
    virtual ~FullTracerInterface(){};
    virtual uint64_t get_cumulative_gas_used(void) = 0;
    virtual uint64_t get_gas_used(void) = 0;
    virtual string & get_new_state_root(void) = 0;
    virtual string & get_new_acc_input_hash(void) = 0;
    virtual string & get_new_local_exit_root(void) = 0;
    virtual unordered_map<string, InfoReadWrite> * get_read_write_addresses(void) = 0;
    virtual vector<Response> & get_responses(void) = 0;
    virtual vector<Block> & get_block_responses(void) = 0;
    virtual vector<Opcode> & get_info(void) = 0;
    virtual uint64_t get_block_number(void) = 0; // block number = 0, 1, 2...
    virtual uint64_t get_tx_number(void) = 0; // tx number = 0, 1, 2...
    virtual string & get_error(void) = 0;
    virtual bool get_invalid_batch(void) = 0;
};

#endif