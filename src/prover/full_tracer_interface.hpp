#ifndef FULL_TRACER_INTERFACE_HPP
#define FULL_TRACER_INTERFACE_HPP

#include <string>
#include <unordered_map>

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
    string effective_gas_price;
    uint32_t effective_percentage;
    bool has_gasprice_opcode;
    bool has_balance_opcode;
    Response() : type(0), gas_left(0), gas_used(0), gas_refunded(0), effective_percentage(0), has_gasprice_opcode(false), has_balance_opcode(false) {};
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

class InfoReadWrite
{
public:
    string nonce;
    string balance;
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
    virtual string & get_new_state_root(void) = 0;
    virtual string & get_new_acc_input_hash(void) = 0;
    virtual string & get_new_local_exit_root(void) = 0;
    virtual unordered_map<string, InfoReadWrite> * get_read_write_addresses(void) = 0;
    virtual vector<Response> & get_responses(void) = 0;
    virtual vector<Opcode> & get_info(void) = 0;
    virtual uint64_t get_tx_number(void) = 0; // tx number = 0, 1, 2...
};

#endif