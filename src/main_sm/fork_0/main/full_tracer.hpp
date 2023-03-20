#ifndef FULL_TRACER_HPP_fork_0
#define FULL_TRACER_HPP_fork_0

#include <gmpxx.h>
#include <unordered_map>
#include "main_sm/fork_0/main/context.hpp"
#include "main_sm/fork_0/main/rom_command.hpp"
#include "main_sm/fork_0/main/rom_line.hpp"
#include "main_sm/fork_0/main/time_metric.hpp"
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "full_tracer_interface.hpp"

namespace fork_0
{

class Context;

class FullTracer: public FullTracerInterface
{
public:
    Goldilocks &fr;
    uint64_t depth;
    uint64_t initGas;
    unordered_map<uint64_t,unordered_map<string,string>> deltaStorage;
    FinalTrace finalTrace;
    unordered_map<uint64_t,uint64_t> txGAS;
    uint64_t txCount;
    uint64_t txTime; // in us
    vector<Opcode> info; // Opcode step traces of the all the processed tx
    vector<vector<mpz_class>> fullStack;// Stack of the transaction
    uint64_t accBatchGas;
    unordered_map<uint64_t,unordered_map<uint64_t,Log>> logs;
    vector<Opcode> call_trace; // TODO: Can we remove this attribute?
    vector<Opcode> execution_trace; // TODO: Can we remove this attribute?
    string lastError;
#ifdef LOG_TIME_STATISTICS
    TimeMetricStorage tms;
    struct timeval t;
    TimeMetricStorage tmsop;
    struct timeval top;
#endif
public:
    void onError (Context &ctx, const RomCommand &cmd);
    void onStoreLog (Context &ctx, const RomCommand &cmd);
    void onProcessTx (Context &ctx, const RomCommand &cmd);
    void onUpdateStorage (Context &ctx, const RomCommand &cmd);
    void onFinishTx (Context &ctx, const RomCommand &cmd);
    void onStartBatch (Context &ctx, const RomCommand &cmd);
    void onFinishBatch (Context &ctx, const RomCommand &cmd);
    void onOpcode (Context &ctx, const RomCommand &cmd);

    FullTracer(Goldilocks &fr) : fr(fr), depth(1), initGas(0), txCount(0), txTime(0), accBatchGas(0) { };
    ~FullTracer()
    {
#ifdef LOG_TIME_STATISTICS
        tms.print("FullTracer");
        tmsop.print("FullTracer onOpcode");
#endif    
    }
    
    void handleEvent (Context &ctx, const RomCommand &cmd);

    FullTracer & operator =(const FullTracer & other)
    {
        depth           = other.depth;
        initGas         = other.initGas;
        deltaStorage    = other.deltaStorage;
        finalTrace      = other.finalTrace;
        txGAS           = other.txGAS;
        txCount         = other.txCount;
        txTime          = other.txTime;
        info            = other.info;
        fullStack       = other.fullStack;
        accBatchGas     = other.accBatchGas;
        logs            = other.logs;
        call_trace      = other.call_trace;
        execution_trace = other.execution_trace;
        lastError       = other.lastError;
        return *this;
    }
    
    // FullTracerInterface methods
    uint64_t get_cumulative_gas_used (void)
    {
        return finalTrace.cumulative_gas_used;
    }
    string & get_new_state_root (void)
    {
        return finalTrace.new_state_root;
    }
    string & get_new_acc_input_hash (void)
    {
        return finalTrace.new_acc_input_hash;
    }
    string & get_new_local_exit_root (void)
    {
        return finalTrace.new_local_exit_root;
    }
    unordered_map<string, InfoReadWrite> * get_read_write_addresses(void)
    {
        return NULL;
    }
    vector<Response> & get_responses(void)
    {
        return finalTrace.responses;
    }
    vector<Opcode> & get_info(void)
    {
        return info;
    }
};

} // namespace

#endif