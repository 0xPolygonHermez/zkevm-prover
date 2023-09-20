#ifndef FULL_TRACER_HPP_fork_5
#define FULL_TRACER_HPP_fork_5

#include <gmpxx.h>
#include <unordered_map>
#include "main_sm/fork_5/main/context.hpp"
#include "main_sm/fork_5/main/rom_command.hpp"
#include "main_sm/fork_5/main/rom_line.hpp"
#include "main_sm/fork_5/main_exec_c/context_c.hpp"
#include "utils/time_metric.hpp"
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "full_tracer_interface.hpp"
#include "zkresult.hpp"

namespace fork_5
{

class Context;

class ContextData
{
public:
    string type;
};

class FullTracer: public FullTracerInterface
{
public:
    Goldilocks &fr;
    uint64_t depth;
    uint64_t prevCTX;
    uint64_t initGas;
    unordered_map<string,unordered_map<string,string>> deltaStorage;
    FinalTrace finalTrace;
    unordered_map<uint64_t,TxGAS> txGAS;
    uint64_t txCount;
    uint64_t txTime; // in us
    vector<vector<mpz_class>> fullStack;// Stack of the transaction
    uint64_t accBatchGas;
    map<uint64_t,map<uint64_t,Log>> logs;
    vector<Opcode> call_trace;
    vector<Opcode> execution_trace;
    string lastError;
    uint64_t numberOfOpcodesInThisTx;
    uint64_t lastErrorOpcode;
    unordered_map<string, InfoReadWrite> read_write_addresses;
    ReturnFromCreate returnFromCreate;
    unordered_map<uint64_t, ContextData> callData;
    string previousMemory;
    bool hasGaspriceOpcode;
    bool hasBalanceOpcode;
#ifdef LOG_TIME_STATISTICS
    TimeMetricStorage tms;
    struct timeval t;
    TimeMetricStorage tmsop;
    struct timeval top;
#endif
public:
    zkresult onError         (Context &ctx, const RomCommand &cmd);
    zkresult onError         (ContextC &ctxc, const string &error);
    zkresult onStoreLog      (Context &ctx, const RomCommand &cmd);
    zkresult onStoreLog      (ContextC &ctxc);
    zkresult onProcessTx     (Context &ctx, const RomCommand &cmd);
    zkresult onProcessTx     (ContextC &ctxc);
    zkresult onUpdateStorage (Context &ctx, const RomCommand &cmd);
    zkresult onUpdateStorage (ContextC &ctxc);
    zkresult onFinishTx      (Context &ctx, const RomCommand &cmd);
    zkresult onFinishTx      (ContextC &ctxc);
    zkresult onStartBatch    (Context &ctx, const RomCommand &cmd);
    zkresult onStartBatch    (ContextC &ctxc);
    zkresult onFinishBatch   (Context &ctx, const RomCommand &cmd);
    zkresult onFinishBatch   (ContextC &ctxc);

    zkresult onOpcode (Context &ctx, const RomCommand &cmd);
    zkresult addReadWriteAddress ( const Goldilocks::Element &address0, const Goldilocks::Element &address1, const Goldilocks::Element &address2, const Goldilocks::Element &faddress3, const Goldilocks::Element &address4, const Goldilocks::Element &address5, const Goldilocks::Element &address6, const Goldilocks::Element &address7,
                                   const Goldilocks::Element &keyType0, const Goldilocks::Element &keyType1, const Goldilocks::Element &keyType2, const Goldilocks::Element &keyType3, const Goldilocks::Element &keyType4, const Goldilocks::Element &keyType5, const Goldilocks::Element &keyType6, const Goldilocks::Element &keyType7,
                                   const mpz_class &value );

    FullTracer(Goldilocks &fr) : fr(fr), depth(1), prevCTX(0), initGas(0), txCount(0), txTime(0), accBatchGas(0), numberOfOpcodesInThisTx(0), lastErrorOpcode(0), hasGaspriceOpcode(false), hasBalanceOpcode(false) { };
    ~FullTracer()
    {
#ifdef LOG_TIME_STATISTICS
        tms.print("FullTracer");
        tmsop.print("FullTracer onOpcode");
#endif    
    }
    
    zkresult handleEvent (Context &ctx, const RomCommand &cmd);

    FullTracer & operator =(const FullTracer & other)
    {
        depth           = other.depth;
        initGas         = other.initGas;
        deltaStorage    = other.deltaStorage;
        finalTrace      = other.finalTrace;
        txGAS           = other.txGAS;
        txCount         = other.txCount;
        txTime          = other.txTime;
        //info            = other.info;
        fullStack       = other.fullStack;
        accBatchGas     = other.accBatchGas;
        logs            = other.logs;
        call_trace      = other.call_trace;
        execution_trace = other.execution_trace;
        lastError       = other.lastError;
        callData        = other.callData;
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
        return &read_write_addresses;
    }
    vector<Response> & get_responses(void)
    {
        return finalTrace.responses;
    }
    vector<Opcode> & get_info(void)
    {
        return execution_trace;
    }
    uint64_t get_tx_number(void)
    {
        return finalTrace.responses.size();
    }
};

void getTransactionHash( string    &to,
                         mpz_class value,
                         uint64_t  nonce,
                         uint64_t  gasLimit,
                         mpz_class gasPrice,
                         string    &data,
                         mpz_class &r,
                         mpz_class &s,
                         uint64_t  v,
                         string    &txHash,
                         string    &rlpTx );
                         
} // namespace

#endif