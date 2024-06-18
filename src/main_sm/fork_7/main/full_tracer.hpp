#ifndef FULL_TRACER_HPP_fork_7
#define FULL_TRACER_HPP_fork_7

#include <gmpxx.h>
#include <unordered_map>
#include "main_sm/fork_7/main/context.hpp"
#include "main_sm/fork_7/main/rom_command.hpp"
#include "main_sm/fork_7/main/rom_line.hpp"
#include "utils/time_metric.hpp"
#include "goldilocks_base_field.hpp"
#include "config.hpp"
#include "full_tracer_interface.hpp"
#include "zkresult.hpp"

namespace fork_7
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
    FinalTraceV2 finalTrace;
    unordered_map<uint64_t,TxGAS> txGAS;
    uint64_t txTime; // in us
    vector<vector<mpz_class>> fullStack;// Stack of the transaction
    uint64_t accBatchGas;
    map<uint64_t,map<uint64_t,LogV2>> logs;
    vector<Opcode> full_trace;
    string lastError;
    uint64_t numberOfOpcodesInThisTx;
    uint64_t lastErrorOpcode;
    unordered_map<string, InfoReadWrite> read_write_addresses;
    ReturnFromCreate returnFromCreate;
    unordered_map<uint64_t, ContextData> callData;
    string previousMemory;
    bool hasGaspriceOpcode;
    bool hasBalanceOpcode;
    uint64_t txIndex; // Transaction index in the current block
    Block currentBlock;
    bool isForced;
#ifdef LOG_TIME_STATISTICS
    TimeMetricStorage tms;
    struct timeval t;
    TimeMetricStorage tmsop;
    struct timeval top;
#endif
public:
    zkresult onError         (Context &ctx, const RomCommand &cmd);
    zkresult onStoreLog      (Context &ctx, const RomCommand &cmd);
    zkresult onStartBlock    (Context &ctx);
    zkresult onFinishBlock   (Context &ctx);
    zkresult onProcessTx     (Context &ctx, const RomCommand &cmd);
    zkresult onUpdateStorage (Context &ctx, const RomCommand &cmd);
    zkresult onFinishTx      (Context &ctx, const RomCommand &cmd);
    zkresult onStartBatch    (Context &ctx, const RomCommand &cmd);
    zkresult onFinishBatch   (Context &ctx, const RomCommand &cmd);

    zkresult onOpcode (Context &ctx, const RomCommand &cmd);
    zkresult addReadWriteAddress ( const Goldilocks::Element &address0, const Goldilocks::Element &address1, const Goldilocks::Element &address2, const Goldilocks::Element &faddress3, const Goldilocks::Element &address4, const Goldilocks::Element &address5, const Goldilocks::Element &address6, const Goldilocks::Element &address7,
                                   const Goldilocks::Element &keyType0, const Goldilocks::Element &keyType1, const Goldilocks::Element &keyType2, const Goldilocks::Element &keyType3, const Goldilocks::Element &keyType4, const Goldilocks::Element &keyType5, const Goldilocks::Element &keyType6, const Goldilocks::Element &keyType7,
                                   const mpz_class &value,
                                   const Goldilocks::Element (&key)[4] );
    zkresult fillInReadWriteAddresses (Context &ctx);

    FullTracer(Goldilocks &fr) : fr(fr), depth(1), prevCTX(0), initGas(0), txTime(0), accBatchGas(0), numberOfOpcodesInThisTx(0), lastErrorOpcode(0), hasGaspriceOpcode(false), hasBalanceOpcode(false), txIndex(0), isForced(false) { };
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
        txIndex         = other.txIndex;
        txTime          = other.txTime;
        //info            = other.info;
        fullStack       = other.fullStack;
        accBatchGas     = other.accBatchGas;
        logs            = other.logs;
        full_trace      = other.full_trace;
        lastError       = other.lastError;
        callData        = other.callData;
        currentBlock    = other.currentBlock;
        isForced        = other.isForced;
        return *this;
    }

    // FullTracerInterface methods
    uint64_t get_cumulative_gas_used (void)
    {
        return finalTrace.cumulative_gas_used;
    }
    uint64_t get_gas_used (void)
    {
        return finalTrace.gas_used;
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
    vector<Response> emptyResponses;
    vector<Response> & get_responses(void)
    {
        zklog.error("FullTracer::get_responses() called in fork 7");
        exitProcess();
        return emptyResponses;
    }
    vector<Block> & get_block_responses(void)
    {
        return finalTrace.block_responses;
    }
    vector<Opcode> & get_info(void)
    {
        return full_trace;
    }
    uint64_t get_block_number(void)
    {
        return finalTrace.block_responses.size();
    }
    uint64_t get_tx_number(void)
    {
        return currentBlock.responses.size();
    }
    string & get_error(void)
    {
        return finalTrace.error;
    }
    bool get_invalid_batch(void)
    {
        return finalTrace.invalid_batch;
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