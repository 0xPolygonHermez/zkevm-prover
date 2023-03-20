#include <iostream>
#include <sys/time.h>
#include <set>
#include "main_sm/fork_2/main/full_tracer.hpp"
#include "main_sm/fork_2/main/context.hpp"
#include "main_sm/fork_2/main/opcode_name.hpp"
#include "main_sm/fork_2/main/eval_command.hpp"
#include "goldilocks_base_field.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "rlp.hpp"
#include "utils.hpp"
#include "timer.hpp"

using namespace std;

namespace fork_2
{

set<string> opIncContext = {
    "CALL",
    "STATICCALL",
    "DELEGATECALL",
    "CALLCODE",
    "CREATE",
    "CREATE2"};

set<string> opDecContext = {
    "SELFDESTRUCT",
    "STOP",
    "INVALID",
    "REVERT",
    "RETURN"};
    
set<string> responseErrors = {
    "OOCS",
    "OOCK",
    "OOCB",
    "OOCM",
    "OOCA",
    "OOCPA",
    "OOCPO",
    "intrinsic_invalid_signature",
    "intrinsic_invalid_chain_id",
    "intrinsic_invalid_nonce",
    "intrinsic_invalid_gas_limit",
    "intrinsic_invalid_gas_overflow",
    "intrinsic_invalid_balance",
    "intrinsic_invalid_batch_gas_limit",
    "intrinsic_invalid_sender_code"};
    
set<string> oocErrors = {
    "OOCS",
    "OOCK",
    "OOCB",
    "OOCM",
    "OOCA",
    "OOCPA",
    "OOCPO"};

//////////
// UTILS
//////////

// Get range from memory
inline void getFromMemory(Context &ctx, mpz_class &offset, mpz_class &length, string &result)
{
    uint64_t offsetCtx = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;
    uint64_t addrMem = offsetCtx + 0x20000;

    result = "";
    double init = addrMem + double(offset.get_ui()) / 32;
    double end = addrMem + double(offset.get_ui() + length.get_ui()) / 32;
    uint64_t initCeil = ceil(init);
    uint64_t initFloor = floor(init);
    uint64_t endFloor = floor(end);

    if (init != double(initCeil))
    {
        mpz_class memScalarStart = 0;
        std::unordered_map<uint64_t, Fea>::iterator it = ctx.mem.find(initFloor);
        if (it != ctx.mem.end())
        {
            fea2scalar(ctx.fr, memScalarStart, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7);
        }
        string hexStringStart = PrependZeros(memScalarStart.get_str(16), 64);
        uint64_t bytesToSkip = (init - double(initFloor)) * 32;
        result += hexStringStart.substr(bytesToSkip * 2, 64);
    }

    for (uint64_t i = initCeil; i < endFloor; i++)
    {
        mpz_class memScalar = 0;
        if (ctx.mem.find(i) != ctx.mem.end())
        {
            Fea memValue = ctx.mem[i];
            fea2scalar(ctx.fr, memScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7);
        }
        result += PrependZeros(memScalar.get_str(16), 64);
    }

    if (end != double(endFloor))
    {
        mpz_class memScalarEnd = 0;
        std::unordered_map<uint64_t, Fea>::iterator it = ctx.mem.find(endFloor);
        if (it != ctx.mem.end())
        {
            fea2scalar(ctx.fr, memScalarEnd, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7);
        }
        string hexStringEnd = PrependZeros(memScalarEnd.get_str(16), 64);
        uint64_t bytesToRetrieve = (end - double(endFloor)) * 32;
        result += hexStringEnd.substr(0, bytesToRetrieve * 2);
    }
}

// Get a global or context variable
inline void getVarFromCtx(Context &ctx, bool global, uint64_t varOffset, mpz_class &result, uint64_t * pContext = NULL)
{
    // global and pContext!=NULL should never happen at the same time
    zkassert((global && (pContext==NULL)) || !global);

    uint64_t offsetCtx = global ? 0 : (pContext != NULL) ? *pContext*0x40000 : ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    uint64_t addressMem = offsetCtx + varOffset;
    unordered_map<uint64_t, Fea>::iterator memIterator;
    memIterator = ctx.mem.find(addressMem);
    if (memIterator == ctx.mem.end())
    {
        //cout << "FullTracer::getVarFromCtx() could not find in ctx.mem address with offset=" << varOffset << endl;
        result = 0;
    }
    else
    {
        Fea value = memIterator->second;
        fea2scalar(ctx.fr, result, value.fe0, value.fe1, value.fe2, value.fe3, value.fe4, value.fe5, value.fe6, value.fe7);
    }
}

// Get the stored calldata in the stack
inline void getCalldataFromStack(Context &ctx, uint64_t offset, uint64_t length, string &result)
{
    uint64_t contextAddress = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    uint64_t firstAddr = contextAddress + 0x10000 + 1024 + offset;
    uint64_t lastAddr = contextAddress + 0x20000;
    mpz_class auxScalar;
    result = "0x";
    
    unordered_map<uint64_t, Fea>::iterator memIterator;
    uint64_t consumedLength = 0;
    for (uint64_t i = firstAddr; i < lastAddr; i++)
    {
        memIterator = ctx.mem.find(i);
        if (memIterator == ctx.mem.end())
        {
            break;
        }
        Fea memVal = memIterator->second;
        fea2scalar(ctx.fr, auxScalar, memVal.fe0, memVal.fe1, memVal.fe2, memVal.fe3, memVal.fe4, memVal.fe5, memVal.fe6, memVal.fe7);
        result += PrependZeros(auxScalar.get_str(16), 64);
        if (length > 0)
        {
            consumedLength += 32;
            if (consumedLength >= length)
            {
                break;
            }
        }
    }

    if (length > 0)
    {
        result = result.substr(0, 2 + length * 2);
    }
    /*if (result.size() <= 2)
    {
        result = "0x0";
    }*/
}

// Get the value of a reg (A, B, C, D, E...)
inline void getRegFromCtx(Context &ctx, tReg reg, mpz_class &result)
{
    RomCommand cmd;
    cmd.reg = reg;
    CommandResult cr;
    eval_getReg(ctx, cmd, cr);
    cr2scalar(ctx, cr, result);
}

inline uint64_t getCurrentTime (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

using namespace rlp;

// Returns a transaction hash from transaction params
inline void getTransactionHash( string    &to,
                                mpz_class value,
                                uint64_t  nonce,
                                uint64_t  gasLimit,
                                mpz_class gasPrice,
                                string    &data,
                                mpz_class &r,
                                mpz_class &s,
                                uint64_t  v,
                                string    &txHash,
                                string    &rlpTx )
{
#ifdef LOG_TX_HASH
    cout << "FullTracer::getTransactionHash() to=" << to << " value=" << value << " nonce=" << nonce << " gasLimit=" << gasLimit << " gasPrice=" << gasPrice << " data=" << data << " r=0x" << r.get_str(16) << " s=0x" << s.get_str(16) << " v=" << v << endl;
#endif

    string raw;

    encode(raw, nonce);
    encode(raw, gasPrice);
    encode(raw, gasLimit);
    if (!encodeHexData(raw, to))
    {
        cout << "ERROR encoding to" << endl;
    }
    encode(raw, value);

    if (!encodeHexData(raw, data))
    {
        cout << "ERROR encoding data" << endl;
    }

    encode(raw, v);
    encode(raw, r);
    encode(raw, s);

    rlpTx.clear();
    encodeLen(rlpTx, raw.length(), true);
    rlpTx += raw;

    txHash = keccak256((const uint8_t *)(rlpTx.c_str()), rlpTx.length());

#ifdef LOG_TX_HASH
    cout << "FullTracer::getTransactionHash() keccak output txHash=" << txHash << " rlpTx=" << ba2string(rlpTx) << endl;
#endif
}

/***************/
/* Full tracer */
/***************/

zkresult FullTracer::handleEvent(Context &ctx, const RomCommand &cmd)
{
    // Full tracer should only be used during a process batch request
    if (ctx.proverRequest.type != prt_processBatch)
    {
        return ZKR_SUCCESS;
    }
    
    if (cmd.function == f_storeLog)
    {
        // if (ctx.proverRequest.bNoCounters) return;
        onStoreLog(ctx, cmd);
        return ZKR_SUCCESS;
    }
    if (cmd.params.size() == 0)
    {
        cerr << "Error: FullTracer::handleEvent() got an invalid event with cmd.params.size()==0 cmd.function=" << function2String(cmd.function) << endl;
        exitProcess();
    }
    if (cmd.params[0]->varName == "onError")
    {
        onError(ctx, cmd);
        return ZKR_SUCCESS;
    }
    if (cmd.params[0]->varName == "onProcessTx")
    {
        onProcessTx(ctx, cmd);
        return ZKR_SUCCESS;
    }
    if (cmd.params[0]->varName == "onFinishTx")
    {
        if ( (oocErrors.find(lastError)==oocErrors.end()) && (ctx.totalTransferredBalance != 0) )
        {
            cerr << "Error: FullTracer::handleEvent(onFinishTx) found ctx.totalTransferredBalance=" << ctx.totalTransferredBalance.get_str(10) << endl;
            return ZKR_SM_MAIN_BALANCE_MISMATCH;
        }
        onFinishTx(ctx, cmd);
        return ZKR_SUCCESS;
    }
    if (cmd.params[0]->varName == "onStartBatch")
    {
        onStartBatch(ctx, cmd);
        return ZKR_SUCCESS;
    }
    if (cmd.params[0]->varName == "onFinishBatch")
    {
        if ( (oocErrors.find(lastError)==oocErrors.end()) && (ctx.totalTransferredBalance != 0) )
        {
            cerr << "Error: FullTracer::handleEvent(onFinishBatch) found ctx.totalTransferredBalance=" << ctx.totalTransferredBalance.get_str(10) << endl;
            return ZKR_SM_MAIN_BALANCE_MISMATCH;
        }
        onFinishBatch(ctx, cmd);
        return ZKR_SUCCESS;
    }
    if (cmd.params[0]->function == f_onOpcode)
    {
        // if (ctx.proverRequest.bNoCounters) return;
        onOpcode(ctx, cmd);
        return ZKR_SUCCESS;
    }
    if (cmd.params[0]->function == f_onUpdateStorage)
    {
        // if (ctx.proverRequest.bNoCounters) return;
        onUpdateStorage(ctx, *cmd.params[0]);
        return ZKR_SUCCESS;
    }
    cerr << "Error: FullTracer::handleEvent() got an invalid event cmd.params[0]->varName=" << cmd.params[0]->varName << " cmd.function=" << function2String(cmd.function) << endl;
    exitProcess();
    return ZKR_INTERNAL_ERROR;
}

void FullTracer::onError(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    // Check params size
    if (cmd.params.size() != 2)
    {
        cerr << "Error: FullTracer::onError() got an invalid cmd.params.size()=" << cmd.params.size() << endl;
        exitProcess();
    }

    // Store the error
    lastError = cmd.params[1]->varName;

    // Intrinsic error should be set at tx level (not opcode)
    if ( (responseErrors.find(lastError) != responseErrors.end()) ||
         (execution_trace.size() == 0) )
    {
        if (finalTrace.responses.size() > txCount)
        {
            finalTrace.responses[txCount].error = lastError;
        }
        else if (finalTrace.responses.size() == txCount)
        {
            Response response;
            response.error = lastError;
            finalTrace.responses.push_back(response);
        }
        else
        {
            cerr << "Error: FullTracer::onError() got error=" << lastError << " with txCount=" << txCount << " but finalTrace.responses.size()=" << finalTrace.responses.size() << endl;
            exitProcess();
        }
    }

    if (execution_trace.size() > 0)
    {
        execution_trace[execution_trace.size() - 1].error = lastError;
    }

    // Revert logs
    uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    mpz_class auxScalar;
    getVarFromCtx(ctx, true, ctx.rom.lastCtxUsedOffset, auxScalar);
    uint64_t lastContextUsed = auxScalar.get_ui();
    for (uint64_t i=CTX; i<=lastContextUsed; i++)
    {
        if (logs.find(i) != logs.end())
        {
            logs.erase(i);
        }
    }

#ifdef LOG_FULL_TRACER_ON_ERROR
    cout << "FullTracer::onError() error=" << lastError << " zkPC=" << *ctx.pZKPC << " rom=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << endl;
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onError", TimeDiff(t));
#endif
}

void FullTracer::onStoreLog (Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    // Get indexLog from the provided register value
    mpz_class indexLogScalar;
    getRegFromCtx(ctx, cmd.params[0]->reg, indexLogScalar);
    uint64_t indexLog = indexLogScalar.get_ui();

    // Get isTopic
    uint64_t isTopic = cmd.params[1]->num.get_ui();

    // Get data
    mpz_class data;
    getRegFromCtx(ctx, cmd.params[2]->reg, data);

    // Init logs[CTX][indexLog], if required
    uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    unordered_map<uint64_t, std::unordered_map<uint64_t, Log>>::iterator itCTX;
    itCTX = logs.find(CTX);
    if (itCTX == logs.end())
    {
        unordered_map<uint64_t, Log> aux;
        Log log;
        aux[indexLog] = log;
        logs[CTX] = aux;
        itCTX = logs.find(CTX);
        zkassert(itCTX != logs.end());
    }
    
    std::unordered_map<uint64_t, Log>::iterator it;
    it = itCTX->second.find(indexLog);
    if (it == itCTX->second.end())
    {
        Log log;
        logs[CTX][indexLog] = log;
        it = itCTX->second.find(indexLog);
        zkassert(it != itCTX->second.end());
    }

    // Store data in the proper vector
    string dataString = PrependZeros(data.get_str(16), 64);
    if (isTopic)
    {
        it->second.topics.push_back(dataString);
    }
    else
    {
        it->second.data.push_back(dataString);
    }

    // Add log info
    mpz_class auxScalar;
    getVarFromCtx(ctx, false, ctx.rom.storageAddrOffset, auxScalar);
    it->second.address = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
    getVarFromCtx(ctx, true, ctx.rom.newNumBatchOffset, auxScalar);
    it->second.batch_number = auxScalar.get_ui();
    it->second.tx_hash = finalTrace.responses[txCount].tx_hash;
    it->second.tx_index = txCount;
    it->second.index = indexLog;

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onStoreLog() CTX=" << to_string(CTX) << " indexLog=" << indexLog << " isTopic=" << to_string(isTopic) << " data=" << dataString << endl;
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onStoreLog", TimeDiff(t));
#endif
}

// Triggered at the very beginning of transaction process
void FullTracer::onProcessTx(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    mpz_class auxScalar;
    Response response;

    /* Fill context object */

    // TX to and type
    getVarFromCtx(ctx, false, ctx.rom.isCreateContractOffset, auxScalar);    
    if (auxScalar.get_ui())
    {
        response.call_trace.context.type = "CREATE";
        response.call_trace.context.to = "0x";
    }
    else
    {
        response.call_trace.context.type = "CALL";
        getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, auxScalar);
        response.call_trace.context.to = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
    }

    // TX data
    getVarFromCtx(ctx, false, ctx.rom.txCalldataLenOffset, auxScalar);
    getCalldataFromStack(ctx, 0, auxScalar.get_ui(), response.call_trace.context.data);

    // TX gas
    getVarFromCtx(ctx, false, ctx.rom.txGasLimitOffset, auxScalar);
    response.call_trace.context.gas = auxScalar.get_ui();

    // TX value
    getVarFromCtx(ctx, false, ctx.rom.txValueOffset, auxScalar);
    response.call_trace.context.value = auxScalar;

    // TX output
    response.call_trace.context.output = "";

    // TX used gas
    response.call_trace.context.gas_used = 0;

    // TX execution time
    response.call_trace.context.execution_time = 0;

    // TX old state root
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]);
    response.call_trace.context.old_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // TX gas price
    getVarFromCtx(ctx, false, ctx.rom.txGasPriceRLPOffset, auxScalar);
    response.call_trace.context.gas_price = auxScalar;

    /* Fill response object */
    
    mpz_class r;
    getVarFromCtx(ctx, false, ctx.rom.txROffset, r);

    mpz_class s;
    getVarFromCtx(ctx, false, ctx.rom.txSOffset, s);

    // chain ID
    getVarFromCtx(ctx, false, ctx.rom.txChainIdOffset, auxScalar);
    uint64_t chainId = auxScalar.get_ui();

    // v
    getVarFromCtx(ctx, false, ctx.rom.txVOffset, auxScalar);
    uint64_t v;
    if (chainId == 0)
    {
        v = auxScalar.get_ui();
    }
    else
    {
        v = auxScalar.get_ui() - 27 + chainId * 2 + 35;
    }

    mpz_class nonceScalar;
    getVarFromCtx(ctx, false, ctx.rom.txNonceOffset, nonceScalar);
    uint64_t nonce = nonceScalar.get_ui();

    // TX hash
    getTransactionHash( response.call_trace.context.to,
                        response.call_trace.context.value,
                        nonce,
                        response.call_trace.context.gas,
                        response.call_trace.context.gas_price,
                        response.call_trace.context.data,
                        r,
                        s,
                        v,
                        response.tx_hash,
                        response.rlp_tx);
    response.type = 0;
    response.return_value.clear();
    response.gas_left = response.call_trace.context.gas;
    response.gas_used = 0;
    response.gas_refunded = 0;
    response.error = "";
    response.create_address = "";
    response.state_root = response.call_trace.context.old_state_root;
    response.logs.clear();
    response.call_trace.steps.clear();
    response.execution_trace.clear();

    // Create current tx object
    finalTrace.responses.push_back(response);
    txTime = getCurrentTime();

    // Reset values
    depth = 0;
    deltaStorage.clear();
    unordered_map<string, string> auxMap;
    deltaStorage[depth] = auxMap;
    txGAS[depth] = {response.call_trace.context.gas, 0};
    lastError = "";

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onProcessTx() finalTrace.responses.size()=" << finalTrace.responses.size() << endl;
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onProcessTx", TimeDiff(t));
#endif
}

// Triggered when storage is updated in opcode processing
void FullTracer::onUpdateStorage(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    if ( ctx.proverRequest.input.traceConfig.bGenerateStorage &&
         ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace )
    {
        zkassert(cmd.params.size() == 2);

        mpz_class regScalar;

        // The storage key is stored in C
        getRegFromCtx(ctx, cmd.params[0]->reg, regScalar);
        string key = PrependZeros(regScalar.get_str(16), 64);

        // The storage value is stored in D
        getRegFromCtx(ctx, cmd.params[1]->reg, regScalar);
        string value = PrependZeros(regScalar.get_str(16), 64);

        if (deltaStorage.find(depth) == deltaStorage.end())
        {
            cerr << "Error: FullTracer::onUpdateStorage() did not found deltaStorage of depth=" << depth << endl;
            exitProcess();
        }

        // Add key/value to deltaStorage
        deltaStorage[depth][key] = value;
        
        // Add deltaStorage to current execution_trace opcode info
        if (execution_trace.size() > 0)
        {
            execution_trace[execution_trace.size() - 1].storage = deltaStorage[depth];
        }

#ifdef LOG_FULL_TRACER
        cout << "FullTracer::onUpdateStorage() depth=" << depth << " key=" << key << " value=" << value << endl;
#endif
    }
#ifdef LOG_TIME_STATISTICS
    tms.add("onUpdateStorage", TimeDiff(t));
#endif
}

// Triggered after processing a transaction
void FullTracer::onFinishTx(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    Response &response = finalTrace.responses[txCount];

    // Set from address
    mpz_class fromScalar;
    getVarFromCtx(ctx, true, ctx.rom.txSrcOriginAddrOffset, fromScalar);
    response.call_trace.context.from = NormalizeTo0xNFormat(fromScalar.get_str(16), 40);

    // Set consumed tx gas
    uint64_t polsGas = fr.toU64(ctx.pols.GAS[*ctx.pStep]);
    if (polsGas > response.gas_left)
    {
        response.gas_used = response.gas_left;
    }
    else
    {
        response.gas_used = response.gas_left - polsGas;
    }
    response.call_trace.context.gas_used = response.gas_used;
    accBatchGas += response.gas_used;

    // Set return data, in case of deploy, get return buffer from stack if there is no error, otherwise get it from memory
    mpz_class offsetScalar;
    getVarFromCtx(ctx, false, ctx.rom.retDataOffsetOffset, offsetScalar);
    mpz_class lengthScalar;
    getVarFromCtx(ctx, false, ctx.rom.retDataLengthOffset, lengthScalar);
    if (response.call_trace.context.to == "0x")
    {
        // Check if there has been any error
        if ( bOpcodeCalled && (response.error.size()>0) )
        {
            getFromMemory(ctx, offsetScalar, lengthScalar, response.return_value);
        }
        else
        {
            getCalldataFromStack(ctx, offsetScalar.get_ui(), lengthScalar.get_ui(), response.return_value);
        }
    }
    else
    {
        getFromMemory(ctx, offsetScalar, lengthScalar, response.return_value);
    }

    // Set create address in case of deploy
    if (response.call_trace.context.to == "0x")
    {
        mpz_class addressScalar;
        getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, addressScalar);
        response.create_address = NormalizeToNFormat(addressScalar.get_str(16), 40);
    }

    // Set gas left
    response.gas_left -= response.gas_used;

    // Set new State Root
    mpz_class auxScalar;
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]);
    response.state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // If processed opcodes
    if ( ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace &&
         (execution_trace.size() > 0) )
    {
        Opcode &lastOpcodeExecution = execution_trace.at(execution_trace.size() - 1);

        // set refunded gas
        response.gas_refunded = lastOpcodeExecution.gas_refund;

        // Set gas price of last opcode
        lastOpcodeExecution.gas_cost = lastOpcodeExecution.gas - response.gas_left;

        response.execution_trace = execution_trace;

        if (response.error.size() == 0)
        {
            response.error = lastOpcodeExecution.error;
        }
    }

    // If processed opcodes
    if ( ctx.proverRequest.input.traceConfig.bGenerateCallTrace &&
         (call_trace.size() > 0) )
    {
        Opcode &lastOpcodeCall = call_trace.at(call_trace.size() - 1);

        // set refunded gas
        response.gas_refunded = lastOpcodeCall.gas_refund;

        // Set counters of last opcode to zero
        //Object.keys(lastOpcodeCall.counters).forEach((key) => {
        //            lastOpcodeCall.counters[key] = 0;
        //        });

        // Set gas price of last opcode
        lastOpcodeCall.gas_cost = lastOpcodeCall.gas - response.gas_left;

        response.call_trace.steps = call_trace;

        if (response.error.size() == 0)
        {
            response.error = lastOpcodeCall.error;
        }
    }
    else if ( ctx.proverRequest.input.bNoCounters &&
              (execution_trace.size() > 0) )
    {
        Opcode &lastOpcodeExecution = execution_trace.at(execution_trace.size() - 1);
        if (finalTrace.responses[finalTrace.responses.size() - 1].error == "")
        {
            finalTrace.responses[finalTrace.responses.size() - 1].error = lastOpcodeExecution.error;
        }
    }

    // Append to response logs
    unordered_map<uint64_t, std::unordered_map<uint64_t, Log>>::iterator logIt;
    unordered_map<uint64_t, Log>::const_iterator it;
    for (logIt=logs.begin(); logIt!=logs.end(); logIt++)
    {
        for (it = logIt->second.begin(); it != logIt->second.end(); it++)
        {
            Log log = it->second;
            finalTrace.responses[finalTrace.responses.size() - 1].logs.push_back(log);
        }
    }

    // Clear logs array
    logs.clear();

    // Increase transaction count
    txCount++;

    // Clean aux array for next iteration
    call_trace.clear();
    execution_trace.clear();
    logs.clear(); // TODO: Should we remove logs?

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onFinishTx() txCount=" << txCount << " finalTrace.responses.size()=" << finalTrace.responses.size() << " create_address=" << response.create_address << " state_root=" << response.state_root << endl;
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onFinishTx", TimeDiff(t));
#endif
}

void FullTracer::onStartBatch(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    if (finalTrace.bInitialized)
    {
#ifdef LOG_TIME_STATISTICS
        tms.add("onStartBatch", TimeDiff(t));
#endif
        return;
    }

    finalTrace.responses.clear();
    finalTrace.bInitialized = true;

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onStartBatch() old_state_root=" << finalTrace.old_state_root << endl;
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onStartBatch", TimeDiff(t));
#endif
}

void FullTracer::onFinishBatch(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    // Update used gas
    finalTrace.cumulative_gas_used = accBatchGas;

    // New state root
    mpz_class auxScalar;
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]);
    finalTrace.new_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // New acc input hash
    getVarFromCtx(ctx, true, ctx.rom.newAccInputHashOffset, auxScalar);
    finalTrace.new_acc_input_hash = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // TODO: Can we simply use finalTrace.new_acc_input_hash when constructing the response? Can we avoid these fields in the .proto?
    for (uint64_t r=0; r<finalTrace.responses.size(); r++)
    {
        finalTrace.responses[r].call_trace.context.batch = finalTrace.new_acc_input_hash;
        for (uint64_t l=0; l<finalTrace.responses[r].logs.size(); l++)
        {
            finalTrace.responses[r].logs[l].batch_hash = finalTrace.new_acc_input_hash;
        }
    }

    // New local exit root
    getVarFromCtx(ctx, true, ctx.rom.newLocalExitRootOffset, auxScalar);
    finalTrace.new_local_exit_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // New batch number
    // getVarFromCtx(ctx, true, "newNumBatch", auxScalar);
    // finalTrace.new_batch_num = auxScalar.get_ui();

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onFinishBatch() new_state_root=" << finalTrace.new_state_root << endl;
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onFinishBatch", TimeDiff(t));
#endif
}

void FullTracer::onOpcode(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    // Remember that at least one opcode was called
    bOpcodeCalled = true;

    Opcode singleInfo;

    if (ctx.proverRequest.input.bNoCounters)
    {
        execution_trace.push_back(singleInfo);
#ifdef LOG_TIME_STATISTICS
        tms.add("onOpcode", TimeDiff(t));
#endif
        return;
    }

    mpz_class auxScalar;

#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif
    // Get opcode info
    zkassert(cmd.params.size() == 1);
    zkassert(cmd.params[0]->params.size() == 1);
    uint8_t codeId;
    switch(cmd.params[0]->params[0]->op)
    {
        case op_number:
            codeId = cmd.params[0]->params[0]->num.get_ui();
            break;
        case op_getReg:
            getRegFromCtx(ctx, cmd.params[0]->params[0]->reg, auxScalar);
            codeId = auxScalar.get_ui();
            break;
        default:
            cerr << "Error: FullTracer::onOpcode() got invalid cmd.params=" << cmd.toString() << endl;
            exitProcess();
            exit(-1);
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("getCodeID", TimeDiff(top));
#endif
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif
    // Get opcode name into singleInfo.opcode, and filter code ID
    singleInfo.opcode = opcodeName[codeId].pName;
    codeId = opcodeName[codeId].codeID;
    singleInfo.op = codeId;

    // Check depth changes and update depth
    getVarFromCtx(ctx, true, ctx.rom.depthOffset, auxScalar);
    uint64_t newDepth = auxScalar.get_ui();
    bool decreaseDepth = newDepth < depth;
    bool increaseDepth = newDepth > depth;
    if (decreaseDepth || increaseDepth)
    {
        depth = newDepth;
    }

    // get previous opcode processed
    uint64_t numOpcodes = call_trace.size();
    Opcode * prevTraceCall = (numOpcodes > 0) ? &call_trace.at(numOpcodes - 1) : NULL;
    Opcode * prevTraceExecution = (numOpcodes > 0) ? &execution_trace.at(numOpcodes - 1) : NULL;

    // If is an ether transfer, don't add stop opcode to trace
    if ( (singleInfo.opcode == opcodeName[0x00/*STOP*/].pName) &&
        ( (prevTraceCall == NULL) || increaseDepth) )
    {
        getVarFromCtx(ctx, false, ctx.rom.bytecodeLengthOffset, auxScalar);
        if (auxScalar == 0)
        {
#ifdef LOG_TIME_STATISTICS
            tmsop.add("getCodeName", TimeDiff(top));
            tms.add("onOpcode", TimeDiff(t));
#endif
            return;
        }
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("getCodeName", TimeDiff(top));
#endif

#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif

    if (ctx.proverRequest.input.traceConfig.bGenerateMemory)
    {
        string finalMemory;
        
        // Get context offset
        uint64_t offsetCtx = fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;

        // Get memory address
        uint64_t addrMem = offsetCtx + 0x20000;

        uint64_t lengthMemOffset = ctx.rom.memLengthOffset;
        uint64_t lenMemValueFinal = 0;
        unordered_map< uint64_t, Fea >::iterator it;
        it = ctx.mem.find(offsetCtx + lengthMemOffset);
        if (it != ctx.mem.end())
        {
            Fea lenMemValue = it->second;
            fea2scalar(ctx.fr, auxScalar, lenMemValue.fe0, lenMemValue.fe1, lenMemValue.fe2, lenMemValue.fe3, lenMemValue.fe4, lenMemValue.fe5, lenMemValue.fe6, lenMemValue.fe7);
            lenMemValueFinal = ceil(double(auxScalar.get_ui()) / 32);
        }

        for (uint64_t i = 0; i < lenMemValueFinal; i++)
        {
            it = ctx.mem.find(addrMem + i);
            if (it == ctx.mem.end())
            {
                finalMemory += "0000000000000000000000000000000000000000000000000000000000000000";
                continue;
            }
            Fea memValue = it->second;
            fea2scalar(ctx.fr, auxScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7);
            finalMemory += PrependZeros(auxScalar.get_str(16), 64);
        }
        
        singleInfo.memory = finalMemory;
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("getMemory", TimeDiff(top));
#endif

#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif
    
    if (ctx.proverRequest.input.traceConfig.bGenerateStack)
    {
        vector<mpz_class> finalStack;

        // Get context offset
        uint64_t offsetCtx = fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;

        // Get stack address
        uint64_t addr = offsetCtx + 0x10000;

        uint16_t sp = fr.toU64(ctx.pols.SP[*ctx.pStep]);
        unordered_map<uint64_t, Fea>::iterator it;
        for (uint16_t i = 0; i < sp; i++)
        {
            it = ctx.mem.find(addr + i);
            if (it == ctx.mem.end())
                continue;
            Fea stack = it->second;
            mpz_class stackScalar;
            fea2scalar(ctx.fr, stackScalar, stack.fe0, stack.fe1, stack.fe2, stack.fe3, stack.fe4, stack.fe5, stack.fe6, stack.fe7);
            finalStack.push_back(stackScalar);
        }

        // save stack to opcode trace
        singleInfo.stack = finalStack;
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("getStack", TimeDiff(top));
#endif
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif
    if (ctx.proverRequest.input.traceConfig.bGenerateTrace)
    {
        singleInfo.depth = depth + 1;
        singleInfo.pc = fr.toU64(ctx.pols.PC[*ctx.pStep]);
        singleInfo.gas = fr.toU64(ctx.pols.GAS[*ctx.pStep]);
        gettimeofday(&singleInfo.startTime, NULL);
        getVarFromCtx(ctx, false, ctx.rom.gasRefundOffset, auxScalar);
        singleInfo.gas_refund = auxScalar.get_ui();
        //singleInfo.error = "";
        fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]);
        singleInfo.state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

        // Set gas forwarded to a new context and save gas left in previous context
        if (increaseDepth)
        {
            // get gas forwarded to current ctx
            uint64_t gasForwarded = fr.toU64(ctx.pols.GAS[*ctx.pStep]);

            // get gas remaining in origin context
            getVarFromCtx(ctx, false, ctx.rom.originCTXOffset, auxScalar);
            uint64_t originCTX = auxScalar.get_ui();
            getVarFromCtx(ctx, false, ctx.rom.gasCTXOffset, auxScalar, &originCTX);
            uint64_t gasRemaining = auxScalar.get_ui();
            txGAS[depth] = {gasForwarded, gasRemaining};
        }

        // Add contract info
        getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, auxScalar);
        singleInfo.contract.address = NormalizeToNFormat(auxScalar.get_str(16), 40);

        getVarFromCtx(ctx, false, ctx.rom.txSrcAddrOffset, auxScalar);
        singleInfo.contract.caller = NormalizeToNFormat(auxScalar.get_str(16), 40);

        getVarFromCtx(ctx, false, ctx.rom.txValueOffset, auxScalar);
        singleInfo.contract.value = auxScalar;

        getCalldataFromStack(ctx, 0, 0, singleInfo.contract.data);
        
        singleInfo.contract.gas = txGAS[depth].forwarded;
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("getSingleInfo", TimeDiff(top));
#endif
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif
    // cout << "singleInfo.remaining_gas=" << singleInfo.remaining_gas << endl;
    // compute: gas spent & zk-counters in previous opcode
    if (numOpcodes > 0)
    {
        // The gas cost of the opcode is gas before - gas after processing the opcode
        int64_t gasCost = prevTraceCall->gas - fr.toS64(ctx.pols.GAS[*ctx.pStep]);
        prevTraceCall->gas_cost = gasCost;
        prevTraceExecution->gas_cost = gasCost;
        // cout << "info[info.size() - 1].gas_cost=" << info[info.size() - 1].gas_cost << endl;

        // going to previous depth
        if (decreaseDepth) {
            // get gas cost consumed by current ctx except last opcode: gasForwarded - gasSecondLast
            uint64_t gasConsumedExceptLastOpcode = txGAS[depth + 1].forwarded - prevTraceCall->gas;
            // get gas remaining at the end of the previous context
            uint64_t gasEndPreviousCtx = singleInfo.gas - txGAS[depth + 1].remaining;
            // get gas spend by previous ctx
            uint64_t gasSpendPreviousCtx = txGAS[depth + 1].forwarded - gasEndPreviousCtx;
            // compute gas spend by the last opcode
            uint64_t gasLastOpcode = gasSpendPreviousCtx - gasConsumedExceptLastOpcode;
            // set opcode gas cost to traces
            prevTraceCall->gas_cost = gasLastOpcode;
            prevTraceExecution->gas_cost = gasLastOpcode;
        }

        prevTraceExecution->duration = TimeDiff(prevTraceExecution->startTime, singleInfo.startTime);

        // Round up to next multiple of 32
        getVarFromCtx(ctx, false, ctx.rom.memLengthOffset, auxScalar);
        singleInfo.memory_size = (auxScalar.get_ui() / 32) * 32;
    }

    if (ctx.proverRequest.input.traceConfig.bGenerateStorage && increaseDepth)
    {
        unordered_map<string, string> auxMap;
        deltaStorage[depth + 1] = auxMap;
    }

    // Return data
    if (ctx.proverRequest.input.traceConfig.bGenerateReturnData)
    {
        singleInfo.return_data.clear();
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("numOpcodesPositive", TimeDiff(top));
#endif
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif

    if (ctx.proverRequest.input.traceConfig.bGenerateCallTrace)
    {
        // Save output traces
        call_trace.push_back(singleInfo);
    }

    if (ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace)
    {
        // Save output traces
        execution_trace.push_back(singleInfo);
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("copySingleInfoIntoTraces", TimeDiff(top));
#endif
#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onOpcode() codeId=" << to_string(codeId) << " opcode=" << singleInfo.opcode << endl;
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onOpcode", TimeDiff(t));
#endif
}

#define SMT_KEY_BALANCE 0
#define SMT_KEY_NONCE 1

/*
   Add an address when it is either read/write in the state-tree
   address - address accessed
   keyType - Parameter accessed in the state-tree
   value - value read/write
 */

void FullTracer::addReadWriteAddress ( const Goldilocks::Element &address0, const Goldilocks::Element &address1, const Goldilocks::Element &address2, const Goldilocks::Element &address3, const Goldilocks::Element &address4, const Goldilocks::Element &address5, const Goldilocks::Element &address6, const Goldilocks::Element &address7,
                                       const Goldilocks::Element &keyType0, const Goldilocks::Element &keyType1, const Goldilocks::Element &keyType2, const Goldilocks::Element &keyType3, const Goldilocks::Element &keyType4, const Goldilocks::Element &keyType5, const Goldilocks::Element &keyType6, const Goldilocks::Element &keyType7,
                                       const mpz_class &value )
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    // Get address
    mpz_class address;
    fea2scalar(fr, address, address0, address1, address2, address3, address4, address5, address6, address7);
    string addressHex = NormalizeTo0xNFormat(address.get_str(16), 40);

    // Get key type
    mpz_class keyType;
    fea2scalar(fr, keyType, keyType0, keyType1, keyType2, keyType3, keyType4, keyType5, keyType6, keyType7);

    unordered_map<string, InfoReadWrite>::iterator it;
    if (keyType == SMT_KEY_BALANCE)
    {
        it = read_write_addresses.find(addressHex);
        if (it == read_write_addresses.end())
        {
            InfoReadWrite infoReadWrite;
            infoReadWrite.balance = value.get_str();
            read_write_addresses[addressHex] = infoReadWrite;
        }
        else
        {
            it->second.balance = value.get_str();
        }
    }
    else if (keyType == SMT_KEY_NONCE)
    {
        it = read_write_addresses.find(addressHex);
        if (it == read_write_addresses.end())
        {
            InfoReadWrite infoReadWrite;
            infoReadWrite.nonce = value.get_str();
            read_write_addresses[addressHex] = infoReadWrite;
        }
        else
        {
            it->second.nonce = value.get_str();
        }
    }

#ifdef LOG_TIME_STATISTICS
    tms.add("addReadWriteAddress", TimeDiff(t));
#endif
}

} // namespace