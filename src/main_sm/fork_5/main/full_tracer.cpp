#include <iostream>
#include <sys/time.h>
#include <set>
#include "main_sm/fork_5/main/full_tracer.hpp"
#include "main_sm/fork_5/main/context.hpp"
#include "main_sm/fork_5/main/opcode_name.hpp"
#include "main_sm/fork_5/main/eval_command.hpp"
#include "goldilocks_base_field.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "rlp.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"

using namespace std;

namespace fork_5
{

set<string> opIncContext = {
    "CALL",
    "STATICCALL",
    "DELEGATECALL",
    "CALLCODE",
    "CREATE",
    "CREATE2" };

set<string> opDecContext = {
    "SELFDESTRUCT",
    "STOP",
    "INVALID",
    "REVERT",
    "RETURN" };

set<string> opCall = {
    "CALL",
    "STATICCALL",
    "DELEGATECALL",
    "CALLCODE" };

set<string> opCreate = {
    "CREATE",
    "CREATE2" };

set<string> zeroCostOp = {
    "STOP",
    "REVERT",
    "RETURN" };
    
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
    "intrinsic_invalid_sender_code",
    "invalidRLP" };
    
set<string> oocErrors = {
    "OOCS",
    "OOCK",
    "OOCB",
    "OOCM",
    "OOCA",
    "OOCPA",
    "OOCPO" };

//////////
// UTILS
//////////

// Get range from memory
inline zkresult getFromMemory(Context &ctx, mpz_class &offset, mpz_class &length, string &result, uint64_t * pContext = NULL)
{
    uint64_t offsetCtx = (pContext != NULL) ? *pContext*0x40000 : ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;
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
            if (!fea2scalar(ctx.fr, memScalarStart, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7))
            {
                zklog.error("getFromMemory() failed calling fea2scalar() 1");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
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
            if (!fea2scalar(ctx.fr, memScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7))
            {
                zklog.error("getFromMemory() failed calling fea2scalar() 2");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
        }
        result += PrependZeros(memScalar.get_str(16), 64);
    }

    if (end != double(endFloor))
    {
        mpz_class memScalarEnd = 0;
        std::unordered_map<uint64_t, Fea>::iterator it = ctx.mem.find(endFloor);
        if (it != ctx.mem.end())
        {
            if (!fea2scalar(ctx.fr, memScalarEnd, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7))
            {
                zklog.error("getFromMemory() failed calling fea2scalar() 2");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
        }
        string hexStringEnd = PrependZeros(memScalarEnd.get_str(16), 64);
        uint64_t bytesToRetrieve = (end - double(endFloor)) * 32;
        result += hexStringEnd.substr(0, bytesToRetrieve * 2);
    }

    // Limit result memory length in case it is a chunk contained in one single slot
    result = result.substr(0, length.get_ui()*2);

    return ZKR_SUCCESS;
}

// Get a global or context variable
inline zkresult getVarFromCtx(Context &ctx, bool global, uint64_t varOffset, mpz_class &result, uint64_t * pContext = NULL)
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
        if (!fea2scalar(ctx.fr, result, value.fe0, value.fe1, value.fe2, value.fe3, value.fe4, value.fe5, value.fe6, value.fe7))
        {
            zklog.error("getVarFromCtx() failed calling fea2scalar()");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
    }
    return ZKR_SUCCESS;
}

#if 0
// Get the stored calldata in the stack
inline zkresult getCalldataFromStack(Context &ctx, uint64_t offset, uint64_t length, string &result)
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
        if (!fea2scalar(ctx.fr, auxScalar, memVal.fe0, memVal.fe1, memVal.fe2, memVal.fe3, memVal.fe4, memVal.fe5, memVal.fe6, memVal.fe7))
        {
            zklog.error("getCalldataFromStack() failed calling fea2scalar()");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
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

    return ZKR_SUCCESS;
}
#endif

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
                                string    &rlpTx )
{
#ifdef LOG_TX_HASH
    zklog.info("FullTracer::getTransactionHash() to=" + to + " value=" + value.get_str(16) + " nonce=" + to_string(nonce) + " gasLimit=" + to_string(gasLimit) + " gasPrice=" + gasPrice.get_str(10) + " data=" + data + " r=0x" + r.get_str(16) + " s=0x" + s.get_str(16) + " v=" + to_string(v));
#endif

    string raw;

    encode(raw, nonce);
    encode(raw, gasPrice);
    encode(raw, gasLimit);
    if (!encodeHexData(raw, to))
    {
        zklog.error("FullTracer::getTransactionHash() ERROR encoding to");
    }
    encode(raw, value);

    if (!encodeHexData(raw, data))
    {
        zklog.error("FullTracer::getTransactionHash() ERROR encoding data");
    }

    encode(raw, v);
    encode(raw, r);
    encode(raw, s);

    rlpTx.clear();
    encodeLen(rlpTx, raw.length(), true);
    rlpTx += raw;

    txHash = keccak256((const uint8_t *)(rlpTx.c_str()), rlpTx.length());

#ifdef LOG_TX_HASH
    zklog.info("FullTracer::getTransactionHash() keccak output txHash=" + txHash + " rlpTx=" + ba2string(rlpTx));
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
        return onStoreLog(ctx, cmd);
    }
    if (cmd.params.size() == 0)
    {
        zklog.error("FullTracer::handleEvent() got an invalid event with cmd.params.size()==0 cmd.function=" + function2String(cmd.function));
        exitProcess();
    }
    if (cmd.params[0]->varName == "onError")
    {
        return onError(ctx, cmd);
    }
    if (cmd.params[0]->varName == "onProcessTx")
    {
        return onProcessTx(ctx, cmd);
    }
    if (cmd.params[0]->varName == "onFinishTx")
    {
        if ( (oocErrors.find(lastError)==oocErrors.end()) && (ctx.totalTransferredBalance != 0) )
        {
            zklog.error("FullTracer::handleEvent(onFinishTx) found ctx.totalTransferredBalance=" + ctx.totalTransferredBalance.get_str(10));
            return ZKR_SM_MAIN_BALANCE_MISMATCH;
        }
        return onFinishTx(ctx, cmd);
    }
    if (cmd.params[0]->varName == "onStartBatch")
    {
        return onStartBatch(ctx, cmd);
    }
    if (cmd.params[0]->varName == "onFinishBatch")
    {
        if ( (oocErrors.find(lastError)==oocErrors.end()) && (ctx.totalTransferredBalance != 0) )
        {
            zklog.error("FullTracer::handleEvent(onFinishBatch) found ctx.totalTransferredBalance=" + ctx.totalTransferredBalance.get_str(10));
            return ZKR_SM_MAIN_BALANCE_MISMATCH;
        }
        return onFinishBatch(ctx, cmd);
    }
    if (cmd.params[0]->function == f_onOpcode)
    {
        // if (ctx.proverRequest.bNoCounters) return;
        return onOpcode(ctx, cmd);
    }
    if (cmd.params[0]->function == f_onUpdateStorage)
    {
        // if (ctx.proverRequest.bNoCounters) return;
        return onUpdateStorage(ctx, *cmd.params[0]);
    }
    zklog.error("FullTracer::handleEvent() got an invalid event cmd.params[0]->varName=" + cmd.params[0]->varName + " cmd.function=" + function2String(cmd.function));
    exitProcess();
    return ZKR_INTERNAL_ERROR;
}

zkresult FullTracer::onError(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    // Check params size
    if (cmd.params.size() != 2)
    {
        zklog.error("FullTracer::onError() got an invalid cmd.params.size()=" + to_string(cmd.params.size()));
        exitProcess();
    }

    zkresult zkr;

    // Store the error
    lastError = cmd.params[1]->varName;
    lastErrorOpcode = numberOfOpcodesInThisTx;

    // Intrinsic error should be set at tx level (not opcode)
    if ( (responseErrors.find(lastError) != responseErrors.end()) ||
         ((execution_trace.size() == 0) && call_trace.size() == 0) )
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
            zklog.error("FullTracer::onError() got error=" + lastError + " with txCount=" + to_string(txCount) + " but finalTrace.responses.size()=" + to_string(finalTrace.responses.size()));
            exitProcess();
        }
    }

    if (execution_trace.size() > 0)
    {
        execution_trace[execution_trace.size() - 1].error = lastError;
    }

    if (call_trace.size() > 0)
    {
        call_trace[call_trace.size() - 1].error = lastError;
    }

    // Revert logs
    uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    mpz_class auxScalar;
    zkr = getVarFromCtx(ctx, true, ctx.rom.lastCtxUsedOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onError() failed calling getVarFromCtx(ctx.rom.lastCtxUsedOffset)");
        return zkr;
    }
    uint64_t lastContextUsed = auxScalar.get_ui();
    for (uint64_t i=CTX; i<=lastContextUsed; i++)
    {
        if (logs.find(i) != logs.end())
        {
            logs.erase(i);
        }
    }

#ifdef LOG_FULL_TRACER_ON_ERROR
    zklog.info("FullTracer::onError() error=" + lastError + " zkPC=" + to_string(*ctx.pZKPC) + " rom=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr));
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onError", TimeDiff(t));
#endif
    return ZKR_SUCCESS;
}

zkresult FullTracer::onError(ContextC &ctxc, const string &error)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    //zkresult zkr;

    // Store the error
    lastError = error;
    lastErrorOpcode = numberOfOpcodesInThisTx;

    // Intrinsic error should be set at tx level (not opcode)
    if ( (responseErrors.find(lastError) != responseErrors.end()) ||
         ((execution_trace.size() == 0) && call_trace.size() == 0) )
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
            zklog.error("FullTracer::onError() got error=" + lastError + " with txCount=" + to_string(txCount) + " but finalTrace.responses.size()=" + to_string(finalTrace.responses.size()));
            exitProcess();
        }
    }

    if (execution_trace.size() > 0)
    {
        execution_trace[execution_trace.size() - 1].error = lastError;
    }

    if (call_trace.size() > 0)
    {
        call_trace[call_trace.size() - 1].error = lastError;
    }

    // Revert logs
    /*uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    mpz_class auxScalar;
    zkr = getVarFromCtx(ctx, true, ctx.rom.lastCtxUsedOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onError() failed calling getVarFromCtx(ctx.rom.lastCtxUsedOffset)");
        return zkr;
    }
    uint64_t lastContextUsed = auxScalar.get_ui();
    for (uint64_t i=CTX; i<=lastContextUsed; i++)
    {
        if (logs.find(i) != logs.end())
        {
            logs.erase(i);
        }
    }*/

#ifdef LOG_FULL_TRACER_ON_ERROR
    zklog.info("FullTracer::onError() error=" + lastError + " tx=" + to_string(ctxc.tx));
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onError", TimeDiff(t));
#endif
    return ZKR_SUCCESS;
}

zkresult FullTracer::onStoreLog (Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    zkresult zkr;
    
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
    map<uint64_t, map<uint64_t, Log>>::iterator itCTX;
    itCTX = logs.find(CTX);
    if (itCTX == logs.end())
    {
        map<uint64_t, Log> aux;
        Log log;
        aux[indexLog] = log;
        logs[CTX] = aux;
        itCTX = logs.find(CTX);
        zkassert(itCTX != logs.end());
    }
    
    map<uint64_t, Log>::iterator it;
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
    zkr = getVarFromCtx(ctx, false, ctx.rom.storageAddrOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onStoreLog() failed calling getVarFromCtx(ctx.rom.storageAddrOffset)");
        return zkr;
    }
    it->second.address = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
    zkr = getVarFromCtx(ctx, true, ctx.rom.newNumBatchOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onStoreLog() failed calling getVarFromCtx(ctx.rom.newNumBatchOffset)");
        return zkr;
    }
    it->second.batch_number = auxScalar.get_ui();
    it->second.tx_hash = finalTrace.responses[txCount].tx_hash;
    it->second.tx_index = txCount;
    it->second.index = indexLog;

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onStoreLog() CTX=" + to_string(CTX) + " indexLog=" + to_string(indexLog) + " isTopic=" + to_string(isTopic) + " data=" + dataString);
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onStoreLog", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult FullTracer::onStoreLog (ContextC &ctxc)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
/*
    zkresult zkr;
    
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
    map<uint64_t, map<uint64_t, Log>>::iterator itCTX;
    itCTX = logs.find(CTX);
    if (itCTX == logs.end())
    {
        map<uint64_t, Log> aux;
        Log log;
        aux[indexLog] = log;
        logs[CTX] = aux;
        itCTX = logs.find(CTX);
        zkassert(itCTX != logs.end());
    }
    
    map<uint64_t, Log>::iterator it;
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
    zkr = getVarFromCtx(ctx, false, ctx.rom.storageAddrOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onStoreLog() failed calling getVarFromCtx(ctx.rom.storageAddrOffset)");
        return zkr;
    }
    it->second.address = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
    zkr = getVarFromCtx(ctx, true, ctx.rom.newNumBatchOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onStoreLog() failed calling getVarFromCtx(ctx.rom.newNumBatchOffset)");
        return zkr;
    }
    it->second.batch_number = auxScalar.get_ui();
    it->second.tx_hash = finalTrace.responses[txCount].tx_hash;
    it->second.tx_index = txCount;
    it->second.index = indexLog;
*/
#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onStoreLog() CTX=" + to_string(CTX) + " indexLog=" + to_string(indexLog) + " isTopic=" + to_string(isTopic) + " data=" + dataString);
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onStoreLog", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

// Triggered at the very beginning of transaction process
zkresult FullTracer::onProcessTx(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    mpz_class auxScalar;
    Response response;
    zkresult zkr;

    /* Fill context object */

    // TX to and type
    zkr = getVarFromCtx(ctx, false, ctx.rom.isCreateContractOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.isCreateContractOffset)");
        return zkr;
    }
    if (auxScalar.get_ui())
    {
        response.call_trace.context.type = "CREATE";
        response.call_trace.context.to = "0x";
    }
    else
    {
        response.call_trace.context.type = "CALL";
        zkr = getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txDestAddrOffset)");
            return zkr;
        }
        response.call_trace.context.to = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
    }

    // TX data
    zkr = getVarFromCtx(ctx, false, ctx.rom.calldataCTXOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.calldataCTXOffset)");
        return zkr;
    }
    uint64_t calldataCTX = auxScalar.get_ui();
    mpz_class calldataOffset;
    zkr = getVarFromCtx(ctx, false, ctx.rom.calldataOffsetOffset, calldataOffset);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.calldataOffsetOffset)");
        return zkr;
    }
    zkr = getVarFromCtx(ctx, false, ctx.rom.txCalldataLenOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txCalldataLenOffset)");
        return zkr;
    }
    zkr = getFromMemory(ctx, calldataOffset, auxScalar, response.call_trace.context.data, &calldataCTX);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getFromMemory()");
        return zkr;
    }

    // TX gas
    zkr = getVarFromCtx(ctx, false, ctx.rom.txGasLimitOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txGasLimitOffset)");
        return zkr;
    }
    response.call_trace.context.gas = auxScalar.get_ui();

    // TX value
    zkr = getVarFromCtx(ctx, false, ctx.rom.txValueOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txValueOffset)");
        return zkr;
    }
    response.call_trace.context.value = auxScalar;

    // TX output
    response.call_trace.context.output = "";

    // TX used gas
    response.call_trace.context.gas_used = 0;

    // TX execution time
    response.call_trace.context.execution_time = 0;

    // TX old state root
    if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
    {
        zklog.error("FullTracer::onProcessTx() failed calling fea2scalar()");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    response.call_trace.context.old_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // TX gas price
    zkr = getVarFromCtx(ctx, false, ctx.rom.txGasPriceRLPOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txGasPriceRLPOffset)");
        return zkr;
    }
    response.call_trace.context.gas_price = auxScalar;

    // Call data
    uint64_t CTX = fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    ContextData contextData;
    contextData.type = "CALL";
    callData[CTX] = contextData;

    // prevCTX
    prevCTX = CTX;

    /* Fill response object */
    
    mpz_class r;
    zkr = getVarFromCtx(ctx, false, ctx.rom.txROffset, r);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txROffset)");
        return zkr;
    }

    mpz_class s;
    zkr = getVarFromCtx(ctx, false, ctx.rom.txSOffset, s);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txSOffset)");
        return zkr;
    }

    // chain ID
    zkr = getVarFromCtx(ctx, false, ctx.rom.txChainIdOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txChainIdOffset)");
        return zkr;
    }
    uint64_t chainId = auxScalar.get_ui();

    // v
    zkr = getVarFromCtx(ctx, false, ctx.rom.txVOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txVOffset)");
        return zkr;
    }
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
    zkr = getVarFromCtx(ctx, false, ctx.rom.txNonceOffset, nonceScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txNonceOffset)");
        return zkr;
    }
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

    // Get effective percentage
    zkr = getVarFromCtx(ctx, false, ctx.rom.effectivePercentageRLPOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.effectivePercentageRLPOffset)");
        return zkr;
    }
    response.effective_percentage = auxScalar.get_ui();

    // Create current tx object
    finalTrace.responses.push_back(response);

    // Clear temporary tx traces
    execution_trace.clear();
    execution_trace.reserve(ctx.config.fullTracerTraceReserveSize);
    call_trace.clear();
    call_trace.reserve(ctx.config.fullTracerTraceReserveSize);

    // Reset previous memory
    previousMemory = "";
    
    txTime = getCurrentTime();

    // Reset values
    depth = 1;
    deltaStorage.clear();
    txGAS[depth] = {response.call_trace.context.gas, 0};
    lastError = "";

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onProcessTx() finalTrace.responses.size()=" + to_string(finalTrace.responses.size()));
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onProcessTx", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

// Triggered at the very beginning of transaction process
zkresult FullTracer::onProcessTx (ContextC &ctxc)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    mpz_class auxScalar;
    Response response;
    //zkresult zkr;
/*
    // Fill context object

    // TX to and type
    zkr = getVarFromCtx(ctx, false, ctx.rom.isCreateContractOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.isCreateContractOffset)");
        return zkr;
    }
    if (auxScalar.get_ui())
    {
        response.call_trace.context.type = "CREATE";
        response.call_trace.context.to = "0x";
    }
    else
    {
        response.call_trace.context.type = "CALL";
        zkr = getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txDestAddrOffset)");
            return zkr;
        }
        response.call_trace.context.to = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
    }

    // TX data
    zkr = getVarFromCtx(ctx, false, ctx.rom.txCalldataLenOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txCalldataLenOffset)");
        return zkr;
    }
    zkr = getCalldataFromStack(ctx, 0, auxScalar.get_ui(), response.call_trace.context.data);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getCalldataFromStack()");
        return zkr;
    }

    // TX gas
    zkr = getVarFromCtx(ctx, false, ctx.rom.txGasLimitOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txGasLimitOffset)");
        return zkr;
    }
    response.call_trace.context.gas = auxScalar.get_ui();

    // TX value
    zkr = getVarFromCtx(ctx, false, ctx.rom.txValueOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txValueOffset)");
        return zkr;
    }
    response.call_trace.context.value = auxScalar;
*/
    // TX output
    response.call_trace.context.output = "";

    // TX used gas
    response.call_trace.context.gas_used = 0;

    // TX execution time
    response.call_trace.context.execution_time = 0;

    // TX old state root
    fea2scalar(ctxc.fr, auxScalar, ctxc.root);
    response.call_trace.context.old_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // TX gas price
    /*zkr = getVarFromCtx(ctx, false, ctx.rom.txGasPriceRLPOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txGasPriceRLPOffset)");
        return zkr;
    }
    response.call_trace.context.gas_price = auxScalar;

    // Call data
    uint64_t CTX = fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    ContextData contextData;
    contextData.type = "CALL";
    callData[CTX] = contextData;

    // prevCTX
    prevCTX = CTX;*/

    // Fill response object
    response.tx_hash = ctxc.batch.tx[ctxc.tx].txHash();
    response.rlp_tx = ctxc.batch.tx[ctxc.tx].txHashRlp;
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
    response.effective_percentage = ctxc.batch.tx[ctxc.tx].gasPercentage;

    // Create current tx object
    finalTrace.responses.push_back(response);

    // Clear temporary tx traces
    execution_trace.clear();
    execution_trace.reserve(ctxc.config.fullTracerTraceReserveSize);
    call_trace.clear();
    call_trace.reserve(ctxc.config.fullTracerTraceReserveSize);

    // Reset previous memory
    previousMemory = "";
    
    txTime = getCurrentTime();

    // Reset values
    depth = 1;
    deltaStorage.clear();
    txGAS[depth] = {response.call_trace.context.gas, 0};
    lastError = "";

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onProcessTx() finalTrace.responses.size()=" + to_string(finalTrace.responses.size()));
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onProcessTx", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

// Triggered when storage is updated in opcode processing
zkresult FullTracer::onUpdateStorage(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    if ( ctx.proverRequest.input.traceConfig.bGenerateStorage &&
         ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace )
    {
        zkassert(cmd.params.size() == 2);

        mpz_class auxScalar;

        // The storage key is stored in C
        getRegFromCtx(ctx, cmd.params[0]->reg, auxScalar);
        string key = auxScalar.get_str(16);

        // The storage value is stored in D
        getRegFromCtx(ctx, cmd.params[1]->reg, auxScalar);
        string value = auxScalar.get_str(16);

        // Delta storage is computed for the affected contract address
        zkresult zkr = getVarFromCtx(ctx, false, ctx.rom.storageAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onUpdateStorage() failed calling getVarFromCtx(storageAddr) result=" + zkresult2string(zkr));
            return zkr;
        }
        string storageAddress = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

        // add key/value to deltaStorage, if undefined, create object
        if (deltaStorage.find(storageAddress) == deltaStorage.end())
        {
            unordered_map<string, string> auxMap;
            deltaStorage[storageAddress] = auxMap;
        }

        // Add key/value to deltaStorage
        deltaStorage[storageAddress][key] = value;
        
        // Add deltaStorage to current execution_trace opcode info
        if (execution_trace.size() > 0)
        {
            execution_trace[execution_trace.size() - 1].storage = deltaStorage[storageAddress];
        }

#ifdef LOG_FULL_TRACER
        zklog.info("FullTracer::onUpdateStorage() depth=" + to_string(depth) + " key=" + key + " value=" + value);
#endif
    }
#ifdef LOG_TIME_STATISTICS
    tms.add("onUpdateStorage", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

// Triggered when storage is updated in opcode processing
zkresult FullTracer::onUpdateStorage (ContextC &ctxc)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
/*
    if ( ctx.proverRequest.input.traceConfig.bGenerateStorage &&
         ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace )
    {
        zkassert(cmd.params.size() == 2);

        mpz_class auxScalar;

        // The storage key is stored in C
        getRegFromCtx(ctx, cmd.params[0]->reg, auxScalar);
        string key = auxScalar.get_str(16);

        // The storage value is stored in D
        getRegFromCtx(ctx, cmd.params[1]->reg, auxScalar);
        string value = auxScalar.get_str(16);

        // Delta storage is computed for the affected contract address
        zkresult zkr = getVarFromCtx(ctx, false, ctx.rom.storageAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onUpdateStorage() failed calling getVarFromCtx(storageAddr) result=" + zkresult2string(zkr));
            return zkr;
        }
        string storageAddress = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

        // add key/value to deltaStorage, if undefined, create object
        if (deltaStorage.find(storageAddress) == deltaStorage.end())
        {
            unordered_map<string, string> auxMap;
            deltaStorage[storageAddress] = auxMap;
        }

        // Add key/value to deltaStorage
        deltaStorage[storageAddress][key] = value;
        
        // Add deltaStorage to current execution_trace opcode info
        if (execution_trace.size() > 0)
        {
            execution_trace[execution_trace.size() - 1].storage = deltaStorage[storageAddress];
        }

#ifdef LOG_FULL_TRACER
        zklog.info("FullTracer::onUpdateStorage() depth=" + to_string(depth) + " key=" + key + " value=" + value);
#endif
    }
    */
#ifdef LOG_TIME_STATISTICS
    tms.add("onUpdateStorage", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

// Triggered after processing a transaction
zkresult FullTracer::onFinishTx(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    zkresult zkr;
    Response &response = finalTrace.responses[txCount];

    // Set from address
    mpz_class fromScalar;
    zkr = getVarFromCtx(ctx, true, ctx.rom.txSrcOriginAddrOffset, fromScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txSrcOriginAddrOffset)");
        return zkr;
    }
    response.call_trace.context.from = NormalizeTo0xNFormat(fromScalar.get_str(16), 40);

    // Set effective gas price
    mpz_class auxScalar;
    zkr = getVarFromCtx(ctx, true, ctx.rom.txGasPriceOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txGasPriceOffset)");
        return zkr;
    }
    response.effective_gas_price = Add0xIfMissing(auxScalar.get_str(16));

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

    // Set return data always; get it from memory
    {
        mpz_class offsetScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.retDataOffsetOffset, offsetScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.retDataOffsetOffset)");
            return zkr;
        }
        mpz_class lengthScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.retDataLengthOffset, lengthScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.retDataLengthOffset)");
            return zkr;
        }
        zkr = getFromMemory(ctx, offsetScalar, lengthScalar, response.return_value);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getFromMemory() 1");
            return zkr;
        }
        if ( ctx.proverRequest.input.traceConfig.bGenerateCallTrace )
        {
            response.call_trace.context.output = response.return_value;
        }
    }

    // Set create address in case of deploy
    if (response.call_trace.context.to == "0x")
    {
        mpz_class addressScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, addressScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txDestAddrOffset)");
            return zkr;
        }
        response.create_address = NormalizeToNFormat(addressScalar.get_str(16), 40);
    }

    // Set gas left
    response.gas_left -= response.gas_used;

    // Set new State Root
    if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
    {
        zklog.error("FullTracer::onFinishTx() failed calling fea2scalar(SR)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    response.state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // If processed opcodes
    if ( ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace &&
         (execution_trace.size() > 0) )
    {
        Opcode &lastOpcodeExecution = execution_trace.at(execution_trace.size() - 1);

        // set refunded gas
        response.gas_refunded = lastOpcodeExecution.gas_refund;

        // Set gas price of last opcode if no error and is not a deploy and is not STOP (RETURN + REVERT)
        if ( (execution_trace.size() > 1) &&
             (lastOpcodeExecution.op != 0x00 /*STOP opcode*/ ) &&
             (lastOpcodeExecution.error.size() == 0) &&
             (response.call_trace.context.to != "0x") )
        {
            lastOpcodeExecution.gas_cost = lastOpcodeExecution.gas - fr.toU64(ctx.pols.GAS[*ctx.pStep]);
        }

        response.execution_trace.swap(execution_trace);

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

        //  Set gas price of last opcode if no error and is not a deploy and is not STOP (RETURN + REVERT)
        if ( (execution_trace.size() > 1) &&
             (lastOpcodeCall.op != 0x00 /*STOP opcode*/ ) &&
             (lastOpcodeCall.error.size() == 0) &&
             (response.call_trace.context.to != "0x") )
        {
            lastOpcodeCall.gas_cost = lastOpcodeCall.gas - fr.toU64(ctx.pols.GAS[*ctx.pStep]);
        }

        response.call_trace.steps.swap(call_trace);

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
        
    if ( !ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace && 
         !ctx.proverRequest.input.traceConfig.bGenerateCallTrace && 
         (numberOfOpcodesInThisTx != 0) &&
         (lastErrorOpcode != numberOfOpcodesInThisTx) )
    {
        finalTrace.responses[finalTrace.responses.size() - 1].error = "";
    }

    // set flags has_gasprice_opcode and has_balance_opcode
    finalTrace.responses[finalTrace.responses.size() - 1].has_gasprice_opcode = hasGaspriceOpcode;
    finalTrace.responses[finalTrace.responses.size() - 1].has_balance_opcode = hasBalanceOpcode;

    // Order all logs (from all CTX) in order of index
    map<uint64_t, Log> auxLogs;
    map<uint64_t, map<uint64_t, Log>>::iterator logIt;
    map<uint64_t, Log>::const_iterator it;
    for (logIt=logs.begin(); logIt!=logs.end(); logIt++)
    {
        for (it = logIt->second.begin(); it != logIt->second.end(); it++)
        {
            auxLogs[it->second.index] = it->second;
        }
    }

    // Append to response logs, overwriting log indexes to be sequential
    uint64_t logIndex = 0;
    map<uint64_t, Log>::iterator auxLogsIt;
    for (auxLogsIt = auxLogs.begin(); auxLogsIt != auxLogs.end(); auxLogsIt++)
    {
        auxLogsIt->second.index = logIndex;
        logIndex++;
        finalTrace.responses[finalTrace.responses.size() - 1].logs.push_back(auxLogsIt->second);
    }

    // Increase transaction count
    txCount++;

    // Clean aux array for next iteration
    call_trace.clear();
    execution_trace.clear();
    logs.clear();
    callData.clear();

    // Reset opcodes counters
    numberOfOpcodesInThisTx = 0;
    lastErrorOpcode = 0;

    // Call semiFlush
    ctx.pHashDB->semiFlush(ctx.proverRequest.uuid, response.state_root, ctx.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onFinishTx() txCount=" + to_string(txCount) + " finalTrace.responses.size()=" + to_string(finalTrace.responses.size()) + " create_address=" + response.create_address + " state_root=" + response.state_root);
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onFinishTx", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

// Triggered after processing a transaction
zkresult FullTracer::onFinishTx (ContextC &ctxc)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    //zkresult zkr;
    Response &response = finalTrace.responses[txCount];

    // Set from address
    response.call_trace.context.from = NormalizeTo0xNFormat(ctxc.batch.tx[ctxc.tx].fromPublicKey.get_str(16), 40);

    // Set effective gas price
    mpz_class auxScalar;
    response.effective_gas_price = "0x" + ctxc.batch.tx[ctxc.tx].effectiveGasPrice.get_str(16);
    response.gas_used = ctxc.batch.tx[ctxc.tx].gas.get_ui();
    response.call_trace.context.gas_used = response.gas_used;
    accBatchGas += response.gas_used;
    /*

    // Set return data always; get it from memory
    {
        mpz_class offsetScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.retDataOffsetOffset, offsetScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.retDataOffsetOffset)");
            return zkr;
        }
        mpz_class lengthScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.retDataLengthOffset, lengthScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.retDataLengthOffset)");
            return zkr;
        }
        zkr = getFromMemory(ctx, offsetScalar, lengthScalar, response.return_value);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getFromMemory() 1");
            return zkr;
        }
        if ( ctx.proverRequest.input.traceConfig.bGenerateCallTrace )
        {
            response.call_trace.context.output = response.return_value;
        }
    }

    // Set create address in case of deploy
    if (response.call_trace.context.to == "0x")
    {
        mpz_class addressScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, addressScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txDestAddrOffset)");
            return zkr;
        }
        response.create_address = NormalizeToNFormat(addressScalar.get_str(16), 40);
    }

    // Set gas left
    response.gas_left -= response.gas_used;
*/
    // Set new State Root
    fea2scalar(ctxc.fr, auxScalar, ctxc.root);
    response.state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // If processed opcodes
    if ( ctxc.proverRequest.input.traceConfig.bGenerateExecuteTrace &&
         (execution_trace.size() > 0) )
    {
        Opcode &lastOpcodeExecution = execution_trace.at(execution_trace.size() - 1);

        // set refunded gas
        response.gas_refunded = lastOpcodeExecution.gas_refund;

        // Set gas price of last opcode if no error and is not a deploy and is not STOP (RETURN + REVERT)
        if ( (execution_trace.size() > 1) &&
             (lastOpcodeExecution.op != 0x00 /*STOP opcode*/ ) &&
             (lastOpcodeExecution.error.size() == 0) &&
             (response.call_trace.context.to != "0x") )
        {
            // TODO lastOpcodeExecution.gas_cost = lastOpcodeExecution.gas - fr.toU64(ctx.pols.GAS[*ctx.pStep]);
        }

        response.execution_trace.swap(execution_trace);

        if (response.error.size() == 0)
        {
            response.error = lastOpcodeExecution.error;
        }
    }

    // If processed opcodes
    if ( ctxc.proverRequest.input.traceConfig.bGenerateCallTrace &&
         (call_trace.size() > 0) )
    {
        Opcode &lastOpcodeCall = call_trace.at(call_trace.size() - 1);

        // set refunded gas
        response.gas_refunded = lastOpcodeCall.gas_refund;

        //  Set gas price of last opcode if no error and is not a deploy and is not STOP (RETURN + REVERT)
        if ( (execution_trace.size() > 1) &&
             (lastOpcodeCall.op != 0x00 /*STOP opcode*/ ) &&
             (lastOpcodeCall.error.size() == 0) &&
             (response.call_trace.context.to != "0x") )
        {
            // TODO lastOpcodeCall.gas_cost = lastOpcodeCall.gas - fr.toU64(ctx.pols.GAS[*ctx.pStep]);
        }

        response.call_trace.steps.swap(call_trace);

        if (response.error.size() == 0)
        {
            response.error = lastOpcodeCall.error;
        }
    }
    else if ( ctxc.proverRequest.input.bNoCounters &&
              (execution_trace.size() > 0) )
    {
        Opcode &lastOpcodeExecution = execution_trace.at(execution_trace.size() - 1);
        if (finalTrace.responses[finalTrace.responses.size() - 1].error == "")
        {
            finalTrace.responses[finalTrace.responses.size() - 1].error = lastOpcodeExecution.error;
        }
    }
        
    if ( !ctxc.proverRequest.input.traceConfig.bGenerateExecuteTrace && 
         !ctxc.proverRequest.input.traceConfig.bGenerateCallTrace && 
         (numberOfOpcodesInThisTx != 0) &&
         (lastErrorOpcode != numberOfOpcodesInThisTx) )
    {
        finalTrace.responses[finalTrace.responses.size() - 1].error = "";
    }

    // Order all logs (from all CTX) in order of index
    map<uint64_t, Log> auxLogs;
    map<uint64_t, map<uint64_t, Log>>::iterator logIt;
    map<uint64_t, Log>::const_iterator it;
    for (logIt=logs.begin(); logIt!=logs.end(); logIt++)
    {
        for (it = logIt->second.begin(); it != logIt->second.end(); it++)
        {
            auxLogs[it->second.index] = it->second;
        }
    }

    // Append to response logs, overwriting log indexes to be sequential
    uint64_t logIndex = 0;
    map<uint64_t, Log>::iterator auxLogsIt;
    for (auxLogsIt = auxLogs.begin(); auxLogsIt != auxLogs.end(); auxLogsIt++)
    {
        auxLogsIt->second.index = logIndex;
        logIndex++;
        finalTrace.responses[finalTrace.responses.size() - 1].logs.push_back(auxLogsIt->second);
    }

    // Increase transaction count
    txCount++;

    // Clean aux array for next iteration
    call_trace.clear();
    execution_trace.clear();
    logs.clear();
    callData.clear();

    // Reset opcodes counters
    numberOfOpcodesInThisTx = 0;
    lastErrorOpcode = 0;

    // Call semiFlush
    ctxc.pHashDB->semiFlush(ctxc.proverRequest.uuid, response.state_root, ctxc.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onFinishTx() txCount=" + to_string(txCount) + " finalTrace.responses.size()=" + to_string(finalTrace.responses.size()) + " create_address=" + response.create_address + " state_root=" + response.state_root);
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onFinishTx", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult FullTracer::onStartBatch(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    if (finalTrace.bInitialized)
    {
#ifdef LOG_TIME_STATISTICS
        tms.add("onStartBatch", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
    }

    finalTrace.responses.clear();
    finalTrace.bInitialized = true;

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onStartBatch()");
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onStartBatch", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult FullTracer::onStartBatch (ContextC &ctxc)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    if (finalTrace.bInitialized)
    {
#ifdef LOG_TIME_STATISTICS
        tms.add("onStartBatch", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
    }

    finalTrace.responses.clear();
    finalTrace.bInitialized = true;

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onStartBatch()");
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onStartBatch", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult FullTracer::onFinishBatch(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    zkresult zkr;

    // Update used gas
    finalTrace.cumulative_gas_used = accBatchGas;

    // New state root
    mpz_class auxScalar;
    if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
    {
        zklog.error("FullTracer::onFinishBatch() failed calling fea2scalar(SR)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    finalTrace.new_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // New acc input hash
    zkr = getVarFromCtx(ctx, true, ctx.rom.newAccInputHashOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBatch() failed calling getVarFromCtx(ctx.rom.newAccInputHashOffset)");
        return zkr;
    }
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
    zkr = getVarFromCtx(ctx, true, ctx.rom.newLocalExitRootOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBatch() failed calling getVarFromCtx(ctx.rom.newLocalExitRootOffset)");
        return zkr;
    }
    finalTrace.new_local_exit_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // New batch number
    // getVarFromCtx(ctx, true, "newNumBatch", auxScalar);
    // finalTrace.new_batch_num = auxScalar.get_ui();

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onFinishBatch() new_state_root=" + finalTrace.new_state_root);
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onFinishBatch", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult FullTracer::onFinishBatch (ContextC &ctxc)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    //zkresult zkr;

    // Update used gas
    finalTrace.cumulative_gas_used = accBatchGas;
    

    // New state root
    mpz_class auxScalar;
    fea2scalar(ctxc.fr, auxScalar, ctxc.root);
    finalTrace.new_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);
/*
    // New acc input hash
    zkr = getVarFromCtx(ctx, true, ctx.rom.newAccInputHashOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBatch() failed calling getVarFromCtx(ctx.rom.newAccInputHashOffset)");
        return zkr;
    }
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
    zkr = getVarFromCtx(ctx, true, ctx.rom.newLocalExitRootOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBatch() failed calling getVarFromCtx(ctx.rom.newLocalExitRootOffset)");
        return zkr;
    }
    finalTrace.new_local_exit_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);
*/
    // New batch number
    // getVarFromCtx(ctx, true, "newNumBatch", auxScalar);
    // finalTrace.new_batch_num = auxScalar.get_ui();

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onFinishBatch() new_state_root=" + finalTrace.new_state_root);
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onFinishBatch", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult FullTracer::onOpcode(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    // Increase opcodes counter
    numberOfOpcodesInThisTx++;

    // Update depth if a variation in CTX is detected
    uint64_t CTX = fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    if (prevCTX > CTX)
    {
        depth -= 1;
    }
    else if (prevCTX < CTX)
    {
        depth += 1;
    }
    prevCTX = CTX;

    zkresult zkr;

    Opcode singleInfo;

    if (ctx.proverRequest.input.bNoCounters)
    {
        execution_trace.emplace_back(singleInfo);
#ifdef LOG_TIME_STATISTICS
        tms.add("onOpcode", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
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
            zklog.error("FullTracer::onOpcode() got invalid cmd.params=" + cmd.toString());
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
    singleInfo.opcode = opcodeInfo[codeId].pName;
    codeId = opcodeInfo[codeId].codeID;
    singleInfo.op = codeId;

    // set flag 'has_gasprice_opcode' if opcode is GASPRICE
    if (codeId == 0x3a /*GASPRICE*/)
    {
        hasGaspriceOpcode = true;
    }

    // set flag 'has_balance_opcode' if opcode is BALANCE
    if (codeId == 0x31 /*BALANCE*/)
    {
        hasBalanceOpcode = true;
    }

    // Check depth changes and update depth
    singleInfo.depth = depth;

    // get previous opcode processed
    uint64_t numOpcodes = call_trace.size();
    Opcode * prevTraceCall = (numOpcodes > 0) ? &call_trace.at(numOpcodes - 1) : NULL;
    Opcode * prevTraceExecution = (numOpcodes > 0) ? &execution_trace.at(numOpcodes - 1) : NULL;

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
            if (!fea2scalar(ctx.fr, auxScalar, lenMemValue.fe0, lenMemValue.fe1, lenMemValue.fe2, lenMemValue.fe3, lenMemValue.fe4, lenMemValue.fe5, lenMemValue.fe6, lenMemValue.fe7))
            {
                zklog.error("FullTracer::onOpcode() failed calling fea2scalar(lenMemValue)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
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
            if (!fea2scalar(ctx.fr, auxScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7))
            {
                zklog.error("FullTracer::onOpcode() failed calling fea2scalar(memValue)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
            finalMemory += PrependZeros(auxScalar.get_str(16), 64);
        }
        
        string baMemory = string2ba(finalMemory);

        if (numOpcodes == 0)
        {
            singleInfo.memory_offset = 0;
            singleInfo.memory = baMemory;
        }
        else if (baMemory != previousMemory)
        {
            uint64_t offset;
            uint64_t length;
            getStringIncrement(previousMemory, baMemory, offset, length);
            if (length > 0)
            {
                singleInfo.memory_offset = offset;
                singleInfo.memory = baMemory.substr(offset, length); // Content of memory, incremental
            }
            previousMemory = baMemory;
        }
        singleInfo.memory_size = baMemory.size();
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
            if (!fea2scalar(ctx.fr, stackScalar, stack.fe0, stack.fe1, stack.fe2, stack.fe3, stack.fe4, stack.fe5, stack.fe6, stack.fe7))
            {
                zklog.error("FullTracer::onOpcode() failed calling fea2scalar(stackScalar)");
                return ZKR_SM_MAIN_FEA2SCALAR;
            }
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
        singleInfo.pc = fr.toU64(ctx.pols.PC[*ctx.pStep]);
        singleInfo.gas = fr.toU64(ctx.pols.GAS[*ctx.pStep]);
        singleInfo.gas_cost = opcodeInfo[codeId].gas;
        gettimeofday(&singleInfo.startTime, NULL);

        // Set gas refund
        zkr = getVarFromCtx(ctx, false, ctx.rom.gasRefundOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.gasRefundOffset)");
            return zkr;
        }
        singleInfo.gas_refund = auxScalar.get_ui();

        //singleInfo.error = "";
        
        // Set state root
        if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
        {
            zklog.error("FullTracer::onOpcode() failed calling fea2scalar()");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }
        singleInfo.state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

        // Add contract info
        zkr = getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.txDestAddrOffset)");
            return zkr;
        }
        singleInfo.contract.address = NormalizeToNFormat(auxScalar.get_str(16), 40);

        zkr = getVarFromCtx(ctx, false, ctx.rom.txSrcAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.txSrcAddrOffset)");
            return zkr;
        }
        singleInfo.contract.caller = NormalizeToNFormat(auxScalar.get_str(16), 40);

        zkr = getVarFromCtx(ctx, false, ctx.rom.txValueOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.txValueOffset)");
            return zkr;
        }
        singleInfo.contract.value = auxScalar;

        if ((prevTraceCall != NULL) && ((opIncContext.find(prevTraceCall->opcode) != opIncContext.end()) || (zeroCostOp.find(prevTraceCall->opcode) != zeroCostOp.end())))
        {
            zkr = getVarFromCtx(ctx, false, ctx.rom.calldataCTXOffset, auxScalar);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.calldataCTXOffset)");
                return zkr;
            }
            uint64_t calldataCTX = auxScalar.get_ui();
            mpz_class calldataOffset;
            zkr = getVarFromCtx(ctx, false, ctx.rom.calldataOffsetOffset, calldataOffset);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.calldataOffsetOffset)");
                return zkr;
            }
            zkr = getVarFromCtx(ctx, false, ctx.rom.txCalldataLenOffset, auxScalar);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.txCalldataLenOffset)");
                return zkr;
            }
            zkr = getFromMemory(ctx, calldataOffset, auxScalar, singleInfo.contract.data, &calldataCTX);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getCalldataFromStack()");
                return zkr;
            }
        }
        
        singleInfo.contract.gas = txGAS[depth].remaining;

        singleInfo.contract.type = "CALL";
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

        if (zeroCostOp.find(prevTraceCall->opcode) != zeroCostOp.end())
        {
            prevTraceCall->gas_cost = 0;
            prevTraceExecution->gas_cost = 0;
        }
        else if (opCreate.find(prevTraceCall->opcode) != opCreate.end())
        {
            // In case of error at create, we can't get the gas cost from next opcodes, so we have to use rom variables
            if (prevTraceExecution->error.size() > 0)
            {
                zkr = getVarFromCtx(ctx, true, ctx.rom.gasCallOffset, auxScalar);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.gasCallOffset)");
                    return zkr;
                }
                uint64_t gasCall = auxScalar.get_ui();
                prevTraceCall->gas_cost = gasCost - gasCall + fr.toS64(ctx.pols.GAS[*ctx.pStep]);
            }
            else
            {
                // If is a create opcode, set gas cost as currentGas - gasCall

                // get gas CTX in origin context
                zkr = getVarFromCtx(ctx, false, ctx.rom.originCTXOffset, auxScalar);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.originCTXOffset)");
                    return zkr;
                }
                uint64_t originCTX = auxScalar.get_ui();
                zkr = getVarFromCtx(ctx, false, ctx.rom.gasCTXOffset, auxScalar, &originCTX);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.gasCTXOffset)");
                    return zkr;
                }
                uint64_t gasCTX = auxScalar.get_ui();

                // Set gas cost
                prevTraceCall->gas_cost = gasCost - gasCTX;
            }
            prevTraceExecution->gas_cost = prevTraceCall->gas_cost;
        }
        else if ( (opCall.find(prevTraceCall->opcode) != opCall.end()) &&
                  (prevTraceCall->depth != singleInfo.depth) )
        {
            // Only check if different depth because we are removing STOP from trace in case the call is empty (CALL-STOP)

            // get gas CTX in origin context
            zkr = getVarFromCtx(ctx, false, ctx.rom.originCTXOffset, auxScalar);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.originCTXOffset)");
                return zkr;
            }
            uint64_t originCTX = auxScalar.get_ui();
            zkr = getVarFromCtx(ctx, false, ctx.rom.gasCTXOffset, auxScalar, &originCTX);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.gasCTXOffset)");
                return zkr;
            }
            uint64_t gasCTX = auxScalar.get_ui();

            prevTraceCall->gas_cost = prevTraceCall->gas - gasCTX;
            prevTraceExecution->gas_cost = prevTraceCall->gas_cost;
        }
        else
        {
            prevTraceCall->gas_cost = gasCost;
            prevTraceExecution->gas_cost = gasCost;
        }

        // cout << "info[info.size() - 1].gas_cost=" << info[info.size() - 1].gas_cost << endl;

        // Set gas refund for sstore opcode
        zkr = getVarFromCtx(ctx, false, ctx.rom.gasRefundOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.gasRefundOffset)");
            return zkr;
        }
        uint64_t gasRefund = auxScalar.get_ui();
        if (gasRefund > 0)
        {
            singleInfo.gas_refund = gasRefund;
            if (prevTraceCall->op == 0x55 /*SSTORE*/)
            {
                prevTraceCall->gas_refund = gasRefund;
                prevTraceExecution->gas_refund = gasRefund;
            }
        }

        prevTraceExecution->duration = TimeDiff(prevTraceExecution->startTime, singleInfo.startTime);
    }

    // Return data
    if (ctx.proverRequest.input.traceConfig.bGenerateReturnData)
    {
        // Write return data from create/create2 until CTX changes
        if (returnFromCreate.enabled)
        {
            if (returnFromCreate.returnValue.size() == 0)
            {
                uint64_t retDataCTX = returnFromCreate.createCTX;
                mpz_class offsetScalar;
                zkr = getVarFromCtx(ctx, false, ctx.rom.retDataOffsetOffset, offsetScalar, &retDataCTX);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.retDataOffsetOffset)");
                    return zkr;
                }
                mpz_class lengthScalar;
                zkr = getVarFromCtx(ctx, false, ctx.rom.retDataLengthOffset, lengthScalar, &retDataCTX);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.retDataLengthOffset)");
                    return zkr;
                }
                string return_value;
                zkr = getFromMemory(ctx, offsetScalar, lengthScalar, return_value, &retDataCTX);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getFromMemory() 1");
                    return zkr;
                }
                returnFromCreate.returnValue.push_back(return_value);
            }

            mpz_class currentCTXScalar;
            zkr = getVarFromCtx(ctx, true, ctx.rom.currentCTXOffset, currentCTXScalar);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.currentCTXOffset)");
                return zkr;
            }
            uint64_t currentCTX = currentCTXScalar.get_ui();
            if (returnFromCreate.originCTX == currentCTX)
            {
                singleInfo.return_data = returnFromCreate.returnValue;
            }
            else
            {
                returnFromCreate.enabled = false;
            }
        }

        // Check if return is called from CREATE/CREATE2
        mpz_class isCreateScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.isCreateOffset, isCreateScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.isCreateOffset)");
            return zkr;
        }
        bool isCreate = isCreateScalar.get_ui();

        if (isCreate)
        {            
            if (singleInfo.opcode == opcodeInfo[0xf3/*RETURN*/].pName)
            {
                returnFromCreate.enabled = true;

                mpz_class originCTXScalar;
                zkr = getVarFromCtx(ctx, false, ctx.rom.originCTXOffset, originCTXScalar);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.originCTXOffset)");
                    return zkr;
                }
                returnFromCreate.originCTX = originCTXScalar.get_ui();

                returnFromCreate.createCTX = fr.toU64(ctx.pols.CTX[*ctx.pStep]);
            }
        }
        else
        {
            mpz_class retDataCTXScalar;
            zkr = getVarFromCtx(ctx, false, ctx.rom.retDataCTXOffset, retDataCTXScalar);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.retDataCTXOffset)");
                return zkr;
            }
            if (retDataCTXScalar != 0)
            {
                uint64_t retDataCTX = retDataCTXScalar.get_ui();
                mpz_class offsetScalar;
                zkr = getVarFromCtx(ctx, false, ctx.rom.retDataOffsetOffset, offsetScalar, &retDataCTX);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.retDataOffsetOffset)");
                    return zkr;
                }
                mpz_class lengthScalar;
                zkr = getVarFromCtx(ctx, false, ctx.rom.retDataLengthOffset, lengthScalar, &retDataCTX);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.retDataLengthOffset)");
                    return zkr;
                }
                string return_value;
                zkr = getFromMemory(ctx, offsetScalar, lengthScalar, return_value, &retDataCTX);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("FullTracer::onOpcode() failed calling getFromMemory() 1");
                    return zkr;
                }
                singleInfo.return_data.push_back(return_value);
            }
        }
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("numOpcodesPositive", TimeDiff(top));
#endif
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&top, NULL);
#endif

    // Check previous step
    Opcode * prevStep = NULL;
    if (execution_trace.size() > 0)
    {
        prevStep = &execution_trace[execution_trace.size() - 1];
        if ((opIncContext.find(prevStep->opcode) != opIncContext.end()) && (prevStep->depth != singleInfo.depth))
        {
            // Create new call data entry
            ContextData contextData;
            contextData.type = prevStep->opcode;
            callData[CTX] = contextData;

            zkr = getVarFromCtx(ctx, true, ctx.rom.gasCallOffset, auxScalar);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.gasCallOffset)");
                return zkr;
            }
            TxGAS gas;
            gas.forwarded = 0;
            gas.remaining = auxScalar.get_ui();
            txGAS[depth] = gas;
            if (ctx.proverRequest.input.traceConfig.bGenerateCallTrace)
            {
                singleInfo.contract.gas = gas.remaining;
            }
        }
    }

    // Set contract params depending on current call type
    singleInfo.contract.type = callData[CTX].type;
    if (singleInfo.contract.type == "DELEGATECALL")
    {
        mpz_class auxScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.storageAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.storageAddrOffset)");
            return zkr;
        }
        singleInfo.contract.caller = NormalizeToNFormat(auxScalar.get_str(16), 40);
    }
        
    // If is an ether transfer, don't add stop opcode to trace
    bool bAddOpcode = true;
    if ( (singleInfo.op == 0x00 /*STOP*/) &&
         ( (prevStep==NULL) || ( (opCreate.find(prevStep->opcode) != opCreate.end()) && (prevStep->gas_cost <= 32000))))
    {
        zkr = getVarFromCtx(ctx, false, ctx.rom.bytecodeLengthOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.bytecodeLengthOffset)");
            return zkr;
        }
        if (auxScalar == 0)
        {
            bAddOpcode = false;
        }
    }

    if (bAddOpcode)
    {
        if (ctx.proverRequest.input.traceConfig.bGenerateCallTrace)
        {
            // Save output traces
            call_trace.emplace_back(singleInfo);
        }

        if (ctx.proverRequest.input.traceConfig.bGenerateExecuteTrace)
        {
            // Save output traces
            execution_trace.emplace_back(singleInfo);
        }
    }

#ifdef LOG_TIME_STATISTICS
    tmsop.add("copySingleInfoIntoTraces", TimeDiff(top));
#endif
#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onOpcode() codeId=" + to_string(codeId) + " opcode=" + string(singleInfo.opcode));
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onOpcode", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

#define SMT_KEY_BALANCE 0
#define SMT_KEY_NONCE 1

/*
   Add an address when it is either read/write in the state-tree
   address - address accessed
   keyType - Parameter accessed in the state-tree
   value - value read/write
 */

zkresult FullTracer::addReadWriteAddress ( const Goldilocks::Element &address0, const Goldilocks::Element &address1, const Goldilocks::Element &address2, const Goldilocks::Element &address3, const Goldilocks::Element &address4, const Goldilocks::Element &address5, const Goldilocks::Element &address6, const Goldilocks::Element &address7,
                                           const Goldilocks::Element &keyType0, const Goldilocks::Element &keyType1, const Goldilocks::Element &keyType2, const Goldilocks::Element &keyType3, const Goldilocks::Element &keyType4, const Goldilocks::Element &keyType5, const Goldilocks::Element &keyType6, const Goldilocks::Element &keyType7,
                                           const mpz_class &value )
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    // Get address
    mpz_class address;
    if (!fea2scalar(fr, address, address0, address1, address2, address3, address4, address5, address6, address7))
    {
        zklog.error("FullTracer::addReadWriteAddress() failed calling fea2scalar(address)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    string addressHex = NormalizeTo0xNFormat(address.get_str(16), 40);

    // Get key type
    mpz_class keyType;
    if (!fea2scalar(fr, keyType, keyType0, keyType1, keyType2, keyType3, keyType4, keyType5, keyType6, keyType7))
    {
        zklog.error("FullTracer::addReadWriteAddress() failed calling fea2scalar(keyType)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }

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

    return ZKR_SUCCESS;
}

} // namespace