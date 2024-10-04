#include <iostream>
#include <sys/time.h>
#include <set>
#include "main_sm/fork_13/main/full_tracer.hpp"
#include "main_sm/fork_13/main/context.hpp"
#include "main_sm/fork_13/main/opcode_name.hpp"
#include "main_sm/fork_13/main/eval_command.hpp"
#include "goldilocks_base_field.hpp"
#include "scalar.hpp"
#include "zkassert.hpp"
#include "rlp.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"

//#define LOG_FULL_TRACER

using namespace std;

namespace fork_13
{

set<string> opIncContext = {
    "CALL",
    "STATICCALL",
    "DELEGATECALL",
    "CALLCODE",
    "CREATE",
    "CREATE2" };

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
    "OOCSH",
    "intrinsic_invalid_signature",
    "intrinsic_invalid_chain_id",
    "intrinsic_invalid_nonce",
    "intrinsic_invalid_gas_limit",
    "intrinsic_invalid_gas_overflow",
    "intrinsic_invalid_balance",
    "intrinsic_invalid_batch_gas_limit",
    "intrinsic_invalid_sender_code",
    "invalid_change_l2_block_limit_timestamp",
    "invalid_change_l2_block_min_timestamp",
    "invalidRLP",
    "invalidDecodeChangeL2Block",
    "invalidNotFirstTxChangeL2Block",
    "invalid_l1_info_tree_index" };
    
set<string> invalidBatchErrors = {
    "OOCS",
    "OOCK",
    "OOCB",
    "OOCM",
    "OOCA",
    "OOCPA",
    "OOCPO",
    "OOCSH",
    "invalid_change_l2_block_limit_timestamp",
    "invalid_change_l2_block_min_timestamp",
    "invalidRLP",
    "invalidDecodeChangeL2Block",
    "invalidNotFirstTxChangeL2Block",
    "invalid_l1_info_tree_index" };
    
set<string> changeBlockErrors = {
    "invalid_change_l2_block_limit_timestamp",
    "invalid_change_l2_block_min_timestamp",
    "invalid_l1_info_tree_index" };
    
set<string> oocErrors = {
    "OOCS",
    "OOCK",
    "OOCB",
    "OOCM",
    "OOCA",
    "OOCPA",
    "OOCPO",
    "OOCSH" };

//////////
// UTILS
//////////

// Get range from memory
inline zkresult getFromMemory(Context &ctx, mpz_class &offset, mpz_class &length, string &result, uint64_t * pContext = NULL)
{
    uint64_t offsetCtx = (pContext != NULL) ? *pContext*0x40000 : ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;
    uint64_t addrMem = offsetCtx + 0x20000;

    result = "";

    // If length is too high this is due to an OOG that will stop processing; just pretend to have read nothing
    if (length > ctx.rom.constants.MAX_MEM_EXPANSION_BYTES)
    {
        zklog.warning("getFromMemory() got length=" + length.get_str(10) + " > rom.constants.MAX_MEM_EXPANSION_BYTES=" + to_string(ctx.rom.constants.MAX_MEM_EXPANSION_BYTES));
        return ZKR_SUCCESS;
    }

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
            zklog.error("FullTracer::handleEvent(onFinishTx) found ctx.totalTransferredBalance=" + ctx.totalTransferredBalance.get_str(10) + " lastError=" + lastError);
            return ZKR_SM_MAIN_BALANCE_MISMATCH;
        }
        return onFinishTx(ctx, cmd);
    }
    if (cmd.params[0]->varName == "onStartBlock")
    {
        return onStartBlock(ctx);
    }
    if (cmd.params[0]->varName == "onFinishBlock")
    {
        if ( (oocErrors.find(lastError)==oocErrors.end()) && (ctx.totalTransferredBalance != 0) )
        {
            zklog.error("FullTracer::handleEvent(onFinishBlock) found ctx.totalTransferredBalance=" + ctx.totalTransferredBalance.get_str(10) + " lastError=" + lastError);
            return ZKR_SM_MAIN_BALANCE_MISMATCH;
        }
        return onFinishBlock(ctx);
    }
    if (cmd.params[0]->varName == "onStartBatch")
    {
        return onStartBatch(ctx, cmd);
    }
    if (cmd.params[0]->varName == "onFinishBatch")
    {
        if ( (oocErrors.find(lastError)==oocErrors.end()) && (ctx.totalTransferredBalance != 0) )
        {
            zklog.error("FullTracer::handleEvent(onFinishBatch) found ctx.totalTransferredBalance=" + ctx.totalTransferredBalance.get_str(10) + " lastError=" + lastError);
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

    // Set invalid_batch flag if error invalidates the full batch
    // Continue function to set the error when it has been triggered
    if (invalidBatchErrors.find(lastError) != invalidBatchErrors.end())
    {
        finalTrace.invalid_batch = true;
        finalTrace.error = lastError;

        // Finish setting errors if there is no block processed
        if (!currentBlock.initialized)
        {
#ifdef LOG_FULL_TRACER_ON_ERROR
            zklog.info("FullTracer::onError() 1 error=" + lastError + " zkPC=" + to_string(*ctx.pZKPC) + " rom=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " block=" + to_string(currentBlock.block_number) + " responses.size=" + to_string(currentBlock.responses.size()));
#endif
#ifdef LOG_TIME_STATISTICS
            tms.add("onError", TimeDiff(t));
#endif
            return ZKR_SUCCESS;
        }
    }

    // Set error at block level if error is triggered by the block transaction
    if (changeBlockErrors.find(lastError) != changeBlockErrors.end())
    {
        currentBlock.error = lastError;
#ifdef LOG_FULL_TRACER_ON_ERROR
        zklog.info("FullTracer::onError() 2 error=" + lastError + " zkPC=" + to_string(*ctx.pZKPC) + " rom=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " block=" + to_string(currentBlock.block_number) + " responses.size=" + to_string(currentBlock.responses.size()));
#endif
#ifdef LOG_TIME_STATISTICS
        tms.add("onError", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
    }

    // Set error at block level if error is an invalid batch and there is no transaction processed in that block
    if ((invalidBatchErrors.find(lastError) != invalidBatchErrors.end()) && currentBlock.responses.empty())
    {
        currentBlock.error = lastError;
#ifdef LOG_FULL_TRACER_ON_ERROR
        zklog.info("FullTracer::onError() 3 error=" + lastError + " zkPC=" + to_string(*ctx.pZKPC) + " rom=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " block=" + to_string(currentBlock.block_number) + " responses.size=" + to_string(currentBlock.responses.size()));
#endif
#ifdef LOG_TIME_STATISTICS
        tms.add("onError", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
    }

    // Intrinsic error should be set at tx level (not opcode)
    // Error triggered with no previous opcode set at tx level
    if ( (responseErrors.find(lastError) != responseErrors.end()) ||
         full_trace.empty() )
    {
        if (currentBlock.responses.empty())
        {
            zklog.error("FullTracer::onError() got error=" + lastError + " with txIndex=" + to_string(txIndex) + " but currentBlock.responses.size()=" + to_string(currentBlock.responses.size()));
            exitProcess();
        }
        currentBlock.responses[currentBlock.responses.size() - 1].error = lastError;
        
#ifdef LOG_FULL_TRACER_ON_ERROR
        zklog.info("FullTracer::onError() 4 error=" + lastError + " zkPC=" + to_string(*ctx.pZKPC) + " rom=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " block=" + to_string(currentBlock.block_number) + " responses.size=" + to_string(currentBlock.responses.size()) + " txIndex=" + to_string(txIndex));
#endif
#ifdef LOG_TIME_STATISTICS
        tms.add("onError", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
    }

    if (full_trace.size() > 0)
    {
        full_trace[full_trace.size() - 1].error = lastError;
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
    zklog.info("FullTracer::onError() error=" + lastError + " zkPC=" + to_string(*ctx.pZKPC) + " rom=" + ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) + " block=" + to_string(currentBlock.block_number) + " responses.size=" + to_string(currentBlock.responses.size()));
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
    map<uint64_t, map<uint64_t, LogV2>>::iterator itCTX;
    itCTX = logs.find(CTX);
    if (itCTX == logs.end())
    {
        map<uint64_t, LogV2> aux;
        LogV2 log;
        aux[indexLog] = log;
        logs[CTX] = aux;
        itCTX = logs.find(CTX);
        zkassert(itCTX != logs.end());
    }
    
    map<uint64_t, LogV2>::iterator it;
    it = itCTX->second.find(indexLog);
    if (it == itCTX->second.end())
    {
        LogV2 log;
        logs[CTX][indexLog] = log;
        it = itCTX->second.find(indexLog);
        zkassert(it != itCTX->second.end());
    }

    // Store data in the proper vector
    if (isTopic)
    {
        string dataString = PrependZeros(data.get_str(16), 64);
        it->second.topics.push_back(dataString);
    }
    else
    {
        // Data length is stored in C
        mpz_class cScalar;
        getRegFromCtx(ctx, reg_C, cScalar);

        // Data always should be 32 or less but limit to 32 for safety
        uint64_t size = zkmin(cScalar.get_ui(), 32);

        // Convert data to hex string and append zeros, left zeros are stored in logs, for example if data = 0x01c8 and size=32, data is 0x00000000000000000000000000000000000000000000000000000000000001c8
        string dataString = PrependZeros(data.get_str(16), 64);

        // Get only left size length from bytes, example if size=1 and data= 0xaa00000000000000000000000000000000000000000000000000000000000000, we get 0xaa
        dataString = dataString.substr(0, size * 2);
        
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
    zkr = getVarFromCtx(ctx, true, ctx.rom.blockNumOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onStoreLog() failed calling getVarFromCtx(ctx.rom.blockNumOffset)");
        return zkr;
    }
    it->second.block_number = auxScalar.get_ui();
    if (currentBlock.responses.empty())
    {
        zklog.error("FullTracer::onStoreLog() found currentBlock.responses empty");
        exitProcess();
    }
    it->second.tx_hash = currentBlock.responses[currentBlock.responses.size() - 1].tx_hash;
    it->second.tx_hash_l2 = currentBlock.responses[currentBlock.responses.size() - 1].tx_hash_l2;
    it->second.tx_index = txIndex;
    it->second.index = indexLog;

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onStoreLog() CTX=" + to_string(CTX) + " indexLog=" + to_string(indexLog) + " isTopic=" + to_string(isTopic) + " data=" + dataString);
#endif
#ifdef LOG_TIME_STATISTICS
    tms.add("onStoreLog", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

// Triggered when a change L2 block transaction is detected
zkresult FullTracer::onStartBlock (Context &ctx)
{
#ifdef LOG_FULL_TRACER
    //zklog.info("FullTracer::onStartBlock()");
#endif

    zkresult zkr;
    mpz_class auxScalar;

    // Get block number
    // When this event is triggered, the block number is not updated yet, so we must add 1
    zkr = getVarFromCtx(ctx, true, ctx.rom.blockNumOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onStartBlock() failed calling getVarFromCtx(ctx.rom.blockNumOffset)");
        return zkr;
    }
    if (auxScalar >= ScalarMask64)
    {
        zklog.error("FullTracer::onStartBlock() called getVarFromCtx(ctx.rom.blockNumOffset) but got a too high value=" + auxScalar.get_str(10));
        return zkr;
    }
    uint64_t blockNumber = auxScalar.get_ui();

    // if this.options.skipFirstChangeL2Block is not active
    if (!ctx.proverRequest.input.bSkipFirstChangeL2Block)
    {
        blockNumber++;
    }

    currentBlock.block_number = blockNumber;

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onStartBlock() block=" + to_string(currentBlock.block_number));
#endif

    // Get sequencer address
    zkr = getVarFromCtx(ctx, true, ctx.rom.sequencerAddrOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onStartBlock() failed calling getVarFromCtx(ctx.rom.sequencerAddrOffset)");
        return zkr;
    }
    if (auxScalar > ScalarMask160)
    {
        zklog.error("FullTracer::onStartBlock() called getVarFromCtx(ctx.rom.sequencerAddrOffset) but got a too high value=" + auxScalar.get_str(10));
        return zkr;
    }
    currentBlock.coinbase = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);

    // Get gas limit
    currentBlock.gas_limit = ctx.rom.constants.BLOCK_GAS_LIMIT;

    // Clear transactions
    currentBlock.responses.clear();

    // Clear error
    currentBlock.error.clear();

    // Get context
    currentBlock.ctx = fr.toU64(ctx.pols.CTX[*ctx.pStep]);

    // Mark as initialized
    currentBlock.initialized = true;
    
    // Call startBlock() with the current state root
    if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
    {
        zklog.error("FullTracer::onStartBlock() failed calling fea2scalar(SR)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    string state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);
    ctx.pHashDB->startBlock(ctx.proverRequest.uuid, state_root, ctx.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);

    return ZKR_SUCCESS;
}

/**
 * Triggered when a block is finished (at begining of next block or after finishing processing last tx of the batch)
 * @param {Object} ctx Current context object
 */
zkresult FullTracer::onFinishBlock (Context &ctx)
{
#ifdef LOG_FULL_TRACER
    //zklog.info("FullTracer::onFinishBlock()");
#endif

    mpz_class auxScalar;
    zkresult zkr;

    // Get data from context
    ////////////////////////

    // Get global exit root
    zkr = getVarFromCtx(ctx, false, ctx.rom.gerL1InfoTreeOffset, auxScalar, &currentBlock.ctx);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBlock() failed calling getVarFromCtx(ctx.rom.gerL1InfoTreeOffset)");
        return zkr;
    }
    currentBlock.ger = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // Get block hash L1
    zkr = getVarFromCtx(ctx, false, ctx.rom.blockHashL1InfoTreeOffset, auxScalar, &currentBlock.ctx);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBlock() failed calling getVarFromCtx(ctx.rom.blockHashL1InfoTreeOffset)");
        return zkr;
    }
    currentBlock.block_hash_l1 = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // Get global data
    //////////////////

    // Get parent hash
    zkr = getVarFromCtx(ctx, true, ctx.rom.previousBlockHashOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBlock() failed calling getVarFromCtx(ctx.rom.previousBlockHashOffset)");
        return zkr;
    }
    currentBlock.parent_hash = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // Get block number
    zkr = getVarFromCtx(ctx, true, ctx.rom.blockNumOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBlock() failed calling getVarFromCtx(ctx.rom.blockNumOffset)");
        return zkr;
    }
    currentBlock.block_number = auxScalar.get_ui();

    // Get timestamp
    zkr = getVarFromCtx(ctx, true, ctx.rom.timestampOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBlock() failed calling getVarFromCtx(ctx.rom.timestampOffset)");
        return zkr;
    }
    currentBlock.timestamp = auxScalar.get_ui();

    // Get gas used
    zkr = getVarFromCtx(ctx, true, ctx.rom.cumulativeGasUsedOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBlock() failed calling getVarFromCtx(ctx.rom.cumulativeGasUsedOffset)");
        return zkr;
    }
    currentBlock.gas_used = auxScalar.get_ui();

    // Get block info root
    zkr = getVarFromCtx(ctx, true, ctx.rom.blockInfoSROffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBlock() failed calling getVarFromCtx(ctx.rom.sequencerAddrOffset)");
        return zkr;
    }
    currentBlock.block_info_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);
    
    // Get block hash
    if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
    {
        zklog.error("FullTracer::onFinishBlock() failed calling fea2scalar()");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    currentBlock.block_hash = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // Clear logs
    currentBlock.logs.clear();

    map<uint64_t, LogV2> auxLogs;

    // Add blockhash to all logs on every tx, and add logs to block response
    for (uint64_t r = 0; r < currentBlock.responses.size(); r++)
    {
        // Set block hash to all txs of block
        currentBlock.responses[r].block_hash = currentBlock.block_hash;
        currentBlock.responses[r].block_number = currentBlock.block_number;

        for (uint64_t l = 0; l < currentBlock.responses[r].logs.size(); l++)
        {
            currentBlock.responses[r].logs[l].block_hash = currentBlock.block_hash;

            // Store all logs in auxLogs, in order of index
            auxLogs[currentBlock.responses[r].logs[l].index] = currentBlock.responses[r].logs[l];
        }
    }

    // Append to response logs, overwriting log indexes to be sequential
    map<uint64_t, LogV2>::iterator auxLogsIt;
    for (auxLogsIt = auxLogs.begin(); auxLogsIt != auxLogs.end(); auxLogsIt++)
    {
        // Store block log
        currentBlock.logs.emplace_back(auxLogsIt->second);
    }

    // Append block to final trace
    finalTrace.block_responses.emplace_back(currentBlock);
    currentBlock.initialized = false;

    // Reset logs
    logs.clear();
    
    // Call finishBlock() with the current state root
    if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
    {
        zklog.error("FullTracer::onFinishBlock() failed calling fea2scalar(SR)");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    string state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onFinishBlock() block=" + to_string(currentBlock.block_number));
#endif

    ctx.pHashDB->finishBlock(ctx.proverRequest.uuid, state_root, ctx.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);

    return ZKR_SUCCESS;
}

// Triggered at the very beginning of transaction process
zkresult FullTracer::onProcessTx (Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif
    mpz_class auxScalar;
    ResponseV2 response;
    zkresult zkr;

    // Create new block if:
    // - it is a change L2 block tx
    // - it is forced batch and the currentTx is 1
    zkr = getVarFromCtx(ctx, true, ctx.rom.txIndexOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txIndexOffset)");
        return zkr;
    }
    if (auxScalar > ScalarMask64)
    {
        zklog.error("FullTracer::onProcessTx() called getVarFromCtx(ctx.rom.txIndexOffset) but got a too high value=" + auxScalar.get_str(10));
        return zkr;
    }
    txIndex = auxScalar.get_ui();

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
        response.full_trace.context.type = "CREATE";
        response.full_trace.context.to = "0x";
    }
    else
    {
        response.full_trace.context.type = "CALL";
        zkr = getVarFromCtx(ctx, false, ctx.rom.txDestAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txDestAddrOffset)");
            return zkr;
        }
        response.full_trace.context.to = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
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
    zkr = getFromMemory(ctx, calldataOffset, auxScalar, response.full_trace.context.data, &calldataCTX);
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
    response.full_trace.context.gas = auxScalar.get_ui();

    // TX value
    zkr = getVarFromCtx(ctx, false, ctx.rom.txValueOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txValueOffset)");
        return zkr;
    }
    response.full_trace.context.value = auxScalar;

    // TX output
    response.full_trace.context.output = "";

    // TX gas used
    response.full_trace.context.gas_used = 0;

    // TX execution time
    response.full_trace.context.execution_time = 0;

    // TX old state root
    if (!fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep]))
    {
        zklog.error("FullTracer::onProcessTx() failed calling fea2scalar()");
        return ZKR_SM_MAIN_FEA2SCALAR;
    }
    response.full_trace.context.old_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // TX gas price
    zkr = getVarFromCtx(ctx, false, ctx.rom.txGasPriceRLPOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txGasPriceRLPOffset)");
        return zkr;
    }
    response.full_trace.context.gas_price = auxScalar;

    // TX chain ID
    zkr = getVarFromCtx(ctx, false, ctx.rom.txChainIdOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.txChainIdOffset)");
        return zkr;
    }
    uint64_t chainId = auxScalar.get_ui();
    response.full_trace.context.chainId = chainId;

    // TX index
    response.full_trace.context.txIndex = txIndex;

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
    getTransactionHash( response.full_trace.context.to,
                        response.full_trace.context.value,
                        nonce,
                        response.full_trace.context.gas,
                        response.full_trace.context.gas_price,
                        response.full_trace.context.data,
                        r,
                        s,
                        v,
                        response.tx_hash,
                        response.rlp_tx);
    response.type = 0;
    response.return_value.clear();
    response.gas_left = response.full_trace.context.gas;
    response.gas_used = 0;
    response.gas_refunded = 0;
    response.error = "";
    response.create_address = "";
    response.state_root = response.full_trace.context.old_state_root;
    response.logs.clear();
    response.full_trace.steps.clear();

    // TX hash L2
    zkr = getVarFromCtx(ctx, false, ctx.rom.l2TxHashOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.l2TxHashOffset)");
        return zkr;
    }
    response.tx_hash_l2 = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // Get effective percentage
    zkr = getVarFromCtx(ctx, false, ctx.rom.effectivePercentageRLPOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onProcessTx() failed calling getVarFromCtx(ctx.rom.effectivePercentageRLPOffset)");
        return zkr;
    }
    response.effective_percentage = auxScalar.get_ui();

    // Create block object if flag skipFirstChangeL2Block is active and this.currentBlock has no properties
    if (ctx.proverRequest.input.bSkipFirstChangeL2Block && !currentBlock.initialized)
    {
        onStartBlock(ctx);
    }

    // Create current tx object
    currentBlock.responses.push_back(response);

    // Clear temporary tx traces
    full_trace.clear();
    full_trace.reserve(ctx.config.fullTracerTraceReserveSize);

    // Reset previous memory
    previousMemory = "";
    
    txTime = getCurrentTime();

    // Reset values
    depth = 1;
    deltaStorage.clear();
    txGAS[depth] = {response.full_trace.context.gas, 0};
    lastError = "";

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onProcessTx() currentBlock.responses.size()=" + to_string(currentBlock.responses.size()));
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
         ctx.proverRequest.input.traceConfig.bGenerateFullTrace )
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
        if (full_trace.size() > 0)
        {
            full_trace[full_trace.size() - 1].storage = deltaStorage[storageAddress];
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

// Triggered after processing a transaction
zkresult FullTracer::onFinishTx(Context &ctx, const RomCommand &cmd)
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    // if the 'onFinishTx' is triggered with no previous transactions, do nothing
    // this can happen when the first transaction of the batch is a changeL2BlockTx or a new block is started with no transactions
    if (currentBlock.responses.empty())
    {
#ifdef LOG_FULL_TRACER
        zklog.info("FullTracer::onFinishTx() txIndex=" + to_string(txIndex) + " currentBlock.responses.size()=" + to_string(currentBlock.responses.size()));
#endif
#ifdef LOG_TIME_STATISTICS
        tms.add("onFinishTx", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
    }

    zkresult zkr;
    ResponseV2 &response = currentBlock.responses[currentBlock.responses.size() - 1];

    // Set from address
    mpz_class fromScalar;
    zkr = getVarFromCtx(ctx, true, ctx.rom.txSrcOriginAddrOffset, fromScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txSrcOriginAddrOffset)");
        return zkr;
    }
    response.full_trace.context.from = NormalizeTo0xNFormat(fromScalar.get_str(16), 40);

    // Set effective gas price
    mpz_class auxScalar;
    zkr = getVarFromCtx(ctx, true, ctx.rom.txGasPriceOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txGasPriceOffset)");
        return zkr;
    }
    response.effective_gas_price = Add0xIfMissing(auxScalar.get_str(16));

    // Set cumulative gas used
    zkr = getVarFromCtx(ctx, true, ctx.rom.cumulativeGasUsedOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.cumulativeGasUsedOffset)");
        return zkr;
    }
    response.cumulative_gas_used = auxScalar.get_ui();

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
    response.full_trace.context.gas_used = response.gas_used;
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
        if ( ctx.proverRequest.input.traceConfig.bGenerateFullTrace )
        {
            response.full_trace.context.output = response.return_value;
        }
    }

    // Set create address in case of deploy
    if (response.full_trace.context.to == "0x")
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

    // Set status
    mpz_class statusScalar;
    zkr = getVarFromCtx(ctx, false, ctx.rom.txStatusOffset, statusScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txStatusOffset)");
        return zkr;
    }
    response.status = statusScalar.get_ui();

    // If processed opcodes
    if ( ctx.proverRequest.input.traceConfig.bGenerateFullTrace &&
         (full_trace.size() > 0) )
    {
        Opcode &lastOpcodeCall = full_trace.at(full_trace.size() - 1);

        // Set gas price of last opcode if no error and is not a deploy and is not STOP (RETURN + REVERT)
        if ( (full_trace.size() > 1) &&
             (lastOpcodeCall.op != 0x00 /*STOP opcode*/ ) &&
             (lastOpcodeCall.error.size() == 0) &&
             (response.full_trace.context.to != "0x") )
        {
            lastOpcodeCall.gas_cost = lastOpcodeCall.gas - fr.toU64(ctx.pols.GAS[*ctx.pStep]) + lastOpcodeCall.gas_refund;
        }

        response.full_trace.steps.swap(full_trace);

        if (response.error.size() == 0)
        {
            response.error = lastOpcodeCall.error;
        }
    }
    else if ( ctx.proverRequest.input.bNoCounters &&
              (full_trace.size() > 0) )
    {
        Opcode &lastOpcodeCall = full_trace.at(full_trace.size() - 1);
        if (currentBlock.responses[currentBlock.responses.size() - 1].error == "")
        {
            currentBlock.responses[currentBlock.responses.size() - 1].error = lastOpcodeCall.error;
        }
    }
        
    if ( !ctx.proverRequest.input.traceConfig.bGenerateFullTrace && 
         (numberOfOpcodesInThisTx != 0) &&
         (lastErrorOpcode != numberOfOpcodesInThisTx) )
    {
        currentBlock.responses[currentBlock.responses.size() - 1].error = "";
    }

    // set flags has_gasprice_opcode and has_balance_opcode
    currentBlock.responses[currentBlock.responses.size() - 1].has_gasprice_opcode = hasGaspriceOpcode;
    currentBlock.responses[currentBlock.responses.size() - 1].has_balance_opcode = hasBalanceOpcode;

    // Check TX status
    if ((responseErrors.find(response.error) == responseErrors.end()) &&
        ( (response.error.empty() && (response.status == 0)) ||
          (!response.error.empty() && (response.status == 1)) )
       )
    {
        zklog.error("FullTracer::onFinishTx() invalid TX status-error error=" + response.error + " status=" + to_string(response.status));
        return ZKR_SM_MAIN_INVALID_TX_STATUS_ERROR;
    }

    // Clean aux array for next iteration
    full_trace.clear();
    callData.clear();

    // Reset opcodes counters
    numberOfOpcodesInThisTx = 0;
    lastErrorOpcode = 0;

    // Order all logs (from all CTX) in order of index
    map<uint64_t, LogV2> auxLogs;
    map<uint64_t, map<uint64_t, LogV2>>::iterator logIt;
    map<uint64_t, LogV2>::const_iterator it;
    for (logIt=logs.begin(); logIt!=logs.end(); logIt++)
    {
        for (it = logIt->second.begin(); it != logIt->second.end(); it++)
        {
            auxLogs[it->second.index] = it->second;
        }
    }

    // Append to response logs, overwriting log indexes to be sequential
    map<uint64_t, LogV2>::iterator auxLogsIt;
    uint64_t lastTx = currentBlock.responses.size() - 1;
    currentBlock.responses[lastTx].logs.clear();
    for (auxLogsIt = auxLogs.begin(); auxLogsIt != auxLogs.end(); auxLogsIt++)
    {
        currentBlock.responses[lastTx].logs.push_back(auxLogsIt->second);
    }

    // Reset logs
    logs.clear();

    // Call finishTx()
    ctx.pHashDB->finishTx(ctx.proverRequest.uuid, response.state_root, ctx.proverRequest.input.bUpdateMerkleTree ? PERSISTENCE_DATABASE : PERSISTENCE_CACHE);

#ifdef LOG_FULL_TRACER
    zklog.info("FullTracer::onFinishTx() txIndex=" + to_string(txIndex) + " currentBlock.responses.size()=" + to_string(currentBlock.responses.size()) + " create_address=" + response.create_address + " state_root=" + response.state_root);
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

    // Set is forced
    mpz_class auxScalar;
    zkresult zkr = getVarFromCtx(ctx, true, ctx.rom.isForcedOffset, auxScalar);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishTx() failed calling getVarFromCtx(ctx.rom.txDestAddrOffset)");
        return zkr;
    }
    isForced = auxScalar.get_ui();

    finalTrace.block_responses.clear();
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
    mpz_class auxScalar;
    zkresult zkr;

    finalTrace.gas_used = accBatchGas;

    // New state root
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

    // Call fillInReadWriteAddresses
    zkr = fillInReadWriteAddresses(ctx);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("FullTracer::onFinishBatch() failed calling fillInReadWriteAddresses()");
        return zkr;
    }

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

    if (ctx.proverRequest.input.bNoCounters && (codeId != 0xa0 /* LOG0 */))
    {
        full_trace.emplace_back(singleInfo);
#ifdef LOG_TIME_STATISTICS
        tms.add("onOpcode", TimeDiff(t));
#endif
        return ZKR_SUCCESS;
    }

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
    uint64_t numOpcodes = full_trace.size();
    Opcode * prevTraceCall = (numOpcodes > 0) ? &full_trace.at(numOpcodes - 1) : NULL;

    // In case there is a log0 with 0 data length (and 0 topics), we must add it manually to logs array because it
    // wont be added detected by onStoreLog event
    if (codeId == 0xa0 /* LOG0 */)
    {
        // Get current log index
        zkr = getVarFromCtx(ctx, true, ctx.rom.currentLogIndexOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.currentLogIndexOffset)");
            return zkr;
        }
        uint64_t indexLog = auxScalar.get_ui();

        // Init logs[CTX][indexLog], if required
        uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
        map<uint64_t, map<uint64_t, LogV2>>::iterator itCTX;
        itCTX = logs.find(CTX);
        if (itCTX == logs.end())
        {
            map<uint64_t, LogV2> aux;
            LogV2 log;
            aux[indexLog] = log;
            logs[CTX] = aux;
            itCTX = logs.find(CTX);
            zkassert(itCTX != logs.end());
        }
        
        map<uint64_t, LogV2>::iterator it;
        it = itCTX->second.find(indexLog);
        if (it == itCTX->second.end())
        {
            LogV2 log;
            logs[CTX][indexLog] = log;
            it = itCTX->second.find(indexLog);
            zkassert(it != itCTX->second.end());
        }

        it->second.data.clear();

        // Add log info
        mpz_class auxScalar;
        zkr = getVarFromCtx(ctx, false, ctx.rom.storageAddrOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.storageAddrOffset)");
            return zkr;
        }
        it->second.address = NormalizeTo0xNFormat(auxScalar.get_str(16), 40);
        zkr = getVarFromCtx(ctx, true, ctx.rom.blockNumOffset, auxScalar);
        if (zkr != ZKR_SUCCESS)
        {
            zklog.error("FullTracer::onOpcode() failed calling getVarFromCtx(ctx.rom.blockNumOffset)");
            return zkr;
        }
        it->second.block_number = auxScalar.get_ui();
        if (currentBlock.responses.empty())
        {
            zklog.error("FullTracer::onOpcode() found currentBlock.responses empty");
            exitProcess();
        }
        it->second.tx_hash = currentBlock.responses[currentBlock.responses.size() - 1].tx_hash;
        it->second.tx_hash_l2 = currentBlock.responses[currentBlock.responses.size() - 1].tx_hash_l2;
        it->second.tx_index = txIndex;
        it->second.index = indexLog;
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
    if (ctx.proverRequest.input.traceConfig.bGenerateFullTrace)
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
        }
        else if (opCreate.find(prevTraceCall->opcode) != opCreate.end())
        {
            // In case of error at create, we can't get the gas cost from next opcodes, so we have to use rom variables
            if (prevTraceCall->error.size() > 0)
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
        }
        else if (prevTraceCall->depth != singleInfo.depth)
        {
            // Means opcode failed with error (ex: oog, invalidStaticTx...)
            if (!prevTraceCall->error.empty())
            {
                prevTraceCall->gas_cost = prevTraceCall->gas;
            }
        }
        else
        {
            prevTraceCall->gas_cost = gasCost;
        }

        // cout << "info[info.size() - 1].gas_cost=" << info[info.size() - 1].gas_cost << endl;

        // If gas cost is negative means gas has been added from a deeper context, it should be recalculated
        if (prevTraceCall->gas_cost < 0)
        {
            if (full_trace.size() > 2)
            {
                prevTraceCall->gas_cost = full_trace[full_trace.size() - 2].gas - prevTraceCall->gas;
            }
            else
            {
                zklog.error("FullTracer::onOpcode() found full_trace.size=" + to_string(full_trace.size()) + " too low");
                return ZKR_UNSPECIFIED;
            }
        }

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
            }
        }

        prevTraceCall->duration = TimeDiff(prevTraceCall->startTime, singleInfo.startTime);
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
    if (full_trace.size() > 0)
    {
        prevStep = &full_trace[full_trace.size() - 1];
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
            if (ctx.proverRequest.input.traceConfig.bGenerateFullTrace)
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
         ( (prevStep==NULL) || ( (opCreate.find(prevStep->opcode) != opCreate.end()) && (prevStep->gas_cost <= 32000) && (prevStep->error == ""))))
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
        if (ctx.proverRequest.input.traceConfig.bGenerateFullTrace)
        {
            // Save output traces
            full_trace.emplace_back(singleInfo);
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
#define SMT_KEY_SC_CODE 2
#define SMT_KEY_SC_STORAGE 3
#define SMT_KEY_SC_LENGTH 4

/*
   Add an address when it is either read/write in the state-tree
   address - address accessed
   keyType - Parameter accessed in the state-tree
   value - value read/write
 */

zkresult FullTracer::addReadWriteAddress ( const Goldilocks::Element &address0, const Goldilocks::Element &address1, const Goldilocks::Element &address2, const Goldilocks::Element &address3, const Goldilocks::Element &address4, const Goldilocks::Element &address5, const Goldilocks::Element &address6, const Goldilocks::Element &address7,
                                           const Goldilocks::Element &keyType0, const Goldilocks::Element &keyType1, const Goldilocks::Element &keyType2, const Goldilocks::Element &keyType3, const Goldilocks::Element &keyType4, const Goldilocks::Element &keyType5, const Goldilocks::Element &keyType6, const Goldilocks::Element &keyType7,
                                           const Goldilocks::Element &storageKey0, const Goldilocks::Element &storageKey1, const Goldilocks::Element &storageKey2, const Goldilocks::Element &storageKey3, const Goldilocks::Element &storageKey4, const Goldilocks::Element &storageKey5, const Goldilocks::Element &storageKey6, const Goldilocks::Element &storageKey7,
                                           const mpz_class &value,
                                           const Goldilocks::Element (&key)[4] )
{
#ifdef LOG_TIME_STATISTICS
    gettimeofday(&t, NULL);
#endif

    zkassert(!fr.isZero(key[0]) || !fr.isZero(key[1]) || !fr.isZero(key[2]) || !fr.isZero(key[3]));

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
            infoReadWrite.balanceKey[0]= key[0];
            infoReadWrite.balanceKey[1]= key[1];
            infoReadWrite.balanceKey[2]= key[2];
            infoReadWrite.balanceKey[3]= key[3];
            read_write_addresses[addressHex] = infoReadWrite;
        }
        else
        {
            it->second.balance = value.get_str();
            it->second.balanceKey[0]= key[0];
            it->second.balanceKey[1]= key[1];
            it->second.balanceKey[2]= key[2];
            it->second.balanceKey[3]= key[3];
        }
    }
    else if (keyType == SMT_KEY_NONCE)
    {
        it = read_write_addresses.find(addressHex);
        if (it == read_write_addresses.end())
        {
            InfoReadWrite infoReadWrite;
            infoReadWrite.nonce = value.get_str();
            infoReadWrite.nonceKey[0]= key[0];
            infoReadWrite.nonceKey[1]= key[1];
            infoReadWrite.nonceKey[2]= key[2];
            infoReadWrite.nonceKey[3]= key[3];
            read_write_addresses[addressHex] = infoReadWrite;
        }
        else
        {
            it->second.nonce = value.get_str();
            it->second.nonceKey[0]= key[0];
            it->second.nonceKey[1]= key[1];
            it->second.nonceKey[2]= key[2];
            it->second.nonceKey[3]= key[3];
        }
    }
    else if (keyType == SMT_KEY_SC_CODE)
    {
        it = read_write_addresses.find(addressHex);
        if (it == read_write_addresses.end())
        {
            InfoReadWrite infoReadWrite;
            infoReadWrite.sc_code = value.get_str(16);
            read_write_addresses[addressHex] = infoReadWrite;
        }
        else
        {
            it->second.sc_code = value.get_str(16);
        }
    }
    else if (keyType == SMT_KEY_SC_STORAGE)
    {
        // Get storage key
        mpz_class storageKey;
        if (!fea2scalar(fr, storageKey, storageKey0, storageKey1, storageKey2, storageKey3, storageKey4, storageKey5, storageKey6, storageKey7))
        {
            zklog.error("FullTracer::addReadWriteAddress() failed calling fea2scalar(storageKey)");
            return ZKR_SM_MAIN_FEA2SCALAR;
        }

        it = read_write_addresses.find(addressHex);
        if (it == read_write_addresses.end())
        {
            InfoReadWrite infoReadWrite;
            infoReadWrite.sc_storage[storageKey.get_str(16)] = value.get_str(16);
            read_write_addresses[addressHex] = infoReadWrite;
        }
        else
        {
            it->second.sc_storage[storageKey.get_str(16)] = value.get_str(16);
        }
    }
    else if (keyType == SMT_KEY_SC_LENGTH)
    {
        it = read_write_addresses.find(addressHex);
        if (it == read_write_addresses.end())
        {
            InfoReadWrite infoReadWrite;
            infoReadWrite.sc_length = value.get_str();
            read_write_addresses[addressHex] = infoReadWrite;
        }
        else
        {
            it->second.sc_length = value.get_str();
        }
    }    

#ifdef LOG_TIME_STATISTICS
    tms.add("addReadWriteAddress", TimeDiff(t));
#endif

    return ZKR_SUCCESS;
}

zkresult FullTracer::fillInReadWriteAddresses (Context &ctx)
{
    zkresult zkr;

    // Get new state root fea
    Goldilocks::Element newStateRoot[4];
    string2fea(fr, NormalizeToNFormat(finalTrace.new_state_root, 64), newStateRoot);

    // For all entries in read_write_addresses
    unordered_map<string, InfoReadWrite>::iterator it;
    for (it = read_write_addresses.begin(); it != read_write_addresses.end(); it++)
    {
        // Re-read balance for this state root
        if (!it->second.balance.empty())
        {
            zkassert(!fr.isZero(it->second.balanceKey[0]) || !fr.isZero(it->second.balanceKey[1]) || !fr.isZero(it->second.balanceKey[2]) || !fr.isZero(it->second.balanceKey[3]));
            mpz_class balance;
            zkr = ctx.pHashDB->get(ctx.proverRequest.uuid, newStateRoot, it->second.balanceKey, balance, NULL, NULL);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::fillInReadWriteAddresses() failed calling ctx.pHashDB->get(balance) result=" + zkresult2string(zkr));
                return zkr;
            }
            it->second.balance = balance.get_str();
        }

        // Re-read nonce for this state root
        if (!it->second.nonce.empty())
        {
            zkassert(!fr.isZero(it->second.nonceKey[0]) || !fr.isZero(it->second.nonceKey[1]) || !fr.isZero(it->second.nonceKey[2]) || !fr.isZero(it->second.nonceKey[3]));
            mpz_class nonce;
            zkr = ctx.pHashDB->get(ctx.proverRequest.uuid, newStateRoot, it->second.nonceKey, nonce, NULL, NULL);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("FullTracer::fillInReadWriteAddresses() failed calling ctx.pHashDB->get(nonce) result=" + zkresult2string(zkr));
                return zkr;
            }
            it->second.nonce = nonce.get_str();
        }
    }

    return ZKR_SUCCESS;
}

} // namespace