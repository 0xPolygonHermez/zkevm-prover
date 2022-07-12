#include <iostream>
#include <sys/time.h>
#include <set>
#include "full_tracer.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "context.hpp"
#include "scalar.hpp"
#include "opcode_name.hpp"
#include "zkassert.hpp"
#include "rlp.hpp"

using namespace std;

set<string> opIncContext = { "CALL", "STATICCALL", "DELEGATECALL", "CALLCODE", "CREATE", "CREATE2" };
set<string> opDecContext = { "SELFDESTRUCT", "STOP", "INVALID", "REVERT", "RETURN" };

void FullTracer::handleEvent (Context &ctx, const RomCommand &cmd)
{
    if ( cmd.params[0]->varName == "onError" ) return onProcessTx(ctx, cmd);
    if ( cmd.params[0]->varName == "onProcessTx" ) return onProcessTx(ctx, cmd);
    if ( cmd.params[0]->varName == "onUpdateStorage" ) return onUpdateStorage(ctx, cmd);
    if ( cmd.params[0]->varName == "onFinishTx" ) return onFinishTx(ctx, cmd);
    if ( cmd.params[0]->varName == "onStartBatch" ) return onStartBatch(ctx, cmd);
    if ( cmd.params[0]->varName == "onFinishBatch" ) return onFinishBatch(ctx, cmd);
    if ( cmd.params[0]->varName == "onOpcode" ) return onOpcode(ctx, cmd);
    if ( cmd.funcName == "storeLog" ) return onStoreLog(ctx, cmd);
    cerr << "FullTracer::handleEvent() got an invalid event name=" << cmd.params[0]->varName << endl;
    exit(-1);
}

void FullTracer::onError (Context &ctx, const RomCommand &cmd)
{
    // Store the error
    string errorName = cmd.params[1]->varName;
    info[info.size()-1].error = errorName;
    depth--;

    // Revert logs
    uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    if (logs.find(CTX) != logs.end())
    {
        logs.erase(CTX);
    }

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onError() error=" << errorName << endl;
#endif
}

void FullTracer::onStoreLog (Context &ctx, const RomCommand &cmd)
{
    // Get indexLog from the provided register value
    mpz_class indexLogScalar;
    getRegFromCtx(ctx, cmd.params[0]->regName, indexLogScalar);
    uint64_t indexLog = indexLogScalar.get_ui();

    // Get isTopic
    uint64_t isTopic = cmd.params[1]->num.get_ui();

    // Get data
    mpz_class data;
    getRegFromCtx(ctx, cmd.params[2]->regName, data);

    // Init logs[CTX][indexLog], if required
    uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
    if (logs.find(CTX) == logs.end())
    {
        map<uint64_t,Log> aux;
        logs[CTX] = aux;
    }
    if (logs[CTX].find(indexLog) == logs[CTX].end())
    {
        Log log;
        logs[CTX][indexLog] = log;
    }

    // Store data in the proper vector
    string dataString = NormalizeToNFormat(data.get_str(16), 64);
    if (isTopic)
    {
        logs[CTX][indexLog].topics.push_back(dataString);
    }
    else
    {
        logs[CTX][indexLog].data.push_back(dataString);
    }

    //Add log info
    mpz_class auxScalar;
    getVarFromCtx(ctx, false, "txDestAddr", auxScalar);
    logs[CTX][indexLog].address = auxScalar.get_str(16);
    logs[CTX][indexLog].batch_number = finalTrace.numBatch;
    logs[CTX][indexLog].tx_hash = finalTrace.responses[txCount].tx_hash;
    logs[CTX][indexLog].tx_index = txCount;
    logs[CTX][indexLog].batch_hash = finalTrace.globalHash;
    logs[CTX][indexLog].index = indexLog;

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onStoreLog() CTX=" << to_string(CTX) << " indexLog=" << indexLog << " isTopic=" << to_string(isTopic) << " data=" << dataString << endl;
#endif
}

// Triggered at the very beginning of transaction process
void FullTracer::onProcessTx (Context &ctx, const RomCommand &cmd)
{
    mpz_class auxScalar;
    Response response;

    /* Fill context object */

    // TX from
    getVarFromCtx(ctx, false, "txSrcAddr", auxScalar);
    response.call_trace.context.from = Add0xIfMissing(auxScalar.get_str(16));

    // TX to
    getVarFromCtx(ctx, false, "txDestAddr", auxScalar);
    response.call_trace.context.to = Add0xIfMissing(auxScalar.get_str(16));
    if (response.call_trace.context.to.size() < 5)
    {
        response.call_trace.context.to = "0x0";
    }

    // TX type
    response.call_trace.context.type = (response.call_trace.context.to == "0x0") ? "CREATE" : "CALL";

    // TX data
    getVarFromCtx(ctx, false, "txCalldataLen", auxScalar);
    getCalldataFromStack(ctx, 0, auxScalar.get_ui(), response.call_trace.context.data);

    // TX gas
    getVarFromCtx(ctx, false, "txGasLimit", auxScalar);
    response.call_trace.context.gas = auxScalar.get_ui();

    // TX value
    getVarFromCtx(ctx, false, "txValue", auxScalar);
    response.call_trace.context.value = auxScalar.get_ui();

    // TX batch
    response.call_trace.context.batch = finalTrace.globalHash;

    // TX output
    response.call_trace.context.output = "";

    // TX used gas
    response.call_trace.context.gas_used = 0;

    // TX execution time
    response.call_trace.context.execution_time = 0;

    // TX old state root
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    response.call_trace.context.old_state_root = Add0xIfMissing(auxScalar.get_str(16));

    response.call_trace.context.logs.clear(); // TODO: is this needed?  Not present in JS any more
    response.call_trace.context.error = ""; // TODO: is this needed?  Not present in JS any more

    // TX nonce
    getVarFromCtx(ctx, false, "txNonce", auxScalar);
    response.call_trace.context.nonce = auxScalar.get_ui();

    // TX gas price
    getVarFromCtx(ctx, false, "txGasPrice", auxScalar);
    response.call_trace.context.gasPrice = auxScalar.get_ui();

    // TX chain ID
    getVarFromCtx(ctx, false, "txChainId", auxScalar);
    response.call_trace.context.chainId = auxScalar.get_ui();

    /* Fill response object */

    // TX hash
    response.tx_hash = getTransactionHash( ctx,
                                           response.call_trace.context.from,
                                           response.call_trace.context.to,
                                           response.call_trace.context.value,
                                           response.call_trace.context.nonce,
                                           response.call_trace.context.gas,
                                           response.call_trace.context.gasPrice,
                                           response.call_trace.context.data,
                                           response.call_trace.context.chainId );
    response.type = 0;
    response.return_value.clear();
    response.gas_left = response.call_trace.context.gas;
    response.gas_used = 0;
    response.gas_refunded = 0;
    response.error = "";
    response.create_address = "";
    response.state_root = response.call_trace.context.old_state_root;
    response.logs.clear();
    response.unprocessed_transaction = false;
    response.call_trace.steps.clear();
    response.execution_trace.clear();

    // Create current tx object
    finalTrace.responses.push_back(response);
    txTime = getCurrentTime();

    // Reset values
    depth = 1;
    deltaStorage.clear();
    map<string,string> auxMap;
    deltaStorage[1] = auxMap;
    txGAS[depth] = response.call_trace.context.gas;

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onProcessTx() finalTrace.responses.size()=" << finalTrace.responses.size() << endl;
#endif
}

// Triggered when storage is updated in opcode processing
void FullTracer::onUpdateStorage (Context &ctx, const RomCommand &cmd)
{
    string regName;
    mpz_class regScalar;

    // The storage key is stored in C
    regName = "C";
    getRegFromCtx(ctx, regName, regScalar);
    string key;
    key = NormalizeToNFormat(regScalar.get_str(16), 64);

    // The storage value is stored in D
    regName = "D";
    getRegFromCtx(ctx, regName, regScalar);
    string value;
    value = NormalizeToNFormat(regScalar.get_str(16), 64);

    deltaStorage[depth][key] = value; // TODO: Do we need to init it previously, e.g. with empty strings?

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onUpdateStorage() depth=" << depth << " key=" << key << " value=" << value << endl;
#endif
}

// Triggered after processing a transaction
void FullTracer::onFinishTx (Context &ctx, const RomCommand &cmd)
{
    Response &response = finalTrace.responses[txCount];

    //Set consumed tx gas
    response.gas_used = response.gas_left - fr.toU64(ctx.pols.GAS[*ctx.pStep]); // Using u64 in C instead of string in JS
    response.call_trace.context.gas_used = response.gas_used;
    accBatchGas += response.gas_used;

    // Set return data
    mpz_class offsetScalar;
    getVarFromCtx(ctx, false, "retDataOffset", offsetScalar);
    mpz_class lengthScalar;
    getVarFromCtx(ctx, false, "retDataLength", lengthScalar);
    if (response.call_trace.context.to == "0x0")
    {
        getCalldataFromStack(ctx, offsetScalar.get_ui(), lengthScalar.get_ui(), response.return_value);
    }
    else
    {
        getFromMemory(ctx, offsetScalar, lengthScalar, response.return_value);
    }

    //Set create address in case of deploy
    if (response.call_trace.context.to == "0x0") {
        mpz_class addressScalar;
        getVarFromCtx(ctx, false, "txDestAddr", addressScalar);
        response.create_address = addressScalar.get_str(16);
    }

    //Set gas left
    response.gas_left -= response.gas_used;

    //Set new State Root
    mpz_class auxScalar;
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    response.state_root = Add0xIfMissing(auxScalar.get_str(16));

    //If processed opcodes
    if (info.size() > 0)
    {
        Opcode lastOpcode = info[info.size() - 1];

        // Set gas price of last opcode
        if (info.size() >= 2)
        {
            Opcode beforeLastOpcode = info[info.size() - 2];
            lastOpcode.gasCost = beforeLastOpcode.remaining_gas - lastOpcode.remaining_gas;
        }

        //Add last opcode
        call_trace.push_back(lastOpcode);
        execution_trace.push_back(lastOpcode);
        if (call_trace.size() < info.size())
        {
            call_trace.erase(call_trace.begin());
            execution_trace.erase(execution_trace.begin());
        }

        //Append processed opcodes to the transaction object
        finalTrace.responses[finalTrace.responses.size() - 1].execution_trace = execution_trace;
        finalTrace.responses[finalTrace.responses.size() - 1].call_trace.steps = call_trace; // TODO: Append? This is replacing the vector...
        finalTrace.responses[finalTrace.responses.size() - 1].error = lastOpcode.error;

        // Remove not requested data
        if (!ctx.proverRequest.bGenerateExecuteTrace)
        {
            finalTrace.responses[finalTrace.responses.size() - 1].execution_trace.clear();
        }
        if (!ctx.proverRequest.bGenerateCallTrace)
        {
            finalTrace.responses[finalTrace.responses.size() - 1].call_trace.steps.clear();
        }

    }

    // Clean aux array for next iteration
    call_trace.clear();
    execution_trace.clear();

    // Append to response logs
    //for(const l of this.logs) {
    //    this.finalTrace.responses[this.txCount].logs = this.finalTrace.responses[this.txCount].logs.concat(Object.values(l)); // TODO: What is this?
    //}

    // Increase transaction count
    txCount++;

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onFinishTx() txCount=" << txCount << " finalTrace.responses.size()=" << finalTrace.responses.size() << " create_address=" << response.create_address << " state_root=" << response.state_root << endl;
#endif
}

void FullTracer::onStartBatch (Context &ctx, const RomCommand &cmd)
{
    if (finalTrace.bInitialized) return;

    mpz_class auxScalar;

    // Batch hash
    getRegFromCtx(ctx, cmd.params[1]->regName, auxScalar);
    finalTrace.batchHash = Add0xIfMissing(auxScalar.get_str(16));

    // Old state root
    getVarFromCtx(ctx, true, "oldStateRoot", auxScalar);
    finalTrace.old_state_root = Add0xIfMissing(auxScalar.get_str(16));

    // Global hash
    getVarFromCtx(ctx, true, "globalHash", auxScalar);
    finalTrace.globalHash = Add0xIfMissing(auxScalar.get_str(16));

    // Number of batch
    getVarFromCtx(ctx, true, "numBatch", auxScalar);
    finalTrace.numBatch = auxScalar.get_ui();

    // Timestamp
    getVarFromCtx(ctx, true, "timestamp", auxScalar);
    finalTrace.timestamp = auxScalar.get_ui();

    // Sequencer address
    getVarFromCtx(ctx, true, "sequencerAddr", auxScalar);
    finalTrace.sequencerAddr = Add0xIfMissing(auxScalar.get_str(16));

    finalTrace.responses.clear();

    finalTrace.bInitialized = true;

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onStartBatch() old_state_root=" << finalTrace.old_state_root << endl;
#endif
}

void FullTracer::onFinishBatch (Context &ctx, const RomCommand &cmd)
{
    // Update used gas
    finalTrace.cumulative_gas_used = accBatchGas;

    // New state root
    mpz_class auxScalar;
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    finalTrace.new_state_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // New local exit root
    getVarFromCtx(ctx, true, "newLocalExitRoot", auxScalar);
    finalTrace.new_local_exit_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onFinishBatch() new_state_root=" << finalTrace.new_state_root << endl;
#endif
}

void FullTracer::onOpcode (Context &ctx, const RomCommand &cmd)
{
    Opcode singleInfo;

    // Get opcode info

    // Code ID = register B
    mpz_class auxScalar;
    fea2scalar(ctx.fr, auxScalar, ctx.pols.B0[*ctx.pStep], ctx.pols.B1[*ctx.pStep], ctx.pols.B2[*ctx.pStep], ctx.pols.B3[*ctx.pStep], ctx.pols.B4[*ctx.pStep], ctx.pols.B5[*ctx.pStep], ctx.pols.B6[*ctx.pStep], ctx.pols.B7[*ctx.pStep] );
    zkassert(auxScalar<256);
    uint8_t codeId = auxScalar.get_ui();

    // Opcode = name (except "op")
    string opcode = opcodeName[codeId]+2;

    // store memory
    uint64_t offsetCtx = fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    uint64_t addrMem = 0;
    addrMem += offsetCtx;
    addrMem += 0x30000;

    string finalMemory;
    uint64_t lengthMemOffset = findOffsetLabel(ctx, "memLength");
    uint64_t lenMemValueFinal = 0;
    if (ctx.mem.find(offsetCtx + lengthMemOffset) != ctx.mem.end())
    {
        Fea lenMemValue = ctx.mem[offsetCtx + lengthMemOffset];
        fea2scalar(ctx.fr, auxScalar, lenMemValue.fe0, lenMemValue.fe1, lenMemValue.fe2, lenMemValue.fe3, lenMemValue.fe4, lenMemValue.fe5, lenMemValue.fe6, lenMemValue.fe7);
        lenMemValueFinal = auxScalar.get_ui();
    }

    for (uint64_t i = 0; i < lenMemValueFinal; i++)
    {
        if (ctx.mem.find(addrMem + i) == ctx.mem.end()) continue;
        Fea memValue = ctx.mem[addrMem + i];
        fea2scalar(ctx.fr, auxScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7);
        string hexString = auxScalar.get_str(16);
        //if ((hexString.size() % 2) > 0) hexString = "0" + hexString;
        hexString = NormalizeToNFormat(hexString, 64);
        finalMemory += hexString;
    }

    // store stack
    uint64_t addr = 0;
    addr += offsetCtx;
    addr += 0x20000;

    vector<uint64_t> finalStack;
    uint16_t sp = fr.toU64(ctx.pols.SP[*ctx.pStep]);
    for (uint16_t i=0; i<sp; i++)
    {
        if (ctx.mem.find(addr + i) == ctx.mem.end()) continue;
        Fea stack = ctx.mem[addr + i];
        mpz_class stackScalar;
        fea2scalar(ctx.fr, stackScalar, stack.fe0, stack.fe1, stack.fe2, stack.fe3, stack.fe4, stack.fe5, stack.fe6, stack.fe7 );
        //string hexString = stackScalar.get_str(16);
        //if ((hexString.size() % 2) > 0) hexString = "0" + hexString;
        //hexString = "0x" + hexString;
        finalStack.push_back(stackScalar.get_ui());
    }

    // add info opcodes
    singleInfo.depth = depth;
    singleInfo.pc = fr.toU64(ctx.pols.PC[*ctx.pStep]);
    singleInfo.remaining_gas = fr.toU64(ctx.pols.GAS[*ctx.pStep]);
    if (info.size() > 0)
    {
        Opcode prevTrace = info[info.size() - 1];

        // The gas cost of the opcode is gas before - gas after processing the opcode
        int64_t gasCost = int64_t(prevTrace.remaining_gas) - int64_t(fr.toU64(ctx.pols.GAS[*ctx.pStep]));
        prevTrace.gasCost = gasCost;

        // If negative gasCost means gas has been added from a deeper context, we should recalculate
        if (prevTrace.gasCost < 0)
        {
            Opcode beforePrevTrace = info[info.size() - 2]; // TODO: protect agains -2
            prevTrace.gasCost = beforePrevTrace.remaining_gas - prevTrace.remaining_gas;
        }
    }

    singleInfo.opcode = opcode;

    getVarFromCtx(ctx, false, "gasRefund", auxScalar);
    singleInfo.refund = auxScalar.get_ui();

    singleInfo.op = codeId;
    singleInfo.error = "";

    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    singleInfo.state_root = Add0xIfMissing(auxScalar.get_str(16));

    //Add contract info
    getVarFromCtx(ctx, false, "txDestAddr", auxScalar);
    singleInfo.contract.address = auxScalar.get_str(16);

    getVarFromCtx(ctx, false, "txSrcAddr", auxScalar);
    singleInfo.contract.caller = auxScalar.get_str(16);

    getVarFromCtx(ctx, false, "txValue", auxScalar);
    singleInfo.contract.value = auxScalar.get_ui();

    getCalldataFromStack(ctx, 0, 0, singleInfo.contract.data);

    singleInfo.contract.gas = txGAS[depth];

    singleInfo.storage = deltaStorage[depth];

    // Round up to next multiple of 32
    getVarFromCtx(ctx, false, "memLength", auxScalar);
    singleInfo.memory_size = (auxScalar.get_ui()/32)*32;

    info.push_back(singleInfo);
    fullStack.push_back(finalStack);

    // build trace
    uint64_t index = fullStack.size();

    if (index > 1)
    {
        Opcode singleCallTrace = info[index - 2];
        singleCallTrace.stack = finalStack;
        singleCallTrace.memory = finalMemory;

        Opcode singleExecuteTrace = info[index - 2];
        singleCallTrace.storage.clear();
        singleCallTrace.memory_size = 0;
        singleExecuteTrace.contract.address = "";
        singleExecuteTrace.contract.caller = "";
        singleExecuteTrace.contract.data = "";
        singleExecuteTrace.contract.gas = 0;
        singleExecuteTrace.contract.value = 0;
        call_trace.push_back(singleCallTrace);
        execution_trace.push_back(singleExecuteTrace);
    }

    // Return data
    singleInfo.return_data.clear();

    //Check previous step
    if (info.size() >= 2)
    {
        Opcode prevStep = info[info.size() - 2];
        if (opIncContext.find(prevStep.opcode) != opIncContext.end())
        {
            //Set gasCall when depth has changed
            getVarFromCtx(ctx, true, "gasCall", auxScalar);
            txGAS[depth] = auxScalar.get_ui();
            if (ctx.proverRequest.bGenerateCallTrace)
            {
                singleInfo.contract.gas = txGAS[depth];
            }
        }
    }

    //Check opcodes that alter depth
    if (opDecContext.find(singleInfo.opcode) != opDecContext.end())
    {
        depth--;
    }
    if (opIncContext.find(singleInfo.opcode) != opIncContext.end())
    {
        depth++;
        map<string,string> auxMap;
        deltaStorage[depth] = auxMap;
    }

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onOpcode() codeId=" << to_string(codeId) << " opcode=" << opcode << endl;
#endif
}

//////////
// UTILS
//////////

//Get range from memory
void FullTracer::getFromMemory(Context &ctx, mpz_class &offset, mpz_class &length, string &result)
{
    uint64_t offsetCtx = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    uint64_t addrMem = 0;
    addrMem += offsetCtx;
    addrMem += 0x30000;

    result = "";
    uint64_t init = addrMem + offset.get_ui()/32;
    uint64_t end = init + length.get_ui()/32;
    for (uint64_t i=init; i<end; i++)
    {
        mpz_class memScalar = 0;
        if (ctx.mem.find(i) != ctx.mem.end())
        {
            Fea memValue = ctx.mem[i];
            fea2scalar(ctx.fr, memScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7);
        }
        result += NormalizeToNFormat(memScalar.get_str(16), 64);
    }
}

// Get a global or context variable
void FullTracer::getVarFromCtx (Context &ctx, bool global, const char * pVarLabel, mpz_class &result)
{
    uint64_t offsetCtx = global ? 0 : fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;
    uint64_t offsetRelative = findOffsetLabel(ctx, pVarLabel);
    uint64_t addressMem = offsetCtx + offsetRelative;
    if (ctx.mem.find(addressMem) == ctx.mem.end())
    {
        cout << "FullTracer::getVarFromCtx() could not find in ctx.mem address=" << pVarLabel << "=" << offsetRelative << endl;
        result = 0;
    }
    else
    {
        Fea value = ctx.mem[addressMem];
        fea2scalar(ctx.fr, result, value.fe0, value.fe1, value.fe2, value.fe3, value.fe4, value.fe5, value.fe6, value.fe7);
    }
}

//Get the stored calldata in the stack
void FullTracer::getCalldataFromStack (Context &ctx, uint64_t offset, uint64_t length, string &result)
{
    uint64_t addr = 0x20000 + 1024 + fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    result = "0x";
    for (uint64_t i = addr + offset; i < 0x30000 + fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000; i++)
    {
        if (ctx.mem.find(i) == ctx.mem.end())
        {
            break;
        }
        Fea memVal = ctx.mem[i];
        mpz_class auxScalar;
        fea2scalar(ctx.fr, auxScalar, memVal.fe0, memVal.fe1, memVal.fe2, memVal.fe3, memVal.fe4, memVal.fe5, memVal.fe6, memVal.fe7);
        result += NormalizeToNFormat(auxScalar.get_str(16), 64);
        result += auxScalar.get_str(16);
    }
    if (length > 0)
    {
        result = result.substr(0, 2 + length*2);
    }
    if (result.size() <= 2)
    {
        result = "0x0";
    }
}

// Get the value of a reg (A, B, C, D, E...)
void FullTracer::getRegFromCtx (Context &ctx, string &reg, mpz_class &result)
{
    if (reg == "A") return fea2scalar(ctx.fr, result, ctx.pols.A0[*ctx.pStep], ctx.pols.A1[*ctx.pStep], ctx.pols.A2[*ctx.pStep], ctx.pols.A3[*ctx.pStep], ctx.pols.A4[*ctx.pStep], ctx.pols.A5[*ctx.pStep], ctx.pols.A6[*ctx.pStep], ctx.pols.A7[*ctx.pStep] );
    if (reg == "B") return fea2scalar(ctx.fr, result, ctx.pols.B0[*ctx.pStep], ctx.pols.B1[*ctx.pStep], ctx.pols.B2[*ctx.pStep], ctx.pols.B3[*ctx.pStep], ctx.pols.B4[*ctx.pStep], ctx.pols.B5[*ctx.pStep], ctx.pols.B6[*ctx.pStep], ctx.pols.B7[*ctx.pStep] );
    if (reg == "C") return fea2scalar(ctx.fr, result, ctx.pols.C0[*ctx.pStep], ctx.pols.C1[*ctx.pStep], ctx.pols.C2[*ctx.pStep], ctx.pols.C3[*ctx.pStep], ctx.pols.C4[*ctx.pStep], ctx.pols.C5[*ctx.pStep], ctx.pols.C6[*ctx.pStep], ctx.pols.C7[*ctx.pStep] );
    if (reg == "D") return fea2scalar(ctx.fr, result, ctx.pols.D0[*ctx.pStep], ctx.pols.D1[*ctx.pStep], ctx.pols.D2[*ctx.pStep], ctx.pols.D3[*ctx.pStep], ctx.pols.D4[*ctx.pStep], ctx.pols.D5[*ctx.pStep], ctx.pols.D6[*ctx.pStep], ctx.pols.D7[*ctx.pStep] );
    if (reg == "E") return fea2scalar(ctx.fr, result, ctx.pols.E0[*ctx.pStep], ctx.pols.E1[*ctx.pStep], ctx.pols.E2[*ctx.pStep], ctx.pols.E3[*ctx.pStep], ctx.pols.E4[*ctx.pStep], ctx.pols.E5[*ctx.pStep], ctx.pols.E6[*ctx.pStep], ctx.pols.E7[*ctx.pStep] );

    cerr << "FullTracer::getRegFromCtx() invalid register name=" << reg << endl;
    exit(-1);
}

uint64_t FullTracer::findOffsetLabel (Context &ctx, const char * pLabel)
{
    string label = pLabel;
    return ctx.rom.getMemoryOffset(label);
}

uint64_t FullTracer::getCurrentTime (void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000000 + tv.tv_usec;
}

// Returns a transaction hash from transaction params
string FullTracer::getTransactionHash(Context &ctx, string &from, string &to, uint64_t value, uint64_t nonce, uint64_t gasLimit, uint64_t gasPrice, string &data, uint64_t chainId)
{
    string raw;

    mpz_class ctxR;
    getVarFromCtx(ctx, false, "txR", ctxR);

    mpz_class ctxS;
    getVarFromCtx(ctx, false, "txS", ctxS);

    mpz_class ctxV;
    getVarFromCtx(ctx, false, "txV", ctxV);


    encodeUInt64(raw, nonce);
    encodeUInt64(raw, gasPrice);
    encodeUInt64(raw, gasLimit);
    encodeLen(raw, getHexValueLen(to));
    if (!encodeHexValue(raw, to)) {
        cout << "ERROR encoding to" << endl;
    }
    encodeUInt64(raw, value);
    encodeLen(raw, getHexValueLen(data));
    if (!encodeHexValue(raw, data)) {
        cout << "ERROR encoding data" << endl;
    }

    uint64_t recoveryParam;
    uint64_t v = ctxV.get_ui();

    if (v == 0 || v == 1) {
        recoveryParam = v;
    } else {
        recoveryParam = 1 - (v % 2);
    }
    uint64_t vToEncode = recoveryParam + 27;

    if (chainId) {
        vToEncode += chainId * 2 + 8;
    }

    encodeUInt64(raw, vToEncode);

    string r = ctxR.get_str(16);
    encodeLen(raw, getHexValueLen(r));
    if (!encodeHexValue(raw, r)) {
        cout << "ERROR encoding r" << endl;
    }

    string s = ctxS.get_str(16);
    encodeLen(raw, getHexValueLen(s));
    if (!encodeHexValue(raw, s)) {
        cout << "ERROR encoding s" << endl;
    }

    string res;
    encodeLen(res, raw.length(), true);
    res += raw;

    return keccak256((const uint8_t *)(res.c_str()), res.length());
}
