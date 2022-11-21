#include <iostream>
#include <sys/time.h>
#include <set>
#include "full_tracer.hpp"
#include "goldilocks_base_field.hpp"
#include "context.hpp"
#include "scalar.hpp"
#include "opcode_name.hpp"
#include "zkassert.hpp"
#include "rlp.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "eval_command.hpp"

using namespace std;

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
    
set<string> responseErrors = {
    "OOC", // TODO: Delete when new rom is available
    "OOCS",
    "OOCK",
    "OOCB",
    "OOCM",
    "OOCA",
    "OOCPA",
    "OOCPO",
    "intrinsic_invalid", // TODO: Delete when new rom is available
    "intrinsic_invalid_signature",
    "intrinsic_invalid_chain_id",
    "intrinsic_invalid_nonce",
    "intrinsic_invalid_gas_limit",
    "intrinsic_invalid_balance",
    "intrinsic_invalid_batch_gas_limit",
    "intrinsic_invalid_sender_code" };

void FullTracer::handleEvent (Context &ctx, const RomCommand &cmd)
{
    if ( cmd.function == f_storeLog )
    {
        //if (ctx.proverRequest.bNoCounters) return;
        return onStoreLog(ctx, cmd);
    }
    if (cmd.params.size() == 0)
    {
        cerr << "FullTracer::handleEvent() got an invalid event with cmd.params.size()==0 cmd.function=" << function2String(cmd.function) << endl;
        exitProcess();
    }
    if ( cmd.params[0]->varName == "onError" ) return onError(ctx, cmd);
    if ( cmd.params[0]->varName == "onProcessTx" ) return onProcessTx(ctx, cmd);
    if ( cmd.params[0]->varName == "onUpdateStorage" )
    {
        //if (ctx.proverRequest.bNoCounters) return;
        return onUpdateStorage(ctx, cmd);
    }
    if ( cmd.params[0]->varName == "onFinishTx" ) return onFinishTx(ctx, cmd);
    if ( cmd.params[0]->varName == "onStartBatch" ) return onStartBatch(ctx, cmd);
    if ( cmd.params[0]->varName == "onFinishBatch" ) return onFinishBatch(ctx, cmd);
    if ( cmd.params[0]->function == f_onOpcode )
    {
        //if (ctx.proverRequest.bNoCounters) return;
        return onOpcode(ctx, cmd);
    }
    cerr << "FullTracer::handleEvent() got an invalid event cmd.params[0]->varName=" << cmd.params[0]->varName << " cmd.function=" << function2String(cmd.function) << endl;
    exitProcess();
}

void FullTracer::onError (Context &ctx, const RomCommand &cmd)
{
    // Check params size
    if (cmd.params.size() != 2)
    {
        cerr << "FullTracer::onError() got an invalid cmd.params.size()=" << cmd.params.size() << endl;
        exitProcess();
    }

    // Store the error
    lastError = cmd.params[1]->varName;

    // Intrinsic error should be set at tx level (not opcode)
    if (responseErrors.find(lastError) != responseErrors.end())
    {
        finalTrace.responses[txCount].error = lastError;
    }
    else
    {
        if (info.size() > 0)
        {
            info[info.size()-1].error = lastError;
        }

        // Revert logs
        uint64_t CTX = ctx.fr.toU64(ctx.pols.CTX[*ctx.pStep]);
        if (logs.find(CTX) != logs.end())
        {
            logs.erase(CTX);
        }
    }

#ifdef LOG_FULL_TRACER_ON_ERROR
    cout << "FullTracer::onError() error=" << lastError << " zkPC=" << *ctx.pZKPC << " rom=" << ctx.rom.line[*ctx.pZKPC].toString(ctx.fr) << endl;
#endif
}

void FullTracer::onStoreLog (Context &ctx, const RomCommand &cmd)
{
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
    if (logs.find(CTX) == logs.end())
    {
        unordered_map<uint64_t,Log> aux;
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
    logs[CTX][indexLog].batch_hash = finalTrace.newAccInputHash;
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

    // TX to
    getVarFromCtx(ctx, false, "txDestAddr", auxScalar);
    string to = Add0xIfMissing(auxScalar.get_str(16));
    response.call_trace.context.to = (to == "0x0") ? "0x" : to;

    // TX type
    response.call_trace.context.type = (to == "0x0") ? "CREATE" : "CALL";

    // TX data
    getVarFromCtx(ctx, false, "txCalldataLen", auxScalar);
    getCalldataFromStack(ctx, 0, auxScalar.get_ui(), response.call_trace.context.data);

    // TX gas
    getVarFromCtx(ctx, false, "txGasLimit", auxScalar);
    response.call_trace.context.gas = auxScalar.get_ui();

    // TX value
    getVarFromCtx(ctx, false, "txValue", auxScalar);
    response.call_trace.context.value = auxScalar;

    // TX batch
    response.call_trace.context.batch = finalTrace.newAccInputHash;

    // TX output
    response.call_trace.context.output = "";

    // TX used gas
    response.call_trace.context.gas_used = 0;

    // TX execution time
    response.call_trace.context.execution_time = 0;

    // TX old state root
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    response.call_trace.context.old_state_root = Add0xIfMissing(auxScalar.get_str(16));

    // TX nonce
    getVarFromCtx(ctx, false, "txNonce", auxScalar);
    response.call_trace.context.nonce = auxScalar.get_ui();

    // TX gas price
    getVarFromCtx(ctx, false, "txGasPriceRLP", auxScalar);
    response.call_trace.context.gasPrice = auxScalar;

    // TX chain ID
    getVarFromCtx(ctx, false, "txChainId", auxScalar);
    response.call_trace.context.chainId = auxScalar.get_ui();

    /* Fill response object */


    mpz_class r;
    getVarFromCtx(ctx, false, "txR", r);

    mpz_class s;
    getVarFromCtx(ctx, false, "txS", s);

    mpz_class ctxV;
    getVarFromCtx(ctx, false, "txV", ctxV);
    uint64_t v = ctxV.get_ui() - 27 + response.call_trace.context.chainId*2 + 35;

    // TX hash
    getTransactionHash( response.call_trace.context.to,
                        response.call_trace.context.value,
                        response.call_trace.context.nonce,
                        response.call_trace.context.gas,
                        response.call_trace.context.gasPrice,
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
    unordered_map<string,string> auxMap;
    deltaStorage[depth] = auxMap;
    txGAS[depth] = response.call_trace.context.gas;
    lastError = "";

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onProcessTx() finalTrace.responses.size()=" << finalTrace.responses.size() << endl;
#endif
}

// Triggered when storage is updated in opcode processing
void FullTracer::onUpdateStorage (Context &ctx, const RomCommand &cmd)
{
    mpz_class regScalar;

    // The storage key is stored in C
    getRegFromCtx(ctx, reg_C, regScalar);
    string key;
    key = NormalizeToNFormat(regScalar.get_str(16), 64);

    // The storage value is stored in D
    getRegFromCtx(ctx, reg_D, regScalar);
    string value;
    value = NormalizeToNFormat(regScalar.get_str(16), 64);

    if (deltaStorage.find(depth) == deltaStorage.end())
    {
        cerr << "FullTracer::onUpdateStorage() did not found deltaStorage of depth=" << depth << endl;
        exitProcess();
    }

    deltaStorage[depth][key] = value;

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onUpdateStorage() depth=" << depth << " key=" << key << " value=" << value << endl;
#endif
}

// Triggered after processing a transaction
void FullTracer::onFinishTx (Context &ctx, const RomCommand &cmd)
{
    Response &response = finalTrace.responses[txCount];

    // Set from address
    mpz_class fromScalar;
    getVarFromCtx(ctx, true, "txSrcOriginAddr", fromScalar);
    response.call_trace.context.from = Add0xIfMissing(fromScalar.get_str(16));

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

    // Set return data
    mpz_class offsetScalar;
    getVarFromCtx(ctx, false, "retDataOffset", offsetScalar);
    mpz_class lengthScalar;
    getVarFromCtx(ctx, false, "retDataLength", lengthScalar);
    if (response.call_trace.context.to == "0x")
    {
        getCalldataFromStack(ctx, offsetScalar.get_ui(), lengthScalar.get_ui(), response.return_value);
    }
    else
    {
        getFromMemory(ctx, offsetScalar, lengthScalar, response.return_value);
    }

    //Set create address in case of deploy
    if (response.call_trace.context.to == "0x") {
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

        // set refunded gas
        response.gas_refunded = lastOpcode.refund; // gas_refund

        // Set gas price of last opcode
        if (info.size() >= 2)
        {
            Opcode beforeLastOpcode = info[info.size() - 2];
            lastOpcode.gas_cost = beforeLastOpcode.remaining_gas - lastOpcode.remaining_gas;
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
        finalTrace.responses[finalTrace.responses.size() - 1].call_trace.steps = call_trace;
        if (finalTrace.responses[finalTrace.responses.size() - 1].error == "")
        {
            finalTrace.responses[finalTrace.responses.size() - 1].error = lastOpcode.error;
        }
    }

    // Clean aux array for next iteration
    call_trace.clear();
    execution_trace.clear();

    // Append to response logs
    unordered_map<uint64_t, Log>::const_iterator it;
    uint64_t context = finalTrace.responses.size();
    if (logs.find(context) != logs.end())
    {
        for (it=logs[context].begin(); it!=logs[context].end(); it++)
        {
            Log log = it->second;
            finalTrace.responses[finalTrace.responses.size() - 1].logs.push_back(log);
        }
    }

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
    getRegFromCtx(ctx, cmd.params[1]->reg, auxScalar);
    finalTrace.batchHash = Add0xIfMissing(auxScalar.get_str(16));

    // Old state root
    getVarFromCtx(ctx, true, "oldStateRoot", auxScalar);
    finalTrace.old_state_root = Add0xIfMissing(auxScalar.get_str(16));

    // Global hash
    getVarFromCtx(ctx, true, "newAccInputHash", auxScalar);
    finalTrace.newAccInputHash = Add0xIfMissing(auxScalar.get_str(16));

    // Number of batch
    getVarFromCtx(ctx, true, "oldNumBatch", auxScalar);
    finalTrace.numBatch = auxScalar.get_ui() + 1;

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

    // New acc input hash
    getVarFromCtx(ctx, true, "newAccInputHash", auxScalar);
    finalTrace.new_acc_input_hash = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // New local exit root
    getVarFromCtx(ctx, true, "newLocalExitRoot", auxScalar);
    finalTrace.new_local_exit_root = NormalizeTo0xNFormat(auxScalar.get_str(16), 64);

    // New batch number
    getVarFromCtx(ctx, true, "newNumBatch", auxScalar);
    finalTrace.new_batch_num = auxScalar.get_ui();

#ifdef LOG_FULL_TRACER
    cout << "FullTracer::onFinishBatch() new_state_root=" << finalTrace.new_state_root << endl;
#endif
}

void FullTracer::onOpcode (Context &ctx, const RomCommand &cmd)
{
    Opcode singleInfo;
    mpz_class auxScalar;

    if (ctx.proverRequest.input.bNoCounters)
    {
        info.push_back(singleInfo);
        return;
    }

    // Get opcode info
    uint8_t codeId;
    if ( (cmd.params.size() >= 1) &&
         (cmd.params[0]->params.size() >= 1) &&
         (cmd.params[0]->params[0]->op == op_number) )
    {
        codeId = cmd.params[0]->params[0]->num.get_ui();
    }
    else if ( (cmd.params.size() >= 1) &&
         (cmd.params[0]->params.size() >= 1) &&
         (cmd.params[0]->params[0]->op == op_getReg) )
    {
        getRegFromCtx(ctx, cmd.params[0]->params[0]->reg, auxScalar);
        codeId = auxScalar.get_ui();
    }
    else
    {
        cerr << "Error: FullTracer::onOpcode() got invalid cmd.params" << endl;
        exitProcess();
        exit(-1);
    }

    // If the codeId does not exist, fallback to 0xfe = invalid code id
    if (opcodeName.find(codeId) == opcodeName.end())
    {
        codeId = 0xfe;
    }

    // Opcode = name (except "op")
    string opcode = opcodeName[codeId]+2;

    // store memory
    uint64_t offsetCtx = fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    uint64_t addrMem = 0;
    addrMem += offsetCtx;
    addrMem += 0x30000;

    string finalMemory;
    if (ctx.proverRequest.generateCallTraces())
    {
        uint64_t lengthMemOffset = findOffsetLabel(ctx, "memLength");
        uint64_t lenMemValueFinal = 0;
        if (ctx.mem.find(offsetCtx + lengthMemOffset) != ctx.mem.end())
        {
            Fea lenMemValue = ctx.mem[offsetCtx + lengthMemOffset];
            fea2scalar(ctx.fr, auxScalar, lenMemValue.fe0, lenMemValue.fe1, lenMemValue.fe2, lenMemValue.fe3, lenMemValue.fe4, lenMemValue.fe5, lenMemValue.fe6, lenMemValue.fe7);
            lenMemValueFinal = ceil(double(auxScalar.get_ui())/32);
        }

        for (uint64_t i = 0; i < lenMemValueFinal; i++)
        {
            if (ctx.mem.find(addrMem + i) == ctx.mem.end())
            {
                finalMemory += "0000000000000000000000000000000000000000000000000000000000000000";
                continue;
            }
            Fea memValue = ctx.mem[addrMem + i];
            fea2scalar(ctx.fr, auxScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7);
            string hexString = auxScalar.get_str(16);
            //if ((hexString.size() % 2) > 0) hexString = "0" + hexString;
            hexString = NormalizeToNFormat(hexString, 64);
            finalMemory += hexString;
        }
    }

    // store stack
    uint64_t addr = 0;
    addr += offsetCtx;
    addr += 0x20000;

    vector<mpz_class> finalStack;
    if (ctx.proverRequest.generateCallTraces())
    {
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
            finalStack.push_back(stackScalar);
        }
    }

    // add info opcodes
    getVarFromCtx(ctx, true, "depth", auxScalar);
    depth = auxScalar.get_ui();
    singleInfo.depth = depth + 1;
    singleInfo.pc = fr.toU64(ctx.pols.PC[*ctx.pStep]);
    singleInfo.remaining_gas = fr.toU64(ctx.pols.GAS[*ctx.pStep]);
    gettimeofday(&singleInfo.startTime, NULL);
    
    //cout << "singleInfo.remaining_gas=" << singleInfo.remaining_gas << endl;
    if (info.size() > 0)
    {
        // The gas cost of the opcode is gas before - gas after processing the opcode
        info[info.size() - 1].gas_cost = int64_t(info[info.size() - 1].remaining_gas) - fr.toS64(ctx.pols.GAS[*ctx.pStep]);
        //cout << "info[info.size() - 1].gas_cost=" << info[info.size() - 1].gas_cost << endl;

        // If negative gasCost means gas has been added from a deeper context, we should recalculate
        if (info[info.size() - 1].gas_cost < 0)
        {
            if (info.size() > 1)
            {
                Opcode beforePrevTrace = info[info.size() - 2];
                info[info.size() - 1].gas_cost = beforePrevTrace.remaining_gas - info[info.size() - 1].remaining_gas;
            }
            else
            {
                cout << "Warning: FullTracer::onOpcode() could not calculate prevTrace.gas_cost" << endl;
                info[info.size() - 1].gas_cost = 0;
            }
        }

        info[info.size() - 1].duration = TimeDiff(info[info.size() - 1].startTime, singleInfo.startTime);
    }

    singleInfo.opcode = opcode;

    if (ctx.proverRequest.generateCallTraces())
    {
        getVarFromCtx(ctx, false, "gasRefund", auxScalar);
        singleInfo.refund = auxScalar.get_ui();

        singleInfo.op = codeId;
    }
    singleInfo.error = "";

    if (ctx.proverRequest.generateCallTraces())
    {
        fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
        singleInfo.state_root = Add0xIfMissing(auxScalar.get_str(16));
    }

    //Add contract info
    if (ctx.proverRequest.generateCallTraces())
    {
        getVarFromCtx(ctx, false, "txDestAddr", auxScalar);
        singleInfo.contract.address = auxScalar.get_str(16);

        getVarFromCtx(ctx, false, "txSrcAddr", auxScalar);
        singleInfo.contract.caller = auxScalar.get_str(16);

        getVarFromCtx(ctx, false, "txValue", auxScalar);
        singleInfo.contract.value = auxScalar;

        getCalldataFromStack(ctx, 0, 0, singleInfo.contract.data);
    }

    singleInfo.contract.gas = txGAS[depth];

    if (ctx.proverRequest.generateCallTraces())
    {
        singleInfo.storage = deltaStorage[depth];

        // Round up to next multiple of 32
        getVarFromCtx(ctx, false, "memLength", auxScalar);
        singleInfo.memory_size = (auxScalar.get_ui()/32)*32;
    }

    info.push_back(singleInfo);
    fullStack.push_back(finalStack);

    // build trace
    uint64_t index = fullStack.size();

    if (index > 1)
    {
        Opcode singleCallTrace = info[index - 2];
        /*if (ctx.proverRequest.generateCallTraces())
        {
            singleCallTrace.stack = finalStack;
            singleCallTrace.memory = finalMemory;
        }*/

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
            if (ctx.proverRequest.generateCallTraces())
            {
                singleInfo.contract.gas = txGAS[depth];
            }
        }
    }

    if (opIncContext.find(singleInfo.opcode) != opIncContext.end())
    {
        unordered_map<string,string> auxMap;
        deltaStorage[depth+1] = auxMap;
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
    double init = addrMem + double(offset.get_ui())/32;
    double end = addrMem + double(offset.get_ui() + length.get_ui())/32;
    uint64_t initCeil = ceil(init);
    uint64_t initFloor = floor(init);
    uint64_t endFloor = floor(end);

    if (init != double(initCeil))
    {
        mpz_class memScalarStart = 0;
        std::unordered_map<uint64_t, Fea>::iterator it = ctx.mem.find(initFloor);
        if (it != ctx.mem.end())
        {
            fea2scalar(fr, memScalarStart, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7);
        }
        string hexStringStart = NormalizeToNFormat(memScalarStart.get_str(16), 64);
        uint64_t bytesToSkip = (init - double(initFloor)) * 32;
        result += hexStringStart.substr(bytesToSkip*2, 64);
    }

    for (uint64_t i=initCeil; i<endFloor; i++)
    {
        mpz_class memScalar = 0;
        if (ctx.mem.find(i) != ctx.mem.end())
        {
            Fea memValue = ctx.mem[i];
            fea2scalar(ctx.fr, memScalar, memValue.fe0, memValue.fe1, memValue.fe2, memValue.fe3, memValue.fe4, memValue.fe5, memValue.fe6, memValue.fe7);
        }
        result += NormalizeToNFormat(memScalar.get_str(16), 64);
    }

    if (end != double(endFloor))
    {
        mpz_class memScalarEnd = 0;
        std::unordered_map<uint64_t, Fea>::iterator it = ctx.mem.find(endFloor);
        if (it != ctx.mem.end())
        {
            fea2scalar(fr, memScalarEnd, it->second.fe0, it->second.fe1, it->second.fe2, it->second.fe3, it->second.fe4, it->second.fe5, it->second.fe6, it->second.fe7);
        }
        string hexStringEnd = NormalizeToNFormat(memScalarEnd.get_str(16), 64);
        uint64_t bytesToRetrieve = (end - double(endFloor)) * 32;
        result += hexStringEnd.substr(0, bytesToRetrieve*2);
    }
}

// Get a global or context variable
void FullTracer::getVarFromCtx (Context &ctx, bool global, const char * pVarLabel, mpz_class &result)
{
    uint64_t offsetCtx = global ? 0 : fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;
    uint64_t offsetRelative = findOffsetLabel(ctx, pVarLabel);
    uint64_t addressMem = offsetCtx + offsetRelative;
    unordered_map< uint64_t, Fea >::iterator memIterator;
    memIterator = ctx.mem.find(addressMem);
    if (memIterator == ctx.mem.end())
    {
        //cout << "FullTracer::getVarFromCtx() could not find in ctx.mem address=" << pVarLabel << "=" << offsetRelative << endl; TODO: Double check this is as designed?
        result = 0;
    }
    else
    {
        Fea value = memIterator->second;
        fea2scalar(ctx.fr, result, value.fe0, value.fe1, value.fe2, value.fe3, value.fe4, value.fe5, value.fe6, value.fe7);
    }
}

//Get the stored calldata in the stack
void FullTracer::getCalldataFromStack (Context &ctx, uint64_t offset, uint64_t length, string &result)
{
    uint64_t addr = 0x20000 + 1024 + fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    result = "0x";
    unordered_map< uint64_t, Fea >::iterator memIterator;
    for (uint64_t i = addr + offset; i < 0x30000 + fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000; i++)
    {
        memIterator = ctx.mem.find(i);
        if (memIterator == ctx.mem.end())
        {
            break;
        }
        Fea memVal = memIterator->second;
        mpz_class auxScalar;
        fea2scalar(ctx.fr, auxScalar, memVal.fe0, memVal.fe1, memVal.fe2, memVal.fe3, memVal.fe4, memVal.fe5, memVal.fe6, memVal.fe7);
        result += NormalizeToNFormat(auxScalar.get_str(16), 64);
    }
    
    if (length > 0)
    {
        result = result.substr(0, 2 + length*2);
    }
    /*if (result.size() <= 2)
    {
        result = "0x0";
    }*/
}

// Get the value of a reg (A, B, C, D, E...)
void FullTracer::getRegFromCtx (Context &ctx, tReg reg, mpz_class &result)
{
    RomCommand cmd;
    cmd.reg = reg;
    CommandResult cr;
    eval_getReg(ctx, cmd, cr);
    cr2scalar(ctx.fr, cr, result);
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

using namespace rlp;

// Returns a transaction hash from transaction params
void FullTracer::getTransactionHash(string &to, mpz_class value, uint64_t nonce, uint64_t gasLimit,
                        mpz_class gasPrice, string &data, mpz_class &r, mpz_class &s, uint64_t v, string &txHash, string &rlpTx)
{
#ifdef LOG_TX_HASH
    cout << "FullTracer::getTransactionHash() to=" << to << " value=" << value << " nonce=" << nonce << " gasLimit=" << gasLimit << " gasPrice=" << gasPrice << " data=" << data << " r=" << r.get_str(16) << " s=" << s.get_str(16) << " v=" << v << endl;
#endif

    string raw;

    encode(raw, nonce);
    encode(raw, gasPrice);
    encode(raw, gasLimit);
    if (!encodeHexData(raw, to)) {
        cout << "ERROR encoding to" << endl;
    }
    encode(raw, value);

    if (!encodeHexData(raw, data)) {
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