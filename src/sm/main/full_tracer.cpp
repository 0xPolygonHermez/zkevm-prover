#include <iostream>
#include <sys/time.h>
#include <set>
#include "full_tracer.hpp"
#include "goldilocks/goldilocks_base_field.hpp"
#include "context.hpp"
#include "scalar.hpp"
#include "opcode_name.hpp"
#include "zkassert.hpp"

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
    string errorName = cmd.params[1]->varName;
    info[info.size()-1].error = errorName;
    if (depth == 0)
    {
        finalTrace.txs[txCount].context.error = errorName;
    }
}

void FullTracer::onStoreLog (Context &ctx, const RomCommand &cmd)
{
    mpz_class indexLogScalar;
    getRegFromCtx(ctx, cmd.params[0]->regName, indexLogScalar);
    uint64_t indexLog = indexLogScalar.get_ui();
    uint64_t isTopic = cmd.params[1]->num;
    mpz_class data;
    getRegFromCtx(ctx, cmd.params[2]->regName, data);

    if (finalTrace.txs[txCount].context.logs.size() < (indexLog+1))
    {
        Log log;
        finalTrace.txs[txCount].context.logs.push_back(log);
    }

    if (isTopic)
    {
        finalTrace.txs[txCount].context.logs[indexLog].topics.push_back(data.get_str(16));
    }
    else
    {
        finalTrace.txs[txCount].context.logs[indexLog].data.push_back(data.get_str(16));
    }
}

// Triggered at the very beginning of transaction process
void FullTracer::onProcessTx (Context &ctx, const RomCommand &cmd)
{
    TxTrace tx;

    string auxString;
    mpz_class auxScalar;
    
    auxString = "txSrcAddr";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    tx.context.from = Add0xIfMissing(auxScalar.get_str(16));
    
    auxString = "txDestAddr";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    tx.context.to = Add0xIfMissing(auxScalar.get_str(16));

    tx.context.type = (tx.to == "0x00") ? "CREATE" : "CALL"; // TODO: This is always "CREATE", right?

    getCalldataFromStack(ctx, tx.context.input);
    
    auxString = "txGas";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    tx.context.gas = auxScalar.get_ui(); // TODO: Using u64 instead of string (JS)
    
    auxString = "txValue";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    tx.context.value = auxScalar.get_str(16);

    //tx.context.output = ""; // No code needed, since this is the default value
    
    auxString = "txNonce";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    tx.context.nonce = auxScalar.get_ui();
    
    auxString = "txGasPrice";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    tx.context.gasPrice = auxScalar.get_str(16);
    
    auxString = "txChainId";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    tx.context.chainId = auxScalar.get_ui();

    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    tx.context.oldStateRoot = Add0xIfMissing(auxScalar.get_str(16));

    // Create current tx object
    finalTrace.txs.push_back(tx);
    txTime = getCurrentTime();

    // Reset values
    depth = 1;
    deltaStorage.clear();
    txGAS[depth] = tx.context.gas;
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
}

void FullTracer::onFinishTx (Context &ctx, const RomCommand &cmd)
{
    TxTraceContext txContext = finalTrace.txs[txCount].context;

    // Set tx runtime
    txContext.time = getCurrentTime() - txTime;

    //Set consumed tx gas
    txContext.gasUsed = txContext.gas - fr.toU64(ctx.pols.GAS[*ctx.pStep]); // Using u64 in C instead of string in JS

    //Set new State Root
    mpz_class auxScalar;
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    txContext.newStateRoot = Add0xIfMissing(auxScalar.get_str(16));

    //If processed opcodes
    if (info.size() > 0)
    {
        Opcode lastOpcode = info[info.size() - 1];

        // Set gas price of last opcode
        if (info.size() > 2)
        {
            Opcode beforeLastOpcode = info[info.size() - 2];
            lastOpcode.gasCost = beforeLastOpcode.gas - lastOpcode.gas;
        }

        //Add last opcode
        trace.push_back(lastOpcode);
        if (trace.size() < info.size())
        {
            trace.erase(trace.begin()); // trace.shift in JS
        }

        //Append processed opcodes to the transaction object
        finalTrace.txs[finalTrace.txs.size() - 1].steps = trace; // TODO: Append? This is replacing the vector...
    }
    
    // Clean aux array for next iteration
    trace.clear();

    /*
    if (!fs.existsSync(this.folderLogs)) {
        fs.mkdirSync(this.folderLogs)
    }
    fs.writeFileSync(`${this.pathLogFile}_${this.txCount}.json`, JSON.stringify(this.finalTrace.txs[this.txCount], null, 2));
    */

    // Increase transaction count
    txCount++;
}

void FullTracer::onStartBatch (Context &ctx, const RomCommand &cmd)
{
    if (finalTrace.bInitialized) return;

    mpz_class auxScalar;
    string auxString;
    
    getRegFromCtx(ctx, cmd.params[1]->regName, auxScalar);
    finalTrace.batchHash = Add0xIfMissing(auxScalar.get_str(16));

    auxString = "oldStateRoot";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    finalTrace.oldStateRoot = Add0xIfMissing(auxScalar.get_str(16));

    auxString = "globalHash";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    finalTrace.globalHash = Add0xIfMissing(auxScalar.get_str(16));

    auxString = "numBatch";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    finalTrace.numBatch = auxScalar.get_ui();

    auxString = "timestamp";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    finalTrace.timestamp = auxScalar.get_ui();

    auxString = "sequencerAddr";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    finalTrace.sequencerAddr = Add0xIfMissing(auxScalar.get_str(16));

    finalTrace.txs.clear();

    finalTrace.bInitialized = true;
}

void FullTracer::onFinishBatch (Context &ctx, const RomCommand &cmd)
{
    // Create ouput files and dirs
    //this.exportTrace();
}

void FullTracer::onOpcode (Context &ctx, const RomCommand &cmd)
{
    Opcode singleTrace;
    Opcode singleInfo;

    //Get opcode info
    mpz_class auxScalar;
    fea2scalar(ctx.fr, auxScalar, ctx.pols.B0[*ctx.pStep], ctx.pols.B1[*ctx.pStep], ctx.pols.B2[*ctx.pStep], ctx.pols.B3[*ctx.pStep], ctx.pols.B4[*ctx.pStep], ctx.pols.B5[*ctx.pStep], ctx.pols.B6[*ctx.pStep], ctx.pols.B7[*ctx.pStep] );
    zkassert(auxScalar<256);
    uint8_t codeId = auxScalar.get_ui();
    string opcode = opcodeName[codeId]+2;

    // store memory
    uint64_t offsetCtx = fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    uint64_t addrMem = 0;
    addrMem += offsetCtx;
    addrMem += 0x30000;

    vector<string> finalMemory;
    string auxString = "memLength";
    uint64_t lengthMemOffset = findOffsetLabel(ctx, auxString);
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
        if ((hexString.size() % 2) > 0) hexString = "0" + hexString;
        hexString = NormalizeToNFormat(hexString, 64);
        finalMemory.push_back(hexString);
    }

    // store stack
    uint64_t addr = 0;
    addr += offsetCtx;
    addr += 0x20000;

    vector<string> finalStack;
    uint16_t sp = fr.toU64(ctx.pols.SP[*ctx.pStep]);
    for (uint16_t i=0; i<sp; i++)
    {
        if (ctx.mem.find(addr + i) == ctx.mem.end()) continue;
        Fea stack = ctx.mem[addr + i];
        mpz_class stackScalar;
        fea2scalar(ctx.fr, stackScalar, stack.fe0, stack.fe1, stack.fe2, stack.fe3, stack.fe4, stack.fe5, stack.fe6, stack.fe7 );
        string hexString = stackScalar.get_str(16);
        if ((hexString.size() % 2) > 0) hexString = "0" + hexString;
        hexString = "0x" + hexString;
        finalStack.push_back(hexString);
    }

    // add info opcodes
    fea2scalar(ctx.fr, auxScalar, ctx.pols.SR0[*ctx.pStep], ctx.pols.SR1[*ctx.pStep], ctx.pols.SR2[*ctx.pStep], ctx.pols.SR3[*ctx.pStep], ctx.pols.SR4[*ctx.pStep], ctx.pols.SR5[*ctx.pStep], ctx.pols.SR6[*ctx.pStep], ctx.pols.SR7[*ctx.pStep] );
    singleInfo.stateRoot = Add0xIfMissing(auxScalar.get_str(16));
    singleInfo.depth = depth;
    singleInfo.pc = fr.toU64(ctx.pols.PC[*ctx.pStep]);
    singleInfo.gas = fr.toU64(ctx.pols.GAS[*ctx.pStep]);
    if (info.size() > 0)
    {
        Opcode prevTrace = info[info.size() - 1];

        // The gas cost of the opcode is gas before - gas after processing the opcode
        int64_t gasCost = int64_t(prevTrace.gas) - int64_t(fr.toU64(ctx.pols.GAS[*ctx.pStep]));
        prevTrace.gasCost = gasCost;
        // If negative gasCost means gas has been added from a deeper context, we should recalculate
        if (prevTrace.gasCost < 0)
        {
            Opcode beforePrevTrace = info[info.size() - 2]; // TODO: protect agains -2
            prevTrace.gasCost = beforePrevTrace.gas - prevTrace.gas;
        }
    }

    singleInfo.opcode = opcode;
    
    auxString = "gasRefund";
    getVarFromCtx(ctx, true, auxString, auxScalar);
    singleInfo.refund = auxScalar.get_ui();
    
    singleInfo.op = codeId;

    // TODO: handle errors
    singleInfo.error = "";
    singleInfo.storage = deltaStorage[depth];

    info.push_back(singleInfo);
    fullStack.push_back(finalStack);

    // build trace
    uint64_t index = fullStack.size();

    if (index > 1)
    {
        singleTrace = info[index - 2];
        singleTrace.stack = finalStack;
        singleTrace.memory = finalMemory;
        trace.push_back(singleTrace);
    }

    //Add contract info

    auxString = "txDestAddr";
    getVarFromCtx(ctx, false, auxString, auxScalar);
    singleInfo.contract.address = Add0xIfMissing(auxScalar.get_str(16));

    auxString = "txSrcAddr";
    getVarFromCtx(ctx, false, auxString, auxScalar);
    singleInfo.contract.caller = Add0xIfMissing(auxScalar.get_str(16));

    auxString = "txValue";
    getVarFromCtx(ctx, false, auxString, auxScalar);
    singleInfo.contract.value = auxScalar.get_str(16);

    getCalldataFromStack(ctx, singleInfo.contract.input);

    singleInfo.contract.gas = txGAS[depth];

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
    //Check previous step
    if (info.size() >= 2)
    {
        Opcode prevStep = info[info.size() - 2]; 
        if (opIncContext.find(prevStep.opcode) != opIncContext.end())
        {
            //Set gasCall when depth has changed
            auxString = "gasCall";
            getVarFromCtx(ctx, false, auxString, auxScalar);
            txGAS[depth] = auxScalar.get_str();
            singleInfo.contract.gas = txGAS[depth];
        }
    }
}

// Get a global or context variable
void FullTracer::getVarFromCtx (Context &ctx, bool global, string &varLabel, mpz_class &result)
{
    uint64_t offsetCtx = global ? 0 : fr.toU64(ctx.pols.CTX[*ctx.pStep]) * 0x40000;
    uint64_t offsetRelative = findOffsetLabel(ctx, varLabel);
    uint64_t addressMem = offsetCtx + offsetRelative;
    if (ctx.mem.find(addressMem) == ctx.mem.end())
    {
        result = 0;
    }
    else
    {
        Fea value = ctx.mem[addressMem];
        fea2scalar(ctx.fr, result, value.fe0, value.fe1, value.fe2, value.fe3, value.fe4, value.fe5, value.fe6, value.fe7);
    }
}

//Get the stored calldata in the stack
void FullTracer::getCalldataFromStack (Context &ctx, string &result)
{
    uint64_t addr = 0x20000 + 1024 + fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000;
    result = "0x";
    //mpz_class num = 0; // TODO: What do we need num for?
    for (uint64_t i = addr; i < 0x30000 + fr.toU64(ctx.pols.CTX[*ctx.pStep])*0x40000; i++)
    {
        if (ctx.mem.find(i) == ctx.mem.end())
        {
            break;
        }
        Fea memVal = ctx.mem[i];
        mpz_class auxScalar;
        fea2scalar(ctx.fr, auxScalar, memVal.fe0, memVal.fe1, memVal.fe2, memVal.fe3, memVal.fe4, memVal.fe5, memVal.fe6, memVal.fe7);
        result += NormalizeToNFormat(auxScalar.get_str(16), 64);
        //num += auxScalar;
    }
    if (result.size() == 2)
    {
        result = "";
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

uint64_t FullTracer::findOffsetLabel (Context &ctx, string &label)
{
    // If label was used before, then return the cached value
    if (labels.find(label) != labels.end())
    {
        return labels[label];
    }

    for (uint64_t i = 0; i < ctx.rom.size; i++)
    {
        if (ctx.rom.line[i].offsetLabel == label)
        {
            labels[label] = ctx.rom.line[i].offset;
            return ctx.rom.line[i].offset;
        }
    }

    return 0;
}

uint64_t FullTracer::getCurrentTime (void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*1000000 + tv.tv_usec;
}