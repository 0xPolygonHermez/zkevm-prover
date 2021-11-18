#include <iostream>
#include "eval_command.hpp"
#include "scalar.hpp"
#include "pols.hpp"

RawFr::Element eval_number(Context &ctx, RomCommand &cmd);
RawFr::Element eval_getReg(Context &ctx, RomCommand &cmd);
RawFr::Element eval_declareVar(Context &ctx, RomCommand &cmd);
RawFr::Element eval_setVar(Context &ctx, RomCommand &cmd);
RawFr::Element eval_getVar(Context &ctx, RomCommand &cmd);
RawFr::Element eval_add(Context &ctx, RomCommand &cmd);
RawFr::Element eval_sub(Context &ctx, RomCommand &cmd);
RawFr::Element eval_neg(Context &ctx, RomCommand &cmd);
RawFr::Element eval_mul(Context &ctx, RomCommand &cmd);
RawFr::Element eval_div(Context &ctx, RomCommand &cmd);
RawFr::Element eval_mod(Context &ctx, RomCommand &cmd);

RawFr::Element evalCommand(Context &ctx, RomCommand &cmd) {
    if (cmd.op=="number") {
        return eval_number(ctx, cmd); // TODO: return a big number, an mpz, >253bits, here and in all evalXxx() to unify
    } else if (cmd.op=="declareVar") {
        return eval_declareVar(ctx, cmd);
    } else if (cmd.op=="setVar") {
        return eval_setVar(ctx, cmd);
    } else if (cmd.op=="getVar") {
        return eval_getVar(ctx, cmd);
    } else if (cmd.op=="getReg") {
        return eval_getReg(ctx, cmd);
    } else if (cmd.op=="functionCall") {
        //return eval_functionCall(ctx, cmd);
    } else if (cmd.op=="add") {
        return eval_add(ctx, cmd);
    } else if (cmd.op=="sub") {
        return eval_sub(ctx, cmd);
    } else if (cmd.op=="neg") {
        return eval_neg(ctx, cmd);
    } else if (cmd.op=="mul") {
        //return eval_mul(ctx, cmd);
    } else if (cmd.op=="div") {
        return eval_div(ctx, cmd);
    } else if (cmd.op=="mod") {
        return eval_mod(ctx, cmd);
    }
    cerr << "Error: evalCommand() found invalid operation: " << cmd.op << endl;
    exit(-1);
}

RawFr::Element eval_number(Context &ctx, RomCommand &cmd) {
    RawFr::Element num;
    ctx.pFr->fromUI(num,cmd.num); // TODO: Check existence and type of num element
    return num;
}

/*************/
/* Variables */
/*************/

/* If defined, logs variable declaration, get and set actions */
#define LOG_VARIABLES

/* Declares a new variable, and fails if it already exists */
RawFr::Element eval_declareVar(Context &ctx, RomCommand &cmd)
{
    // Check the variable name
    if (cmd.varName == "") {
        cerr << "Error: eval_declareVar() Variable name not found" << endl;
        exit(-1);  
    }

    // Check that this variable does not exists
    if ( ctx.vars.find(cmd.varName) != ctx.vars.end() ) {
        cerr << "Error: eval_declareVar() Variable already declared: " << cmd.varName << endl;
        exit(-1);
    }

    // Create the new variable with a zero value
    ctx.vars[cmd.varName] = ctx.pFr->zero(); // TODO: Should it be Scalar.e(0)?
#ifdef LOG_VARIABLES
    cout << "Declare variable: " << cmd.varName << endl;
#endif
    return ctx.vars[cmd.varName];
}

/* Gets the value of the variable, and fails if it does not exist */
RawFr::Element eval_getVar(Context &ctx, RomCommand &cmd)
{
    // Check the variable name
    if (cmd.varName == "") {
        cerr << "Error: eval_getVar() Variable name not found" << endl;
        exit(-1);  
    }

    // Check that this variable exists
    if ( ctx.vars.find(cmd.varName) == ctx.vars.end() ) {
        cerr << "Error: eval_getVar() Undefined variable: " << cmd. varName << endl;
        exit(-1);
    }

#ifdef LOG_VARIABLES
    cout << "Get variable: " << cmd.varName << endl;
#endif
    return ctx.vars[cmd.varName];
}

string eval_left(Context &ctx, RomCommand &cmd);

/* Sets variable to value, and fails if it does not exist */
RawFr::Element eval_setVar(Context &ctx, RomCommand &cmd)
{
    // Check that tag contains a values array
    if (cmd.values.size()==0) {
        cerr << "Error: eval_setVar() could not find array values in setVar command" << endl;
        exit(-1);
    }
    

    // Get varName from the first element in values
    string varName = eval_left(ctx,*cmd.values[0]);

    // Check that this variable exists
    if ( ctx.vars.find(varName) == ctx.vars.end() ) {
        cerr << "Error: eval_setVar() Undefined variable: " << varName << endl;
        exit(-1);
    }

    ctx.vars[varName] = evalCommand(ctx, *cmd.values[1]);
#ifdef LOG_VARIABLES
    cout << "Set variable: " << varName << endl;
#endif
    return ctx.vars[varName];
}

string eval_left(Context &ctx, RomCommand &cmd)
{
    if (cmd.op == "declareVar") {
        eval_declareVar(ctx, cmd);
        return cmd.varName;
    } else if (cmd.op == "getVar") {
        return cmd.varName;
    }
    cerr << "Error: invalid left expression, op: " << cmd.op << "ln: " << ctx.ln << endl;
    exit(-1);
}





RawFr::Element eval_getReg(Context &ctx, RomCommand &cmd) {
    if (cmd.regName=="A") { // TODO: Consider using a string local variable to avoid searching every time
        //return fea2bn(ctx.pFr,ctx.pols[]);
        mpz_t result;
        mpz_init(result);
        fea2bn(ctx, result, pols(A0)[ctx.step], pols(A1)[ctx.step], pols(A2)[ctx.step], pols(A3)[ctx.step]);
        RawFr::Element feResult;
        ctx.pFr->fromMpz(feResult, result);
        mpz_clear(result);
        return feResult;
        //return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="B") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="C") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="D") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="E") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="SR") {
        return ctx.pFr->zero();//return pols[SR][ctx.step];
    } else if (cmd.regName=="CTX") {
        return ctx.pFr->zero();//return pols[CTX][ctx.step];
    } else if (cmd.regName=="SP") {
        return ctx.pFr->zero();//return pols[SP][ctx.step];
    } else if (cmd.regName=="PC") {
        return ctx.pFr->zero();//return pols[PC][ctx.step];
    } else if (cmd.regName=="MAXMEM") {
        return ctx.pFr->zero();//return pols[MAXMEM][ctx.step];
    } else if (cmd.regName=="GAS") {
        return ctx.pFr->zero();//return pols[GAS][ctx.step];
    } else if (cmd.regName=="zkPC") {
        return ctx.pFr->zero();//pols(zkPC,ctx.step);//pols[zkPC][ctx.step];
    }
    cerr << "Error: eval_getReg() Invalid register: " << cmd.regName << ": " << ctx.ln << endl;
    exit(-1);
}
/*
function eval_getReg(ctx, tag) {
    if (tag.regName == "A") {
        return fea2bn(ctx.Fr, ctx.A);
    } else if (tag.regName == "B") {
        return fea2bn(ctx.Fr, ctx.B);
    } else if (tag.regName == "C") {
        return fea2bn(ctx.Fr, ctx.C);
    } else if (tag.regName == "D") {
        return fea2bn(ctx.Fr, ctx.D);
    } else if (tag.regName == "E") {
        return fea2bn(ctx.Fr, ctx.E);
    } else if (tag.regName == "SR") {
        return ctx.SR;
    } else if (tag.regName == "CTX") {
        return ctx.CTX;
    } else if (tag.regName == "SP") {
        return ctx.SP;
    } else if (tag.regName == "PC") {
        return ctx.PC;
    } else if (tag.regName == "MAXMEM") {
        return ctx.MAXMEM;
    } else if (tag.regName == "GAS") {
        return ctx.GAS;
    } else if (tag.regName == "zkPC") {
        return ctx.zkPC;
    } else {
        throw new Error(`Invalid register ${tag.regName}:  ${ctx.ln}`);
    }
}
*/
RawFr::Element eval_add(Context &ctx, RomCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    ctx.pFr->add(r,a,b);
    return r;
}
/*
function eval_add(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Scalar.add(a,b);
}*/
RawFr::Element eval_sub(Context &ctx, RomCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    ctx.pFr->sub(r,a,b);
    return r;
}
/*
function eval_sub(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Scalar.sub(a,b);
}*/
RawFr::Element eval_neg(Context &ctx, RomCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element r;
    ctx.pFr->neg(r,a);
    return r;
}
/*
function eval_neg(ctx, tag) {
    const a = evalCommand(ctx, values[0]);
    return Scalar.neg(a);
}*/

/*
function eval_mul(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Sacalar.and(Scalar.mul(a,b), Mask256);
}*/
RawFr::Element eval_div(Context &ctx, RomCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    ctx.pFr->div(r,a,b);
    return r;
}
/*
function eval_div(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Sacalar.div(a,b);
}*/
RawFr::Element eval_mod(Context &ctx, RomCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    //ctx.pFr->mod(r,a,b); // TODO: Migrate.  This method does not exist in C.
    return r;
}
/*
function eval_mod(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Sacalar.mod(a,b);
}
*/
RawFr::Element eval_functionCall(Context &ctx, RomCommand &cmd) {
    if (cmd.funcName == "getGlobalHash") {
        //return eval_getGlobalHash(ctx, tag);
    } else if (cmd.funcName == "getOldStateRoot") {
        //return eval_getOldStateRoot(ctx, tag);
    } else if (cmd.funcName == "getNewStateRoot") {
        //return eval_getNewStateRoot(ctx, tag);
    } else if (cmd.funcName == "getNTxs") {
        //return eval_getNTxs(ctx, tag);
    } else if (cmd.funcName == "getRawTx") {
        //return eval_getRawTx(ctx, tag);
    } else if (cmd.funcName == "getTxSigR") {
        //return eval_getTxSigR(ctx, tag);
    } else if (cmd.funcName == "getTxSigS") {
        //return eval_getTxSigS(ctx, tag);
    } else if (cmd.funcName == "getTxSigV") {
        //return eval_getTxSigV(ctx, tag);
    } else if (cmd.funcName == "getSequencerAddr") {
        //return eval_getSequencerAddr(ctx, tag);
    } else if (cmd.funcName == "getChainId") {
        //return eval_getChainId(ctx, tag);
    }
    cerr << "Error: eval_functionCall() function not defined: " << cmd.funcName << " line: " << ctx.ln << endl;
    exit(-1); 
}

/*
function eval_functionCall(ctx, tag) {
    if (tag.funcName == "getGlobalHash") {
        return eval_getGlobalHash(ctx, tag);
    } else if (tag.funcName == "getOldStateRoot") {
        return eval_getOldStateRoot(ctx, tag);
    } else if (tag.funcName == "getNewStateRoot") {
        return eval_getNewStateRoot(ctx, tag);
    } else if (tag.funcName == "getNTxs") {
        return eval_getNTxs(ctx, tag);
    } else if (tag.funcName == "getRawTx") {
        return eval_getRawTx(ctx, tag);
    } else if (tag.funcName == "getTxSigR") {
        return eval_getTxSigR(ctx, tag);
    } else if (tag.funcName == "getTxSigS") {
        return eval_getTxSigS(ctx, tag);
    } else if (tag.funcName == "getTxSigV") {
        return eval_getTxSigV(ctx, tag);
    } else if (tag.funcName == "getSequencerAddr") {
        return eval_getSequencerAddr(ctx, tag);
    } else if (tag.funcName == "getChainId") {
        return eval_getChainId(ctx, tag);
    } else {
        throw new Error(`function not defined ${tag.funcName}:  ${ctx.ln}`);
    }
}
*/
/*

function eval_getGlobalHash(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return bn2bna(ctx.Fr, Scalar.e(ctx.globalHash));
}

function eval_getSequencerAddr(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return bn2bna(ctx.Fr, Scalar.e(ctx.input.sequencerAddr));
}

function eval_getChainId(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.input.chainId), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getOldStateRoot(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.input.oldStateRoot), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getNewStateRoot(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.input.newStateRoot), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getNTxs(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.pTxs.length), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getRawTx(ctx, tag) {
    if (tag.params.length != 3) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    const offset = Number(evalCommand(ctx,tag.params[1]));
    const len = Number(evalCommand(ctx,tag.params[2]));
    let d = "0x" +ctx.pTxs[txId].signData.slice(2+offset*2, 2+offset*2 + len*2);
    if (d.length == 2) d = d+'0';
    return bn2bna(ctx.Fr, Scalar.e(d));
}

function eval_getTxSigR(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    return bn2bna(ctx.Fr, Scalar.e(ctx.pTxs[txId].signature.r));
}

function eval_getTxSigS(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    return bn2bna(ctx.Fr, Scalar.e(ctx.pTxs[txId].signature.s));
}

function eval_getTxSigV(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    return bn2bna(ctx.Fr, Scalar.e(ctx.pTxs[txId].signature.v));
}
*/

