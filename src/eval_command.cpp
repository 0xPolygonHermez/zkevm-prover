#include <iostream>
#include "eval_command.hpp"
#include "scalar.hpp"
#include "pols.hpp"


void eval_number(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getReg(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_declareVar(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_setVar(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getVar(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_add(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_sub(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_neg(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_mul(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_div(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_mod(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_functionCall(Context &ctx, RomCommand &cmd, CommandResult &cr);


void evalCommand(Context &ctx, RomCommand &cmd, CommandResult &cr) {
    if (cmd.op=="number") {
        eval_number(ctx, cmd, cr); // TODO: return a big number, an mpz, >253bits, here and in all evalXxx() to unify
    } else if (cmd.op=="declareVar") {
        eval_declareVar(ctx, cmd, cr);
    } else if (cmd.op=="setVar") {
        eval_setVar(ctx, cmd, cr);
    } else if (cmd.op=="getVar") {
        eval_getVar(ctx, cmd, cr);
    } else if (cmd.op=="getReg") {
        eval_getReg(ctx, cmd, cr);
    } else if (cmd.op=="functionCall") {
        eval_functionCall(ctx, cmd, cr);
    } else if (cmd.op=="add") {
        eval_add(ctx, cmd, cr);
    } else if (cmd.op=="sub") {
        eval_sub(ctx, cmd, cr);
    } else if (cmd.op=="neg") {
        eval_neg(ctx, cmd, cr);
    } else if (cmd.op=="mul") {
        return eval_mul(ctx, cmd, cr);
    } else if (cmd.op=="div") {
        eval_div(ctx, cmd, cr);
    } else if (cmd.op=="mod") {
        eval_mod(ctx, cmd, cr);
    } else {
        cerr << "Error: evalCommand() found invalid operation: " << cmd.op << endl;
        exit(-1);
    }
}

void eval_number(Context &ctx, RomCommand &cmd, CommandResult &cr) {
    cr.type = crt_fe;
    ctx.pFr->fromUI(cr.fe, cmd.num);
}

/*************/
/* Variables */
/*************/

/* If defined, logs variable declaration, get and set actions */
#define LOG_VARIABLES

/* Declares a new variable, and fails if it already exists */
void eval_declareVar(Context &ctx, RomCommand &cmd, CommandResult &cr)
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
    cr.type = crt_fe;
    ctx.pFr->copy(cr.fe, ctx.vars[cmd.varName]);
}

/* Gets the value of the variable, and fails if it does not exist */
void eval_getVar(Context &ctx, RomCommand &cmd, CommandResult &cr)
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
    cr.type = crt_fe;
    ctx.pFr->copy(cr.fe, ctx.vars[cmd.varName]);
}

void eval_left(Context &ctx, RomCommand &cmd, CommandResult &cr);

/* Sets variable to value, and fails if it does not exist */
void eval_setVar(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check that tag contains a values array
    if (cmd.values.size()==0) {
        cerr << "Error: eval_setVar() could not find array values in setVar command" << endl;
        exit(-1);
    }
    

    // Get varName from the first element in values
    eval_left(ctx,*cmd.values[0], cr);
    if (cr.type != crt_string) {
        cerr << "Error: unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    string varName = cr.str;

    // Check that this variable exists
    if ( ctx.vars.find(varName) == ctx.vars.end() ) {
        cerr << "Error: eval_setVar() Undefined variable: " << varName << endl;
        exit(-1);
    }
    evalCommand(ctx, *cmd.values[1], cr);
    if (cr.type != crt_fe)
    {
        cerr << "Error: eval_setVar() got unexpected result type: " << cr.type << " of function: " << cmd.values[1]->varName << endl;
        exit(-1);
    }
    ctx.pFr->copy(ctx.vars[varName], cr.fe);
#ifdef LOG_VARIABLES
    cout << "Set variable: " << varName << endl;
#endif
    cr.type = crt_fe;
    ctx.pFr->copy(cr.fe, ctx.vars[cmd.varName]);
}

void eval_left(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    if (cmd.op == "declareVar") {
        eval_declareVar(ctx, cmd, cr);
        cr.type = crt_string;
        cr.str = cmd.varName;
        return;
    } else if (cmd.op == "getVar") {
        cr.type = crt_string;
        cr.str = cmd.varName;
        return;
    }
    cerr << "Error: invalid left expression, op: " << cmd.op << "ln: " << ctx.ln << endl;
    exit(-1);
}

void eval_getReg(Context &ctx, RomCommand &cmd, CommandResult &cr) {
    if (cmd.regName=="A") { // TODO: Consider using a string local variable to avoid searching every time
        cr.type = crt_scalar;
        fea2scalar(*(ctx.pFr), cr.scalar, pols(A0)[ctx.step], pols(A1)[ctx.step], pols(A2)[ctx.step], pols(A3)[ctx.step]);
    } else if (cmd.regName=="B") {
        cr.type = crt_scalar;
        fea2scalar(*(ctx.pFr), cr.scalar, pols(B0)[ctx.step], pols(B1)[ctx.step], pols(B2)[ctx.step], pols(B3)[ctx.step]);
    } else if (cmd.regName=="C") {
        cr.type = crt_scalar;
        fea2scalar(*(ctx.pFr), cr.scalar, pols(C0)[ctx.step], pols(C1)[ctx.step], pols(C2)[ctx.step], pols(C3)[ctx.step]);
    } else if (cmd.regName=="D") {
        cr.type = crt_scalar;
        fea2scalar(*(ctx.pFr), cr.scalar, pols(D0)[ctx.step], pols(D1)[ctx.step], pols(D2)[ctx.step], pols(D3)[ctx.step]);
    } else if (cmd.regName=="E") {
        cr.type = crt_scalar;
        fea2scalar(*(ctx.pFr), cr.scalar, pols(E0)[ctx.step], pols(E1)[ctx.step], pols(E2)[ctx.step], pols(E3)[ctx.step]);
    } else if (cmd.regName=="SR") {
        cr.type = crt_fe;
        return ctx.pFr->copy(cr.fe, pols(SR)[ctx.step]);
    } else if (cmd.regName=="CTX") {
        cr.type = crt_u32;
        cr.u32 = pols(CTX)[ctx.step];
    } else if (cmd.regName=="SP") {
        cr.type = crt_u16;
        cr.u16 = pols(SP)[ctx.step];
    } else if (cmd.regName=="PC") {
        cr.type = crt_u32;
        cr.u32 = pols(PC)[ctx.step];
    } else if (cmd.regName=="MAXMEM") {
        cr.type = crt_u32;
        cr.u32 = pols(MAXMEM)[ctx.step];
    } else if (cmd.regName=="GAS") {
        cr.type = crt_u64;
        cr.u64 = pols(CTX)[ctx.step];
    } else if (cmd.regName=="zkPC") {
        cr.type = crt_u32;
        cr.u32 = pols(zkPC)[ctx.step];
    } else {
        cerr << "Error: eval_getReg() Invalid register: " << cmd.regName << ": " << ctx.ln << endl;
        exit(-1);
    }
}

void eval_add(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    RawFr::Element a;
    evalCommand(ctx, *cmd.values[0], cr);
    ctx.pFr->copy(a, cr.fe);

    RawFr::Element b;
    evalCommand(ctx, *cmd.values[1], cr);
    ctx.pFr->copy(b, cr.fe);

    cr.type = crt_fe;
    ctx.pFr->add(cr.fe, a, b); // TODO: Should this be a scalar addition? return Scalar.add(a,b);
}

void eval_sub(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    RawFr::Element a;
    evalCommand(ctx, *cmd.values[0], cr);
    ctx.pFr->copy(a, cr.fe);

    RawFr::Element b;
    evalCommand(ctx, *cmd.values[1], cr);
    ctx.pFr->copy(b, cr.fe);

    cr.type = crt_fe;
    ctx.pFr->sub(cr.fe, a, b);
}

void eval_neg(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    RawFr::Element a;
    evalCommand(ctx, *cmd.values[0], cr);
    ctx.pFr->copy(a, cr.fe);

    cr.type = crt_fe;
    ctx.pFr->neg(cr.fe, a);
}

void eval_mul(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    RawFr::Element a;
    evalCommand(ctx, *cmd.values[0], cr);
    ctx.pFr->copy(a, cr.fe);

    RawFr::Element b;
    evalCommand(ctx, *cmd.values[1], cr);
    ctx.pFr->copy(b, cr.fe);

    cr.type = crt_fe;
    ctx.pFr->mul(cr.fe, a, b); // Sacalar.and(Scalar.mul(a,b), Mask256);
}

void eval_div(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    RawFr::Element a;
    evalCommand(ctx, *cmd.values[0], cr);
    ctx.pFr->copy(a, cr.fe);

    RawFr::Element b;
    evalCommand(ctx, *cmd.values[1], cr);
    ctx.pFr->copy(b, cr.fe);

    cr.type = crt_fe;
    ctx.pFr->div(cr.fe, a, b);
}

void eval_mod(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    RawFr::Element a;
    evalCommand(ctx, *cmd.values[0], cr);
    ctx.pFr->copy(a, cr.fe);

    RawFr::Element b;
    evalCommand(ctx, *cmd.values[1], cr);
    ctx.pFr->copy(b, cr.fe);

    RawFr::Element r;
    //ctx.pFr->mod(r,a,b); // TODO: Migrate.  This method does not exist in C.
}

void eval_getGlobalHash(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getOldStateRoot(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getNewStateRoot(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getNTxs(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getRawTx(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getTxSigR(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getTxSigS(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getTxSigV(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getSequencerAddr(Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getChainId(Context &ctx, RomCommand &cmd, CommandResult &cr);

void eval_functionCall(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    if (cmd.funcName == "getGlobalHash") {
        eval_getGlobalHash(ctx, cmd, cr);
    } else if (cmd.funcName == "getOldStateRoot") {
        eval_getOldStateRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNewStateRoot") {
        eval_getNewStateRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNTxs") {
        eval_getNTxs(ctx, cmd, cr);
    } else if (cmd.funcName == "getRawTx") {
        eval_getRawTx(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxSigR") {
        eval_getTxSigR(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxSigS") {
        eval_getTxSigS(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxSigV") {
        eval_getTxSigV(ctx, cmd, cr);
    } else if (cmd.funcName == "getSequencerAddr") {
        eval_getSequencerAddr(ctx, cmd, cr);
    } else if (cmd.funcName == "getChainId") {
        eval_getChainId(ctx, cmd, cr);
    } else {
        cerr << "Error: eval_functionCall() function not defined: " << cmd.funcName << " line: " << ctx.ln << endl;
        exit(-1);
    } 
}

void eval_getGlobalHash(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }
    cr.type = crt_fea;
    scalar2fea(*ctx.pFr, ctx.globalHash, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}

void eval_getSequencerAddr(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }
    cr.type = crt_fea;
    mpz_class sequencerAddr;
    sequencerAddr.set_str(ctx.sequencerAddr, 16);
    scalar2fea(*ctx.pFr, sequencerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}

void eval_getChainId(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    cr.type = crt_fea;
    ctx.pFr->fromUI(cr.fea0, ctx.chainId);
    cr.fea1 = ctx.pFr->zero();
    cr.fea2 = ctx.pFr->zero();
    cr.fea3 = ctx.pFr->zero();
}

void eval_getOldStateRoot(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    cr.type = crt_fea;
    ctx.pFr->fromString(cr.fea0, ctx.oldStateRoot);
    cr.fea1 = ctx.pFr->zero();
    cr.fea2 = ctx.pFr->zero();
    cr.fea3 = ctx.pFr->zero();
}

void eval_getNewStateRoot(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    cr.type = crt_fea;
    ctx.pFr->fromString(cr.fea0, ctx.newStateRoot);
    cr.fea1 = ctx.pFr->zero();
    cr.fea2 = ctx.pFr->zero();
    cr.fea3 = ctx.pFr->zero();
}

void eval_getNTxs(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    cr.type = crt_fea;
    ctx.pFr->fromUI(cr.fea0, ctx.txs.size());
    cr.fea1 = ctx.pFr->zero();
    cr.fea2 = ctx.pFr->zero();
    cr.fea3 = ctx.pFr->zero();
}

void eval_getRawTx(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != cr.u64) {
        cerr << "Error: unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = cr.u64;

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != cr.u64) {
        cerr << "Error: unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t offset = cr.u64;

    // Get len by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != cr.u64) {
        cerr << "Error: unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t len = cr.u64;

    string d;
    //let d = "0x" +ctx.pTxs[txId].signData.slice(2+offset*2, 2+offset*2 + len*2);
    if (d.size() == 2) d = d + "0";

    cr.type = crt_fea;
    mpz_class tx;
    tx.set_str(d, 16);
    scalar2fea(*ctx.pFr, tx, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}

void eval_getTxSigR(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != cr.u64) {
        cerr << "Error: unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = cr.u64;

    cr.type = crt_fea;
    mpz_class sigr;
    //sigr = ctx.pTxs[txId].signature.r
    scalar2fea(*ctx.pFr, sigr, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}

void eval_getTxSigS(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != cr.u64) {
        cerr << "Error: unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = cr.u64;

    cr.type = crt_fea;
    mpz_class sigs;
    //sigs = ctx.pTxs[txId].signature.s
    scalar2fea(*ctx.pFr, sigs, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}

void eval_getTxSigV(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: Invalid number of parameters function " << cmd.funcName << " : " << ctx.ln << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != cr.u64) {
        cerr << "Error: unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = cr.u64;

    cr.type = crt_fea;
    mpz_class sigv;
    //sigv = ctx.pTxs[txId].signature.v
    scalar2fea(*ctx.pFr, sigv, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}
