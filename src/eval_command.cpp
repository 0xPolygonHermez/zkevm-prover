#include <iostream>

#include "config.hpp"
#include "eval_command.hpp"
#include "scalar.hpp"
#include "pols.hpp"

// Forwar declarations of internal functions
void eval_number       (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getReg       (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_declareVar   (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_setVar       (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getVar       (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_add          (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_sub          (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_neg          (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_mul          (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_div          (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_mod          (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_functionCall (Context &ctx, RomCommand &cmd, CommandResult &cr);

void evalCommand (Context &ctx, RomCommand &cmd, CommandResult &cr) {
    if (cmd.op=="number") {
        eval_number(ctx, cmd, cr);
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
    cr.type = crt_scalar;
    cr.scalar = cmd.num;
}

/*************/
/* Variables */
/*************/

/* Declares a new variable, and fails if it already exists */
void eval_declareVar (Context &ctx, RomCommand &cmd, CommandResult &cr)
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
    ctx.vars[cmd.varName] = ctx.fr.zero();

#ifdef LOG_VARIABLES
    cout << "Declare variable: " << cmd.varName << endl;
#endif

    // Return the current value of this variable
    cr.type = crt_fe;
    cr.fe = ctx.vars[cmd.varName];
}

/* Gets the value of the variable, and fails if it does not exist */
void eval_getVar (Context &ctx, RomCommand &cmd, CommandResult &cr)
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
    cout << "Get variable: " << cmd.varName << " fe: " << ctx.fr.toString(ctx.vars[cmd.varName], 16) << endl;
#endif

    // Return the current value of this variable
    cr.type = crt_fe;
    cr.fe = ctx.vars[cmd.varName];
}

void eval_left (Context &ctx, RomCommand &cmd, CommandResult &cr);

/* Sets variable to value, and fails if it does not exist */
void eval_setVar (Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check that tag contains a values array
    if (cmd.values.size()==0) {
        cerr << "Error: eval_setVar() could not find array values in setVar command" << endl;
        exit(-1);
    }

    // Get varName from the first element in values
    eval_left(ctx,*cmd.values[0], cr);
    if (cr.type != crt_string) {
        cerr << "Error: eval_setVar() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    string varName = cr.str;

    // Check that this variable exists
    if ( ctx.vars.find(varName) == ctx.vars.end() ) {
        cerr << "Error: eval_setVar() Undefined variable: " << varName << endl;
        exit(-1);
    }

    // Call evalCommand() to build the field element value for this variable
    evalCommand(ctx, *cmd.values[1], cr);

    // Get the field element value from the command result
    RawFr::Element fe;
    cr2fe(ctx.fr, cr, fe);

    // Store the value as the new variable value
    ctx.vars[varName] = fe;

    // Return the current value of the variable
    cr.type = crt_fe;
    cr.fe = ctx.vars[cmd.varName];

#ifdef LOG_VARIABLES
    cout << "Set variable: " << varName << " fe: " << ctx.fr.toString(ctx.vars[varName], 16) << endl;
#endif
}

void eval_left (Context &ctx, RomCommand &cmd, CommandResult &cr)
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
    cerr << "Error: eval_left() invalid left expression, op: " << cmd.op << "ln: " << ctx.zkPC << endl;
    exit(-1);
}

void eval_getReg (Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Get registry value, with the proper registry type
    if (cmd.regName=="A") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, pol(A0)[ctx.step], pol(A1)[ctx.step], pol(A2)[ctx.step], pol(A3)[ctx.step]);
    } else if (cmd.regName=="B") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, pol(B0)[ctx.step], pol(B1)[ctx.step], pol(B2)[ctx.step], pol(B3)[ctx.step]);
    } else if (cmd.regName=="C") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, pol(C0)[ctx.step], pol(C1)[ctx.step], pol(C2)[ctx.step], pol(C3)[ctx.step]);
    } else if (cmd.regName=="D") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, pol(D0)[ctx.step], pol(D1)[ctx.step], pol(D2)[ctx.step], pol(D3)[ctx.step]);
    } else if (cmd.regName=="E") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, pol(E0)[ctx.step], pol(E1)[ctx.step], pol(E2)[ctx.step], pol(E3)[ctx.step]);
    } else if (cmd.regName=="SR") {
        cr.type = crt_fe;
        cr.fe = pol(SR)[ctx.step];
    } else if (cmd.regName=="CTX") {
        cr.type = crt_u32;
        cr.u32 = pol(CTX)[ctx.step];
    } else if (cmd.regName=="SP") {
        cr.type = crt_u16;
        cr.u16 = pol(SP)[ctx.step];
    } else if (cmd.regName=="PC") {
        cr.type = crt_u32;
        cr.u32 = pol(PC)[ctx.step];
    } else if (cmd.regName=="MAXMEM") {
        cr.type = crt_u32;
        cr.u32 = pol(MAXMEM)[ctx.step];
    } else if (cmd.regName=="GAS") {
        cr.type = crt_u64;
        cr.u64 = pol(CTX)[ctx.step];
    } else if (cmd.regName=="zkPC") {
        cr.type = crt_u32;
        cr.u32 = pol(zkPC)[ctx.step];
    } else {
        cerr << "Error: eval_getReg() Invalid register: " << cmd.regName << ": " << ctx.zkPC << endl;
        exit(-1);
    }
}

void cr2fe(RawFr &fr, CommandResult &cr, RawFr::Element &fe)
{
    if (cr.type == crt_fe)
    {
        fe = cr.fe;
    }
    else if (cr.type == crt_scalar)
    {
        scalar2fe(fr, cr.scalar, fe);
    }
    else
    {
        cerr << "Error: cr2fe() unexpected type: " << cr.type << endl;
        exit(-1);
    }
}

void cr2scalar(RawFr &fr, CommandResult &cr, mpz_class &s)
{
    if (cr.type == crt_scalar)
    {
        s = cr.scalar;
    }
    else if (cr.type == crt_fe)
    {
        fe2scalar(fr, s, cr.fe);
    }
    else if (cr.type == crt_u64)
    {
        s = cr.u64;
    }
    else if (cr.type == crt_u32)
    {
        s = cr.u32;
    }
    else if (cr.type == crt_u16)
    {
        s = cr.u16;
    }
    else
    {
        cerr << "Error: cr2scalar() unexpected type: " << cr.type << endl;
        exit(-1);
    }
}

void eval_add(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a + b;
}

void eval_sub(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a - b;
}

void eval_neg(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    cr.type = crt_scalar;
    cr.scalar = -a;
}

void eval_mul(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    mpz_class mask256;
    mpz_class one(1);
    mask256 = (one << 256) - one;

    cr.type = crt_scalar;
    cr.scalar = (a * b) & mask256;
}

void eval_div(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a / b;
}

void eval_mod(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    cr.scalar = a % b;
}

// Forward declaration of internal callable functions
void eval_getGlobalHash    (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getOldStateRoot  (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getNewStateRoot  (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getNTxs          (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getRawTx         (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getTxSigR        (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getTxSigS        (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getTxSigV        (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getSequencerAddr (Context &ctx, RomCommand &cmd, CommandResult &cr);
void eval_getChainId       (Context &ctx, RomCommand &cmd, CommandResult &cr);

void eval_functionCall (Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Call the proper internal function
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
        cerr << "Error: eval_functionCall() function not defined: " << cmd.funcName << " line: " << ctx.zkPC << endl;
        exit(-1);
    } 
}

void eval_getGlobalHash(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalHash() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.globalHash as a field element
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.globalHash, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}

void eval_getSequencerAddr(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getSequencerAddr() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.sequencerAddr as a field element array
    cr.type = crt_fea;
    mpz_class sequencerAddr(ctx.sequencerAddr);
    scalar2fea(ctx.fr, sequencerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3);
}

void eval_getChainId(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getChainId() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.chainId as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.chainId);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
}

void eval_getOldStateRoot(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getOldStateRoot() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.oldStateRoot as a field element array
    cr.type = crt_fea;
    string2fe(ctx.fr, ctx.oldStateRoot, cr.fea0);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
}

void eval_getNewStateRoot(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNewStateRoot() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.newStateRoot as a field element array
    cr.type = crt_fea;
    string2fe(ctx.fr, ctx.newStateRoot, cr.fea0);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
}

void eval_getNTxs(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNTxs() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return the number of transactions as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.txs.size());
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();

#ifdef LOG_TXS
    cout << "eval_getNTxs() returns " << ctx.txs.size() << endl;
#endif
}

void eval_getRawTx(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 3) {
        cerr << "Error: eval_getRawTx() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_getRawTx() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = fe2n(ctx.fr, ctx.prime, cr.fe);

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_getRawTx() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t offset = fe2n(ctx.fr, ctx.prime, cr.fe);

    // Get len by executing cmd.params[2]
    evalCommand(ctx, *cmd.params[2], cr);
    if (cr.type != crt_scalar)
    { 
        cerr << "Error: eval_getRawTx() 3 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t len = cr.scalar.get_ui();

    // Build an hexa string with the requested transaction, offset and length
    string d = "0x" + ctx.txs[txId].signData.substr(2+offset*2, len*2);
    if (d.size() == 2) d = d + "0";

    // Return the requested transaction as a field element array
    cr.type = crt_fea;
    mpz_class tx(d);
    scalar2fea(ctx.fr, tx, cr.fea0, cr.fea1, cr.fea2, cr.fea3);

#ifdef LOG_TXS
    cout << "eval_getRawTx() returns " << d << endl;
#endif
}

void eval_getTxSigR(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_getTxSigR() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_getTxSigR() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = fe2u64(ctx.fr, cr.fe);

    // Return the requested transaction signature r element as a scalar
    cr.type = crt_scalar;
    cr.scalar = ctx.txs[txId].r;

#ifdef LOG_TXS
    cout << "eval_getTxSigR() returns " << ctx.txs[txId].r.get_str(16) << endl;
#endif
}

void eval_getTxSigS(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_getTxSigS() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_getTxSigS() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = fe2u64(ctx.fr, cr.fe);

    // Return the requested transaction signature s element as a scalar
    cr.type = crt_scalar;
    cr.scalar = ctx.txs[txId].s;

#ifdef LOG_TXS
    cout << "eval_getTxSigS() returns " << ctx.txs[txId].r.get_str(16) << endl;
#endif
}

void eval_getTxSigV(Context &ctx, RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_getTxSigV() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get txId by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_getTxSigV() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t txId = fe2u64(ctx.fr, cr.fe);

    // Return the requested transaction signature v element as an unsigned 16 bits integer
    cr.type = crt_u16;
    cr.u16 = ctx.txs[txId].v;

#ifdef LOG_TXS
    cout << "eval_getTxSigV() returns " << ctx.txs[txId].v << endl;
#endif
}
