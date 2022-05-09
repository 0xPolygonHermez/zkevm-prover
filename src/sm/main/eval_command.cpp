#include <iostream>

#include "config.hpp"
#include "eval_command.hpp"
#include "scalar.hpp"
#include "pols.hpp"
#include "opcode_address.hpp"

// Forwar declarations of internal functions
void eval_number       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getReg       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_declareVar   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_setVar       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getVar       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_add          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sub          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_neg          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mul          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_div          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mod          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_functionCall (Context &ctx, const RomCommand &cmd, CommandResult &cr);

void evalCommand (Context &ctx, const RomCommand &cmd, CommandResult &cr) {
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

void eval_number(Context &ctx, const RomCommand &cmd, CommandResult &cr) {
    cr.type = crt_scalar;
    cr.scalar = cmd.num;
}

/*************/
/* Variables */
/*************/

/* Declares a new variable, and fails if it already exists */
void eval_declareVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
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
void eval_getVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
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

void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr);

/* Sets variable to value, and fails if it does not exist */
void eval_setVar (Context &ctx, const RomCommand &cmd, CommandResult &cr)
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
    FieldElement fe;
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

void eval_left (Context &ctx, const RomCommand &cmd, CommandResult &cr)
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

void eval_getReg (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Get registry value, with the proper registry type
    if (cmd.regName=="A") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.A0[ctx.step], ctx.pols.A1[ctx.step], ctx.pols.A2[ctx.step], ctx.pols.A3[ctx.step], ctx.pols.A4[ctx.step], ctx.pols.A5[ctx.step], ctx.pols.A6[ctx.step], ctx.pols.A7[ctx.step]);
    } else if (cmd.regName=="B") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.B0[ctx.step], ctx.pols.B1[ctx.step], ctx.pols.B2[ctx.step], ctx.pols.B3[ctx.step], ctx.pols.B4[ctx.step], ctx.pols.B5[ctx.step], ctx.pols.B6[ctx.step], ctx.pols.B7[ctx.step]);
    } else if (cmd.regName=="C") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.C0[ctx.step], ctx.pols.C1[ctx.step], ctx.pols.C2[ctx.step], ctx.pols.C3[ctx.step], ctx.pols.C4[ctx.step], ctx.pols.C5[ctx.step], ctx.pols.C6[ctx.step], ctx.pols.C7[ctx.step]);
    } else if (cmd.regName=="D") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.D0[ctx.step], ctx.pols.D1[ctx.step], ctx.pols.D2[ctx.step], ctx.pols.D3[ctx.step], ctx.pols.D4[ctx.step], ctx.pols.D5[ctx.step], ctx.pols.D6[ctx.step], ctx.pols.D7[ctx.step]);
    } else if (cmd.regName=="E") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.E0[ctx.step], ctx.pols.E1[ctx.step], ctx.pols.E2[ctx.step], ctx.pols.E3[ctx.step], ctx.pols.E4[ctx.step], ctx.pols.E5[ctx.step], ctx.pols.E6[ctx.step], ctx.pols.E7[ctx.step]);
    } else if (cmd.regName=="SR") {
        cr.type = crt_scalar;
        fea2scalar(ctx.fr, cr.scalar, ctx.pols.SR0[ctx.step], ctx.pols.SR1[ctx.step], ctx.pols.SR2[ctx.step], ctx.pols.SR3[ctx.step], ctx.pols.SR4[ctx.step], ctx.pols.SR5[ctx.step], ctx.pols.SR6[ctx.step], ctx.pols.SR7[ctx.step]);
    } else if (cmd.regName=="CTX") {
        cr.type = crt_u32;
        cr.u32 = ctx.pols.CTX[ctx.step];
    } else if (cmd.regName=="SP") {
        cr.type = crt_u16;
        cr.u16 = ctx.pols.SP[ctx.step];
    } else if (cmd.regName=="PC") {
        cr.type = crt_u32;
        cr.u32 = ctx.pols.PC[ctx.step];
    } else if (cmd.regName=="MAXMEM") {
        cr.type = crt_u32;
        cr.u32 = ctx.pols.MAXMEM[ctx.step];
    } else if (cmd.regName=="GAS") {
        cr.type = crt_u64;
        cr.u64 = ctx.pols.CTX[ctx.step];
    } else if (cmd.regName=="zkPC") {
        cr.type = crt_u32;
        cr.u32 = ctx.pols.zkPC[ctx.step];
    } else {
        cerr << "Error: eval_getReg() Invalid register: " << cmd.regName << ": " << ctx.zkPC << endl;
        exit(-1);
    }
}

void cr2fe(FiniteField &fr, const CommandResult &cr, FieldElement &fe)
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

void cr2scalar(FiniteField &fr, const CommandResult &cr, mpz_class &s)
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

void eval_add(Context &ctx, const RomCommand &cmd, CommandResult &cr)
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

void eval_sub(Context &ctx, const RomCommand &cmd, CommandResult &cr)
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

void eval_neg(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    cr.type = crt_scalar;
    cr.scalar = -a;
}

void eval_mul(Context &ctx, const RomCommand &cmd, CommandResult &cr)
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

void eval_div(Context &ctx, const RomCommand &cmd, CommandResult &cr)
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

void eval_mod(Context &ctx, const RomCommand &cmd, CommandResult &cr)
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
void eval_getGlobalHash       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getGlobalExitRoot   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getOldStateRoot     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getNewStateRoot     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getOldLocalExitRoot (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getNewLocalExitRoot (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getNTxs             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getRawTx            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getSequencerAddr    (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getChainId          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getDefaultChainId   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBatchNum         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBatchHashData    (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTxs              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTxsLen           (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_addrOp              (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getTimestamp        (Context &ctx, const RomCommand &cmd, CommandResult &cr);

void eval_functionCall (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Call the proper internal function
    if (cmd.funcName == "getGlobalHash") {
        eval_getGlobalHash(ctx, cmd, cr);
    } else if (cmd.funcName == "getGlobalExitRoot") {
        eval_getGlobalExitRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getOldStateRoot") {
        eval_getOldStateRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNewStateRoot") {
        eval_getNewStateRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getOldLocalExitRoot") {
        eval_getOldLocalExitRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNewLocalExitRoot") {
        eval_getNewLocalExitRoot(ctx, cmd, cr);
    /*} else if (cmd.funcName == "getNTxs") {
        eval_getNTxs(ctx, cmd, cr);
    } else if (cmd.funcName == "getRawTx") {
        eval_getRawTx(ctx, cmd, cr);*/
    } else if (cmd.funcName == "getSequencerAddr") {
        eval_getSequencerAddr(ctx, cmd, cr);
    } else if (cmd.funcName == "getChainId") {
        eval_getChainId(ctx, cmd, cr);
    } else if (cmd.funcName == "getDefaultChainId") {
        eval_getDefaultChainId(ctx, cmd, cr);
    } else if (cmd.funcName == "getNumBatch") {
        eval_getBatchNum(ctx, cmd, cr);
    } else if (cmd.funcName == "getBatchHashData") { // To be generated by preprocess_TX()
        eval_getBatchHashData(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxs") {
        eval_getTxs(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxsLen") {
        eval_getTxsLen(ctx, cmd, cr);
    } else if (cmd.funcName == "addrOp") {
        eval_addrOp(ctx, cmd, cr);
    } else if (cmd.funcName == "getTimestamp") {
        eval_getTimestamp(ctx, cmd, cr);
    } else {
        cerr << "Error: eval_functionCall() function not defined: " << cmd.funcName << " line: " << ctx.zkPC << endl;
        exit(-1);
    } 
}

void eval_getGlobalHash(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalHash() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.globalHash as a field element
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.input.globalHash, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getGlobalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalExitRoot() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.globalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class globalExitRoot(ctx.input.globalExitRoot);
    scalar2fea(ctx.fr, globalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getSequencerAddr(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getSequencerAddr() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.sequencerAddr as a field element array
    cr.type = crt_fea;
    mpz_class sequencerAddr(ctx.input.publicInputs.sequencerAddr);
    scalar2fea(ctx.fr, sequencerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getChainId(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getChainId() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.chainId as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.chainId);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getDefaultChainId(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getDefaultChainId() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.defaultChainId as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.defaultChainId);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getBatchNum(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getBatchNum() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.batchNum as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.batchNum);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getOldStateRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getOldStateRoot() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.oldStateRoot as a field element array
    cr.type = crt_fea;
    mpz_class oldStateRoot(ctx.input.publicInputs.oldStateRoot); // This field could be parsed out of the main loop, but it is only called once
    scalar2fea(ctx.fr, oldStateRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getNewStateRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNewStateRoot() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.newStateRoot as a field element array
    cr.type = crt_fea;
    mpz_class newStateRoot(ctx.input.publicInputs.newStateRoot); // This field could be parsed out of the main loop, but it is only called once
    scalar2fea(ctx.fr, newStateRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getOldLocalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getOldLocalExitRoot() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.oldLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class oldLocalExitRoot(ctx.input.publicInputs.oldLocalExitRoot);
    scalar2fea(ctx.fr, oldLocalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getNewLocalExitRoot(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNewLocalExitRoot() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.newLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class newLocalExitRoot(ctx.input.publicInputs.newLocalExitRoot);
    scalar2fea(ctx.fr, newLocalExitRoot, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getTxsLen(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getTxsLen() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.txsLen/2 as a field element array
    cr.type = crt_fea;
    u642fe(ctx.fr, cr.fea0, (ctx.input.batchL2Data.size() - 2) / 2);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getTxs(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_getTxs() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_getTxs() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t offset = fe2n(ctx.fr, cr.fe);

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getTxs() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t len = cr.scalar.get_ui();

    string resultString = ctx.input.batchL2Data.substr(2+offset*2, len*2);
    if (resultString.size() == 0) resultString += "0";

    // Return result as a field element array
    mpz_class resultScalar(resultString, 16);
    cr.type = crt_fea;
    scalar2fea(ctx.fr, resultScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

void eval_getBatchHashData(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getBatchHashData() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.batchHashData as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, ctx.input.batchHashData, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/*void eval_getNTxs(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getNTxs() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return the number of transactions as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.txs.size());
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();

#ifdef LOG_TXS
    cout << "eval_getNTxs() returns " << ctx.input.txs.size() << endl;
#endif
}

void eval_getRawTx(Context &ctx, const RomCommand &cmd, CommandResult &cr)
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
    string d = "0x" + ctx.input.txs[txId].substr(2+offset*2, len*2);
    if (d.size() == 2) d = d + "0";

    // Return the requested transaction as a field element array
    cr.type = crt_fea;
    mpz_class tx(d);
    scalar2fea(ctx.fr, tx, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);

#ifdef LOG_TXS
    cout << "eval_getRawTx() returns " << d << endl;
#endif
}*/

void eval_addrOp(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_addrOp() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_addrOp() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t codeId = fe2n(ctx.fr, cr.fe);

    // Get offset by executing cmd.params[1]
    /*evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getTxs() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t len = cr.scalar.get_ui();*/

    //string resultString = ctx.input.batchL2Data.substr(2+offset*2, len*2);
    //if (resultString.size() == 0) resultString += "0";

    // Return result as a field element array
    //mpz_class resultScalar(resultString, 16);
    cr.type = crt_fea;
    //scalar2fea(ctx.fr, resultScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3);

    uint64_t addr = opcodeAddress[codeId];//ctx.rom.labels[codes[codeId]];
    ctx.fr.fromUI(cr.fea0, addr);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

void eval_getTimestamp(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getTimestamp() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.chainId as a field element array
    cr.type = crt_fea;
    ctx.fr.fromUI(cr.fea0, ctx.input.publicInputs.timestamp);
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}