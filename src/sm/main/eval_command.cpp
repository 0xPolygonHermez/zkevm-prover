#include <iostream>

#include "config.hpp"
#include "eval_command.hpp"
#include "scalar.hpp"
#include "pols.hpp"
#include "opcode_address.hpp"

// Forwar declarations of internal functions
void eval_number            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getReg            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_declareVar        (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_setVar            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getVar            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_add               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sub               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_neg               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mul               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_div               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_mod               (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_logical_operation (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bit_operation     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_if                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getMemValue       (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_functionCall      (Context &ctx, const RomCommand &cmd, CommandResult &cr);

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
        eval_mul(ctx, cmd, cr);
    } else if (cmd.op=="div") {
        eval_div(ctx, cmd, cr);
    } else if (cmd.op=="mod") {
        eval_mod(ctx, cmd, cr);
    } else if (cmd.op == "or" || cmd.op == "and" || cmd.op == "gt" || cmd.op == "ge" || cmd.op == "lt" || cmd.op == "le" ||
               cmd.op == "eq" || cmd.op == "ne" || cmd.op == "not" ) {
        eval_logical_operation(ctx, cmd, cr);
    } else if (cmd.op == "bitand" || cmd.op == "bitor" || cmd.op == "bitxor" || cmd.op == "bitnot"|| cmd.op == "shl" || cmd.op == "shr") {
        eval_bit_operation(ctx, cmd, cr);
    } else if (cmd.op == "if") {
        eval_if(ctx, cmd, cr);
    } else if (cmd.op == "getMemValue") {
        eval_getMemValue(ctx, cmd, cr);
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

void eval_logical_operation (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    if (cmd.op == "not")
    {
        cr.type = crt_scalar;
        cr.scalar = (a) ? 0 : 1;
        return;
    }

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    
         if (cmd.op == "or" ) cr.scalar = (a || b) ? 1 : 0;
    else if (cmd.op == "and") cr.scalar = (a && b) ? 1 : 0;
    else if (cmd.op == "eq" ) cr.scalar = (a == b) ? 1 : 0;
    else if (cmd.op == "ne" ) cr.scalar = (a != b) ? 1 : 0;
    else if (cmd.op == "gt" ) cr.scalar = (a >  b) ? 1 : 0;
    else if (cmd.op == "ge" ) cr.scalar = (a >= b) ? 1 : 0;
    else if (cmd.op == "lt" ) cr.scalar = (a <  b) ? 1 : 0;
    else if (cmd.op == "le" ) cr.scalar = (a <= b) ? 1 : 0;
    else
    {
        cerr << "Error: eval_logical_operation() operation not defined: " << cmd.op << endl;
        exit(-1);
    }
}

void eval_bit_operation (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    if (cmd.op == "bitnot")
    {
        cr.type = crt_scalar;
        cr.scalar = ~a;
        return;
    }

    evalCommand(ctx, *cmd.values[1], cr);
    mpz_class b;
    cr2scalar(ctx.fr, cr, b);

    cr.type = crt_scalar;
    
         if (cmd.op == "bitor" ) cr.scalar = a | b;
    else if (cmd.op == "bitand") cr.scalar = a & b;
    else if (cmd.op == "bitxor") cr.scalar = a ^ b;
    else if (cmd.op == "shl"   ) cr.scalar = (a << b.get_ui());
    else if (cmd.op == "shr"   ) cr.scalar = (a >> b.get_ui());
    else if (cmd.op == "ge" ) cr.scalar = (a >= b) ? 1 : 0;
    else if (cmd.op == "lt" ) cr.scalar = (a <  b) ? 1 : 0;
    else if (cmd.op == "le" ) cr.scalar = (a <= b) ? 1 : 0;
    else
    {
        cerr << "Error: eval_bit_operation() operation not defined: " << cmd.op << endl;
        exit(-1);
    }
}

void eval_if (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    evalCommand(ctx, *cmd.values[0], cr);
    mpz_class a;
    cr2scalar(ctx.fr, cr, a);

    if (a)
    {
        evalCommand(ctx, *cmd.values[1], cr);
        mpz_class b;
        cr2scalar(ctx.fr, cr, b);

        cr.type = crt_scalar;
        cr.scalar = b;
    }
    else
    {
        evalCommand(ctx, *cmd.values[2], cr);
        mpz_class c;
        cr2scalar(ctx.fr, cr, c);

        cr.type = crt_scalar;
        cr.scalar = c;
    }
}

void eval_getMemValue (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    cr.type = crt_scalar;
    Fea fea = ctx.mem[cmd.offset];
    fea2scalar(ctx.fr, cr.scalar, fea.fe0, fea.fe1, fea.fe2, fea.fe3, fea.fe4, fea.fe5, fea.fe6, fea.fe7);
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
void eval_cond                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_inverseFpEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_inverseFnEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_sqrtFpEc            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_dumpRegs            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_dump                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_dumphex             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yAddPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_xDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_yDblPointEc         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBytecode         (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getByte             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getBytecodeLength   (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getHashBytecode     (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_touchedAddress      (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_touchedStorageSlots (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_bitwise             (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_comp                (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_loadScalar          (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_getGlobalExitRootManagerAddr (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_log                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_copyTouchedAddress  (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_exp                 (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_storeLog            (Context &ctx, const RomCommand &cmd, CommandResult &cr);
void eval_precompiled         (Context &ctx, const RomCommand &cmd, CommandResult &cr);

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
    } else if (cmd.funcName == "getSequencerAddr") {
        eval_getSequencerAddr(ctx, cmd, cr);
    } else if (cmd.funcName == "getChainId") {
        eval_getChainId(ctx, cmd, cr);
    } else if (cmd.funcName == "getOldLocalExitRoot") {
        eval_getOldLocalExitRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNewLocalExitRoot") {
        eval_getNewLocalExitRoot(ctx, cmd, cr);
    } else if (cmd.funcName == "getNumBatch") {
        eval_getBatchNum(ctx, cmd, cr);
    } else if (cmd.funcName == "getTimestamp") {
        eval_getTimestamp(ctx, cmd, cr);
    //} else if (cmd.funcName == "getDefaultChainId") {
    //    eval_getDefaultChainId(ctx, cmd, cr);
    } else if (cmd.funcName == "getBatchHashData") { // To be generated by preprocess_TX()
        eval_getBatchHashData(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxs") {
        eval_getTxs(ctx, cmd, cr);
    } else if (cmd.funcName == "getTxsLen") {
        eval_getTxsLen(ctx, cmd, cr);
    } else if (cmd.funcName == "addrOp") {
        eval_addrOp(ctx, cmd, cr);
    } else if (cmd.funcName == "cond") {
        return eval_cond(ctx, cmd, cr);
    /*} else if (cmd.funcName == "inverseFpEc") {
        return eval_inverseFpEc(ctx, cmd, cr);
    } else if (cmd.funcName == "inverseFnEc") {
        return eval_inverseFnEc(ctx, cmd, cr);
    } else if (cmd.funcName == "sqrtFpEc") {
        return eval_sqrtFpEc(ctx, cmd, cr);
    } else if (cmd.funcName == "dumpRegs") {
        return eval_dumpRegs(ctx, cmd, cr);
    } else if (cmd.funcName == "dump") {
        return eval_dump(ctx, cmd, cr);
    } else if (cmd.funcName == "dumphex") {
        return eval_dumphex(ctx, cmd, cr);
    } else if (cmd.funcName == "xAddPointEc") {
        return eval_xAddPointEc(ctx, cmd, cr);
    } else if (cmd.funcName == "yAddPointEc") {
        return eval_yAddPointEc(ctx, cmd, cr);
    } else if (cmd.funcName == "xDblPointEc") {
        return eval_xDblPointEc(ctx, cmd, cr);
    } else if (cmd.funcName == "yDblPointEc") {
        return eval_yDblPointEc(ctx, cmd, cr);
    } else if (cmd.funcName == "getBytecode") { // Added by opcodes
        return eval_getBytecode(ctx, cmd, cr);*/
    } else if (cmd.funcName == "getByte") {
        return eval_getByte(ctx, cmd, cr);
    /*} else if (cmd.funcName == "getBytecodeLength") {
        return eval_getBytecodeLength(ctx, cmd, cr);
    } else if (cmd.funcName == "getHashBytecode") {
        return eval_getHashBytecode(ctx, cmd, cr);*/
    } else if (cmd.funcName == "touchedAddress") {
        return eval_touchedAddress(ctx, cmd, cr);
    /*} else if (cmd.funcName == "touchedStorageSlots") {
        return eval_touchedStorageSlots(ctx, cmd, cr);
    } else if (cmd.funcName.find("bitwise") != string::npos){
        return eval_bitwise(ctx, cmd, cr);
    } else if (cmd.funcName.find("comp_") == 0) {
        return eval_comp(ctx, cmd, cr);
    } else if (cmd.funcName == "loadScalar") {
        return eval_loadScalar(ctx, cmd, cr);*/
    } else if (cmd.funcName == "getGlobalExitRootManagerAddr") {
        return eval_getGlobalExitRootManagerAddr(ctx, cmd, cr);
    /*} else if (cmd.funcName == "log") {
        return eval_log(ctx, cmd, cr);
    } else if (cmd.funcName == "copyTouchedAddress"){
        return eval_copyTouchedAddress(ctx, cmd, cr);
    } else if (cmd.funcName == "exp") {
        return eval_exp(ctx, cmd, cr);
    } else if (cmd.funcName == "storeLog") {
        return eval_storeLog(ctx, cmd, cr);
    } else if (cmd.funcName.find("precompiled_") == 0) {
        return eval_precompiled(ctx, cmd, cr);*/
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

void eval_cond (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 1) {
        cerr << "Error: eval_cond() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_fe) {
        cerr << "Error: eval_cond() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t result = fe2n(ctx.fr, cr.fe);
    
    cr.type = crt_fea;
    if (result)
    {
        cr.fea0 = ctx.fr.zero() - ctx.fr.one(); // -1
    }
    else
    {
        cr.fea0 = ctx.fr.zero();
    }
    cr.fea1 = ctx.fr.zero();
    cr.fea2 = ctx.fr.zero();
    cr.fea3 = ctx.fr.zero();
    cr.fea4 = ctx.fr.zero();
    cr.fea5 = ctx.fr.zero();
    cr.fea6 = ctx.fr.zero();
    cr.fea7 = ctx.fr.zero();
}

/*

function eval_getBytecode(ctx, tag) {
    if (tag.params.length != 2 && tag.params.length != 3) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let hashcontract = evalCommand(ctx,tag.params[0]);
    hashcontract = "0x" + hashcontract.toString(16).padStart(64, '0');
    const bytecode = ctx.input.contractsBytecode[hashcontract];
    const offset = Number(evalCommand(ctx,tag.params[1]));
    let len;
    if(tag.params[2])
        len = Number(evalCommand(ctx,tag.params[2]));
    else
        len = 1;
    let d = "0x" + bytecode.slice(2+offset*2, 2+offset*2 + len*2);
    if (d.length == 2) d = d+'0';
    return scalar2fea(ctx.Fr, Scalar.e(d));
}*/

/*function eval_getByte(ctx, tag) {
    if (tag.params.length != 2 && tag.params.length != 3) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const bytes = evalCommand(ctx,tag.params[0]).toString(16).padStart(64, "0");
    let offset = Number(evalCommand(ctx,tag.params[1]));
    let len;
    if(tag.params[2])
        len = Number(evalCommand(ctx,tag.params[2]));
    else
        len = 1;
    if(bytes.startsWith("0x"))
        offset = 2+offset*2;
    else
        offset = offset*2;
    let d = "0x" + bytes.slice(offset, offset + len*2);
    if (d.length == 2) d = d+'0';
    return scalar2fea(ctx.Fr, Scalar.e(d));
}*/

void eval_getByte (Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2 && cmd.params.size() != 3) {
        cerr << "Error: eval_getByte() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get bytes by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getByte() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    string aux = cr.scalar.get_str(16);
    string bytes = NormalizeToNFormat(aux, 64);

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_getByte() unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint64_t offset = cr.scalar.get_ui();

    // Get length by executing cmd.params[2]
    uint64_t len = 1;
    if (cmd.params.size() == 3)
    {
        evalCommand(ctx, *cmd.params[2], cr);
        if (cr.type != crt_fe) {
            cerr << "Error: eval_getByte() unexpected command result type: " << cr.type << endl;
            exit(-1);
        }
        len = fe2n(ctx.fr, cr.fe);
    }

    // Check the total length
    if ((offset+2*len) > 64)
    {
        cerr << "Error: eval_getByte() invalid values offset=" << offset << " len=" << len << endl;
        exit(-1);
    }

    // Get the requested substring
    string auxString = bytes.substr(offset, 2*len);
    mpz_class auxScalar;
    auxScalar.set_str(auxString, 16);

    // Return as a field element array
    cr.type = crt_fea;
    scalar2fea(ctx.fr, auxScalar, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/*function eval_getBytecodeLength(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let hashcontract = evalCommand(ctx,tag.params[0]);
    hashcontract = "0x" + hashcontract.toString(16).padStart(64, '0');
    const bytecode = ctx.input.contractsBytecode[hashcontract];
    const d = (bytecode.length - 2)/2;
    return scalar2fea(ctx.Fr, Scalar.e(d));
}

function eval_getHashBytecode(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let addr = evalCommand(ctx,tag.params[0]);
    addr = addr.toString(16).padStart(40, '0')
    return scalar2fea(ctx.Fr, Scalar.e(ctx.input.contractsBytecode["0x"+addr.toString(16).toLowerCase()]));
}*/

/*function eval_touchedAddress(ctx, tag) {
    if (tag.params.length != 2) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let addr = evalCommand(ctx,tag.params[0]);
    let context = evalCommand(ctx,tag.params[1]);

    // if address is precompiled smart contract considered warm access
    if (Scalar.gt(addr, 0) && Scalar.lt(addr, 10)){
        return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
    }

    // if address in touchedAddress return 0
    if(ctx.input.touchedAddress[context] && ctx.input.touchedAddress[context].filter(x => x == addr).length > 0) {
        return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
    } else {
    //if address not in touchedAddress, return 1
        if(ctx.input.touchedAddress[context]) {
            ctx.input.touchedAddress[context].push(addr);
        } else {
            ctx.input.touchedAddress[context] = [addr];
        }
        return [ctx.Fr.e(1), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
    }
}*/


void eval_touchedAddress(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 2) {
        cerr << "Error: eval_touchedAddress() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Get offset by executing cmd.params[0]
    evalCommand(ctx, *cmd.params[0], cr);
    if (cr.type != crt_scalar) {
        cerr << "Error: eval_touchedAddress() 1 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    mpz_class addr = cr.scalar;

    // Get offset by executing cmd.params[1]
    evalCommand(ctx, *cmd.params[1], cr);
    if (cr.type != crt_u32) {
        cerr << "Error: eval_touchedAddress() 2 unexpected command result type: " << cr.type << endl;
        exit(-1);
    }
    uint32_t context = cr.u32;

    // if address is precompiled smart contract considered warm access
    if ((addr>0) && (addr<10))
    {
        cr.type = crt_fea;
        cr.fea0 = ctx.fr.zero();
        cr.fea1 = ctx.fr.zero();
        cr.fea2 = ctx.fr.zero();
        cr.fea3 = ctx.fr.zero();
        cr.fea4 = ctx.fr.zero();
        cr.fea5 = ctx.fr.zero();
        cr.fea6 = ctx.fr.zero();
        cr.fea7 = ctx.fr.zero();
    }
    else
    {
        // If address is not in touchedAddress, then return 1, and store it in the map's vector
        if(ctx.touchedAddress.find(context) != ctx.touchedAddress.end())
        {
            ctx.touchedAddress[context].push_back(addr);
        }
        else
        {
            vector<mpz_class> aux;
            aux.push_back(addr);
            ctx.touchedAddress[context] = aux;
        }
        cr.type = crt_fea;
        cr.fea0 = ctx.fr.one();
        cr.fea1 = ctx.fr.zero();
        cr.fea2 = ctx.fr.zero();
        cr.fea3 = ctx.fr.zero();
        cr.fea4 = ctx.fr.zero();
        cr.fea5 = ctx.fr.zero();
        cr.fea6 = ctx.fr.zero();
        cr.fea7 = ctx.fr.zero();
    }
}

/*function eval_copyTouchedAddress(ctx, tag) {
    if (tag.params.length != 2) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let ctx1 = evalCommand(ctx,tag.params[0]);
    let ctx2 = evalCommand(ctx,tag.params[1]);
    // if address in touchedAddress return 0
    if(ctx.input.touchedAddress[ctx1])
        ctx.input.touchedAddress[ctx2] = ctx.input.touchedAddress[ctx1];
    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_touchedStorageSlots(ctx, tag) {
    if (tag.params.length != 3) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    let addr = evalCommand(ctx,tag.params[0]);
    let key = evalCommand(ctx,tag.params[1])
    let context = evalCommand(ctx,tag.params[2]);
    // if address in touchedStorageSlots return 0
    if(ctx.input.touchedStorageSlots[context] && ctx.input.touchedStorageSlots[context].filter(x => (x.addr == addr && x.key == key)).length > 0) {
        return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
    } else {
    //if addres not in touchedStorageSlots, return 1
        if(ctx.input.touchedStorageSlots[context]) {
            ctx.input.touchedStorageSlots[context].push({addr, key});
        } else {
            ctx.input.touchedStorageSlots[context] = [{addr, key}];
        }
        return [ctx.Fr.e(1), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
    }
}

function eval_exp(ctx, tag) {
    if (tag.params.length != 2) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    const a = evalCommand(ctx,tag.params[0]);
    const b = evalCommand(ctx,tag.params[1])
    return scalar2fea(ctx.Fr, Scalar.exp(a,b));;
}

function eval_bitwise(ctx, tag){
    const func = tag.funcName.split('_')[1];
    const a = evalCommand(ctx,tag.params[0]);
    let b;

    switch (func){
        case 'and':
            checkParams(ctx, tag, 2);
            b = evalCommand(ctx,tag.params[1]);
            return Scalar.band(a, b);
        case 'or':
            checkParams(ctx, tag, 2);
            b = evalCommand(ctx,tag.params[1]);
            return Scalar.bor(a, b);
        case 'xor':
            checkParams(ctx, tag, 2);
            b = evalCommand(ctx,tag.params[1]);
            return Scalar.bxor(a, b);
        case 'not':
            checkParams(ctx, tag, 1);
            return Scalar.bxor(a, Mask256);
        default:
            throw new Error(`Invalid bitwise operation ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function eval_comp(ctx, tag){
    checkParams(ctx, tag, 2);

    const func = tag.funcName.split('_')[1];
    const a = evalCommand(ctx,tag.params[0]);
    const b = evalCommand(ctx,tag.params[1]);

    switch (func){
        case 'lt':
            return Scalar.lt(a, b) ? 1 : 0;
        case 'gt':
            return Scalar.gt(a, b) ? 1 : 0;
        case 'eq':
            return Scalar.eq(a, b) ? 1 : 0;
        default:
            throw new Error(`Invalid bitwise operation ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function eval_loadScalar(ctx, tag){
    checkParams(ctx, tag, 1);
    return evalCommand(ctx,tag.params[0]);
}
*/

// Will be replaced by hardcoding this address directly in the ROM once the CONST register can be 256 bits
/*function eval_getGlobalExitRootManagerAddr(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    return scalar2fea(ctx.Fr, Scalar.e(ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2));
}*/

void eval_getGlobalExitRootManagerAddr(Context &ctx, const RomCommand &cmd, CommandResult &cr)
{
    // Check parameters list size
    if (cmd.params.size() != 0) {
        cerr << "Error: eval_getGlobalExitRootManagerAddr() invalid number of parameters function " << cmd.funcName << " : " << ctx.zkPC << endl;
        exit(-1);
    }

    // Return ctx.input.publicInputs.oldLocalExitRoot as a field element array
    cr.type = crt_fea;
    mpz_class globalExitRootManagerAddr(ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2);
    scalar2fea(ctx.fr, globalExitRootManagerAddr, cr.fea0, cr.fea1, cr.fea2, cr.fea3, cr.fea4, cr.fea5, cr.fea6, cr.fea7);
}

/*
function eval_storeLog(ctx, tag){
    checkParams(ctx, tag, 3);

    const indexLog = evalCommand(ctx, tag.params[0]);
    const isTopic = evalCommand(ctx, tag.params[1]);
    const data = evalCommand(ctx, tag.params[2]);

    if (typeof ctx.outLogs[indexLog] === "undefined"){
        ctx.outLogs[indexLog] = {
            data: [],
            topics: []
        }
    }

    if (isTopic) {
        ctx.outLogs[indexLog].topics.push(data.toString(16));
    } else {
        ctx.outLogs[indexLog].data.push(data.toString(16));
    }

    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_log(ctx, tag) {
    const frLog = ctx[tag.params[0].regName];
    const scalarLog = fea2scalar(ctx.Fr, frLog);
    const hexLog = `0x${scalarLog.toString(16)}`
    console.log(`Log regname ${tag.params[0].regName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`);
    console.log({scalarLog})
    console.log({hexLog})
    return scalar2fea(ctx.Fr, Scalar.e(0));
}

function eval_precompiled(ctx, tag){
    const primaryFunc = tag.funcName.split('_')[1];
    const secondaryFunc = tag.funcName.split('_')[2];

    switch (primaryFunc){
        case 'ecrecover':
            return preEcrecover(secondaryFunc, ctx, tag);
        case 'sha256':
            return preSha256(secondaryFunc, ctx, tag);
        case 'ripemd160':
            return preRipemd160(secondaryFunc, ctx, tag);
        case 'blake2f':
            return preBlake2f(secondaryFunc, ctx, tag);
        case 'ecAdd':
            return preEcAdd(secondaryFunc, ctx, tag);
        case 'ecMul':
            return preEcMul(secondaryFunc, ctx, tag);
        case 'ecPairing':
            return preEcPairing(secondaryFunc, ctx, tag);
        default:
            throw new Error(`Invalid precompiled operation ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preSha256(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.sha256 === 'undefined'){
        ctx.precompiled.sha256 = {
            data: "0x",
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 2);
            const size = Number(evalCommand(ctx, tag.params[1]));
            const bytesToAdd = (evalCommand(ctx, tag.params[0])).toString(16).padStart(2*size, "0");
            ctx.precompiled.sha256.data += bytesToAdd;

            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'digest':
            ctx.precompiled.sha256.result = ethers.utils.soliditySha256(["bytes"], [ctx.precompiled.sha256.data]);
            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'read':
            return scalar2fea(ctx.Fr, ctx.precompiled.sha256.result);

        default:
            throw new Error(`Invalid precompiled Sha256 functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preRipemd160(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ripemd160 === 'undefined'){
        ctx.precompiled.ripemd160 = {
            data: "0x",
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 2);
            const size = Number(evalCommand(ctx, tag.params[1]));
            const bytesToAdd = (evalCommand(ctx, tag.params[0])).toString(16).padStart(2*size, "0");
            ctx.precompiled.ripemd160.data += bytesToAdd;

            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'digest':
            const bytesToHash = ethers.utils.solidityPack(["bytes"], [ctx.precompiled.ripemd160.data]);
            ctx.precompiled.ripemd160.result = ethers.utils.ripemd160(bytesToHash);
            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'read':
            return scalar2fea(ctx.Fr, ctx.precompiled.ripemd160.result);

        default:
            throw new Error(`Invalid precompiled ripemd160 functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preBlake2f(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.blake2f === 'undefined'){
        ctx.precompiled.blake2f = {
            data: Buffer.alloc(0),
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 2);
            const size = Number(evalCommand(ctx, tag.params[1]));
            const bytesToAdd = (evalCommand(ctx, tag.params[0])).toString(16).padStart(2*size, "0");
            const buffToAdd = Buffer.from(bytesToAdd, "hex");
            ctx.precompiled.blake2f.data = Buffer.concat([ctx.precompiled.blake2f.data, buffToAdd]);

            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];

        case 'digest':
            // prepare data
            const rounds = ctx.precompiled.blake2f.data.slice(0, 4).readUInt32BE(0)
            const hRaw = ctx.precompiled.blake2f.data.slice(4, 68)
            const mRaw = ctx.precompiled.blake2f.data.slice(68, 196)
            const tRaw = ctx.precompiled.blake2f.data.slice(196, 212)
            // final
            const f = (ctx.precompiled.blake2f.data.slice(212, 213)[0]) === 1

            const h = new Uint32Array(16)
            for (let i = 0; i < 16; i++) {
                h[i] = hRaw.readUInt32LE(i * 4)
            }

            const m = new Uint32Array(32)
            for (let i = 0; i < 32; i++) {
                m[i] = mRaw.readUInt32LE(i * 4)
            }

            const t = new Uint32Array(4)
            for (let i = 0; i < 4; i++) {
                t[i] = tRaw.readUInt32LE(i * 4)
            }

            const resBlake = blake2f(h, m, t, f, rounds);

            ctx.precompiled.blake2f.result = [];
            ctx.precompiled.blake2f.result.push("0x" + resBlake.slice(0, 32).toString("hex"));
            ctx.precompiled.blake2f.result.push("0x" + resBlake.slice(32, 64).toString("hex"));

            return scalar2fea(ctx.Fr, Number(rounds));

        case 'read':
            checkParams(ctx, tag, 1);
            const chunk = Number(evalCommand(ctx, tag.params[0]));
            return scalar2fea(ctx.Fr, ctx.precompiled.blake2f.result[chunk]);

        default:
            throw new Error(`Invalid precompiled blake2f functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preEcAdd(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ecAdd === 'undefined'){
        ctx.precompiled.ecAdd = {
            data: "0x",
        }
    }

    switch (func){
        case 'add':
            checkParams(ctx, tag, 4);
            const input0 = "0x" + evalCommand(ctx, tag.params[0]).toString(16);
            const input1 = "0x" + evalCommand(ctx, tag.params[1]).toString(16);
            const input2 = "0x" + evalCommand(ctx, tag.params[2]).toString(16);
            const input3 = "0x" + evalCommand(ctx, tag.params[3]).toString(16);
            const buf0 = toBuffer(ethers.utils.hexZeroPad(input0, 32));
            const buf1 = toBuffer(ethers.utils.hexZeroPad(input1, 32));
            const buf2 = toBuffer(ethers.utils.hexZeroPad(input2, 32));
            const buf3 = toBuffer(ethers.utils.hexZeroPad(input3, 32));
            const buf = Buffer.concat([buf0, buf1, buf2, buf3]);
            ctx.precompiled.ecAdd.result = bn128.add(buf).toString("hex");
            ctx.precompiled.ecAdd.result0 = ctx.precompiled.ecAdd.result.substring(0,64);
            ctx.precompiled.ecAdd.result1 = ctx.precompiled.ecAdd.result.substring(64);
        case 'result0':
            if(!ctx.precompiled.ecAdd.result0)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecAdd.result0, 16));
        case 'result1':
            if(!ctx.precompiled.ecAdd.result1)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecAdd.result1, 16));
        default:
            throw new Error(`Invalid precompiled ecAdd functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preEcMul(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ecMul === 'undefined'){
        ctx.precompiled.ecMul = {
            data: "0x",
        }
    }

    switch (func){
        case 'mul':
            checkParams(ctx, tag, 3);
            const input0 = "0x" + evalCommand(ctx, tag.params[0]).toString(16);
            const input1 = "0x" + evalCommand(ctx, tag.params[1]).toString(16);
            const input2 = "0x" + evalCommand(ctx, tag.params[2]).toString(16);
            const buf0 = toBuffer(ethers.utils.hexZeroPad(input0, 32));
            const buf1 = toBuffer(ethers.utils.hexZeroPad(input1, 32));
            const buf2 = toBuffer(ethers.utils.hexZeroPad(input2, 32));
            const buf = Buffer.concat([buf0, buf1, buf2]);
            ctx.precompiled.ecMul.result = bn128.mul(buf).toString("hex");
            ctx.precompiled.ecMul.result0 = ctx.precompiled.ecMul.result.substring(0,64);
            ctx.precompiled.ecMul.result1 = ctx.precompiled.ecMul.result.substring(64);
        case 'result0':
            if(!ctx.precompiled.ecMul.result0)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecMul.result0, 16));
        case 'result1':
            if(!ctx.precompiled.ecMul.result1)
                throw new Error(`First _add`)
            else
                return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecMul.result1, 16));
        default:
            throw new Error(`Invalid precompiled ecMul functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function preEcPairing(func, ctx, tag){
    // initialize object
    if (typeof ctx.precompiled.ecPairing === 'undefined'){
        ctx.precompiled.ecPairing = {
            data: "0x",
        }
    }

    if (typeof ctx.precompiled.ecPairing.buf === 'undefined'){
        ctx.precompiled.ecPairing.buf = Buffer.from("");
    }

    switch (func){
        case 'pairing':
            ctx.precompiled.ecPairing.result = bn128.pairing(ctx.precompiled.ecPairing.buf).toString("hex");
            return scalar2fea(ctx.Fr, Scalar.fromString(ctx.precompiled.ecPairing.result, 16));
        case 'add':
            checkParams(ctx, tag, 1);
            const input0 = "0x" + evalCommand(ctx, tag.params[0]).toString(16);
            const buf0 = toBuffer(ethers.utils.hexZeroPad(input0, 32));
            ctx.precompiled.ecPairing.buf = Buffer.concat([ctx.precompiled.ecPairing.buf, buf0])
            return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
        default:
            throw new Error(`Invalid precompiled ecPairing functionality ${func}. ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`)
    }
}

function checkParams(ctx, tag, expectedParams){
    if (tag.params.length != expectedParams) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`);
}

function eval_dumpRegs(ctx, tag) {

    console.log(`dumpRegs ${ctx.fileName}:${ctx.line}`);

    console.log(['A', fea2scalar(ctx.Fr, ctx.A)]);
    console.log(['B', fea2scalar(ctx.Fr, ctx.B)]);
    console.log(['C', fea2scalar(ctx.Fr, ctx.C)]);
    console.log(['D', fea2scalar(ctx.Fr, ctx.D)]);
    console.log(['E', fea2scalar(ctx.Fr, ctx.E)]);

    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_dump(ctx, tag) {
    console.log("\x1b[38;2;175;175;255mDUMP on " + ctx.fileName + ":" + ctx.line+"\x1b[0m");

    tag.params.forEach((value) => {
        let name = value.varName || value.paramName || value.regName || value.offsetLabel;
        if (typeof name == 'undefined' && value.path) {
            name = value.path.join('.');
        }
        console.log("\x1b[35m"+ name +"\x1b[0;35m: "+evalCommand(ctx, value)+"\x1b[0m");
    });

    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_dumphex(ctx, tag) {
    console.log("\x1b[38;2;175;175;255mDUMP on " + ctx.fileName + ":" + ctx.line+"\x1b[0m");

    tag.params.forEach((value) => {
        let name = value.varName || value.paramName || value.regName;
        if (typeof name == 'undefined' && value.path) {
            name = value.path.join('.');
        }
        console.log("\x1b[35m"+ name +"\x1b[0;35m: 0x"+evalCommand(ctx, value).toString(16)+"\x1b[0m");
    });

    return [ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}
function eval_inverseFpEc(ctx, tag) {
    const a = evalCommand(ctx, tag.params[0]);
    if (ctx.Fec.isZero(a)) {
        throw new Error(`inverseFpEc: Division by zero  on: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`);
    }
    return ctx.Fec.inv(a);
}

function eval_inverseFnEc(ctx, tag) {
    const a = evalCommand(ctx, tag.params[0]);
    if (ctx.Fnec.isZero(a)) {
        throw new Error(`inverseFpEc: Division by zero  on: ${ctx.ln} at ${ctx.fileName}:${ctx.line}`);
    }
    return ctx.Fnec.inv(a);
}

function eval_sqrtFpEc(ctx, tag) {
    const a = evalCommand(ctx, tag.params[0]);
    console.log(a);
    const r = ctx.Fec.sqrt(a);
    console.log(r);
    return r;
}

function eval_xAddPointEc(ctx, tag) {
    return eval_AddPointEc(ctx, tag, false)[0];
}

function eval_yAddPointEc(ctx, tag) {
    return eval_AddPointEc(ctx, tag, false)[1];
}

function eval_xDblPointEc(ctx, tag) {
    return eval_AddPointEc(ctx, tag, true)[0];
}

function eval_yDblPointEc(ctx, tag) {
    return eval_AddPointEc(ctx, tag, true)[1];
}

function eval_AddPointEc(ctx, tag, dbl)
{
    const x1 = evalCommand(ctx, tag.params[0]);
    const y1 = evalCommand(ctx, tag.params[1]);
    const x2 = evalCommand(ctx, tag.params[dbl ? 0 : 2]);
    const y2 = evalCommand(ctx, tag.params[dbl ? 1 : 3]);

    console.log([x1,y1,x2,y2]);
    if (dbl) {
        // TODO: y1 == 0 => division by zero ==> how manage?
        s = ctx.Fec.div(ctx.Fec.mul(3n, ctx.Fec.mul(x1, x1)), ctx.Fec.add(y1, y1));
    }
    else {
        let deltaX = ctx.Fec.sub(x2, x1)
        // TODO: deltaX == 0 => division by zero ==> how manage?
        s = ctx.Fec.div(ctx.Fec.sub(y2, y1), deltaX );
    }

    const x3 = ctx.Fec.sub(ctx.Fec.mul(s, s), ctx.Fec.add(x1, x2));
    const y3 = ctx.Fec.sub(ctx.Fec.mul(s, ctx.Fec.sub(x1,x3)), y1);
    console.log([x3, y3]);
    return [x3, y3];
}
*/