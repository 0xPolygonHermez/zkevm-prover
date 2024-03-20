#include <vector>
#include <iostream>
#include <string>
#include "main_sm/fork_6/main/rom_command.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

namespace fork_6
{

string RomCommand::toString (void) const
{
    string result;

    if (!isPresent) return "";

    if (op != op_empty) result += " op=" + op2String(op);
    if (function != f_empty) result += " funcName=" + function2String(function);
    if (varName.size() != 0) result += " varName=" + varName;
    if (reg != reg_empty) result += " regName=" + reg2string(reg);
    if (num != mpz_class(0)) result += " num=" + num.get_str(16);
    if (offset != 0) result += " offset=" + to_string(offset);

    for (uint64_t i=0; i<values.size(); i++)
    {
        result += " values[" + to_string(i) +"]={" + values[i]->toString() + " }";
    }

    for (uint64_t i=0; i<params.size(); i++)
    {
        result += " params[" + to_string(i) +"]={" + params[i]->toString() + " }";
    }

    return result;
}

tFunction string2Function(string s)
{
    if (s == "beforeLast")                          return f_beforeLast;
    else if (s == "getGlobalExitRoot")              return f_getGlobalExitRoot;
    else if (s == "getSequencerAddr")               return f_getSequencerAddr;
    else if (s == "getTimestamp")                   return f_getTimestamp;
    else if (s == "getTxs")                         return f_getTxs;
    else if (s == "getTxsLen")                      return f_getTxsLen;
    else if (s == "eventLog")                       return f_eventLog;
    else if (s == "cond")                           return f_cond;
    else if (s == "inverseFpEc")                    return f_inverseFpEc;
    else if (s == "inverseFnEc")                    return f_inverseFnEc;
    else if (s == "sqrtFpEc")                       return f_sqrtFpEc;
    else if (s == "sqrtFpEcParity")                 return f_sqrtFpEcParity;
    else if (s == "xAddPointEc")                    return f_xAddPointEc;
    else if (s == "yAddPointEc")                    return f_yAddPointEc;
    else if (s == "xDblPointEc")                    return f_xDblPointEc;
    else if (s == "yDblPointEc")                    return f_yDblPointEc;
    else if (s == "f_bitwise_and")                  return f_bitwise_and;
    else if (s == "f_bitwise_or")                   return f_bitwise_or;
    else if (s == "f_bitwise_xor")                  return f_bitwise_xor;
    else if (s == "f_bitwise_not")                  return f_bitwise_not;
    else if (s == "f_comp_lt")                      return f_comp_lt;
    else if (s == "f_comp_gt")                      return f_comp_gt;
    else if (s == "f_comp_eq")                      return f_comp_eq;
    else if (s == "loadScalar")                     return f_loadScalar;
    else if (s == "log")                            return f_log;
    else if (s == "exp")                            return f_exp;
    else if (s == "storeLog")                       return f_storeLog;
    else if (s == "memAlignWR_W0")                  return f_memAlignWR_W0;
    else if (s == "memAlignWR_W1")                  return f_memAlignWR_W1;
    else if (s == "memAlignWR8_W0")                 return f_memAlignWR8_W0;
    else if (s == "onOpcode")                       return f_onOpcode;
    else if (s == "onUpdateStorage")                return f_onUpdateStorage;
    else if (s == "")                               return f_empty;
    else {
        zklog.error("string2function() invalid string = " + s);
        exitProcess();
        return f_empty;
    }
}

string function2String(tFunction f) 
{
    switch (f)
    {
        case f_beforeLast:                      return "beforeLast";
        case f_getGlobalExitRoot:               return "getGlobalExitRoot";
        case f_getSequencerAddr:                return "getSequencerAddr";
        case f_getTimestamp:                    return "getTimestamp";
        case f_getTxs:                          return "getTxs";
        case f_getTxsLen:                       return "getTxsLen";
        case f_eventLog:                        return "eventLog";
        case f_cond:                            return "cond";
        case f_inverseFpEc:                     return "inverseFpEc";
        case f_inverseFnEc:                     return "inverseFnEc";
        case f_sqrtFpEc:                        return "sqrtFpEc";
        case f_sqrtFpEcParity:                  return "sqrtFpEcParity";
        case f_xAddPointEc:                     return "xAddPointEc";
        case f_yAddPointEc:                     return "yAddPointEc";
        case f_xDblPointEc:                     return "xDblPointEc";
        case f_yDblPointEc:                     return "yDblPointEc";
        case f_bitwise_and:                     return "bitwise_and";
        case f_bitwise_or:                      return "bitwise_or";
        case f_bitwise_xor:                     return "bitwise_xor";
        case f_bitwise_not:                     return "bitwise_not";
        case f_comp_lt:                         return "comp_lt";
        case f_comp_gt:                         return "comp_gt";
        case f_comp_eq:                         return "comp_eq";
        case f_loadScalar:                      return "loadScalar";
        case f_log:                             return "log";
        case f_exp:                             return "exp";
        case f_storeLog:                        return "storeLog";
        case f_memAlignWR_W0:                   return "memAlignWR_W0";
        case f_memAlignWR_W1:                   return "memAlignWR_W1";
        case f_memAlignWR8_W0:                  return "memAlignWR8_W0";
        case f_onOpcode:                        return "onOpcode";
        case f_onUpdateStorage:                 return "onUpdateStorage";
        case f_empty:                           return "";
        default:                                return "unknown";
    }
}

tOp string2Op(string s)
{
    if (s == "number")               return op_number;
    else if (s == "declareVar")      return op_declareVar;
    else if (s == "setVar")          return op_setVar;
    else if (s == "getVar")          return op_getVar;
    else if (s == "getReg")          return op_getReg;
    else if (s == "functionCall")    return op_functionCall;
    else if (s == "add")             return op_add;
    else if (s == "sub")             return op_sub;
    else if (s == "neg")             return op_neg;
    else if (s == "mul")             return op_mul;
    else if (s == "div")             return op_div;
    else if (s == "mod")             return op_mod;
    else if (s == "or")              return op_or;
    else if (s == "and")             return op_and;
    else if (s == "gt")              return op_gt;
    else if (s == "ge")              return op_ge;
    else if (s == "lt")              return op_lt;
    else if (s == "le")              return op_le;
    else if (s == "eq")              return op_eq;
    else if (s == "ne")              return op_ne;
    else if (s == "not")             return op_not;
    else if (s == "bitand")          return op_bitand;
    else if (s == "bitor")           return op_bitor;
    else if (s == "bitxor")          return op_bitxor;
    else if (s == "bitnot")          return op_bitnot;
    else if (s == "shl")             return op_shl;
    else if (s == "shr")             return op_shr;
    else if (s == "if")              return op_if;
    else if (s == "getMemValue")     return op_getMemValue;
    else if (s == "")                return op_empty;
    else {
        zklog.error("string2op() invalid string = " + s);
        exitProcess();
        return op_empty;
    }
}

string op2String(tOp op)
{
    switch (op)
    {
        case op_number:             return "number";
        case op_declareVar:         return  "declareVar";
        case op_setVar:             return  "setVar";
        case op_getVar:             return  "getVar";
        case op_getReg:             return  "getReg";
        case op_functionCall:       return  "functionCall";
        case op_add:                return  "add";
        case op_sub:                return  "sub";
        case op_neg:                return  "neg";
        case op_mul:                return  "mul";
        case op_div:                return  "div";
        case op_mod:                return  "mod";
        case op_or:                 return  "or";
        case op_and:                return  "and";
        case op_gt:                 return  "gt";
        case op_ge:                 return  "ge";
        case op_lt:                 return  "lt";
        case op_le:                 return  "le";
        case op_eq:                 return  "eq";
        case op_ne:                 return  "ne";
        case op_not:                return  "not";
        case op_bitand:             return  "bitand";
        case op_bitor:              return  "bitor";
        case op_bitxor:             return  "bitxor";
        case op_bitnot:             return  "bitnot";
        case op_shl:                return  "shl";
        case op_shr:                return  "shr";
        case op_if:                 return  "if";
        case op_getMemValue:        return  "getMemValue";
        case op_empty:              return  "";
        default:                    return  "unknown";  
    }
}

tReg string2reg(string s)
{
    if (s == "A") return reg_A;
    else if (s == "B") return reg_B;
    else if (s == "C") return reg_C;
    else if (s == "D") return reg_D;
    else if (s == "E") return reg_E;
    else if (s == "SR") return reg_SR;
    else if (s == "CTX") return reg_CTX;
    else if (s == "SP") return reg_SP;
    else if (s == "PC") return reg_PC;
    else if (s == "GAS") return reg_GAS;
    else if (s == "zkPC") return reg_zkPC;
    else if (s == "RR") return reg_RR;
    else if (s == "CNT_ARITH") return reg_CNT_ARITH;
    else if (s == "CNT_BINARY") return reg_CNT_BINARY;
    else if (s == "CNT_KECCAK_F") return reg_CNT_KECCAK_F;
    else if (s == "CNT_MEM_ALIGN") return reg_CNT_MEM_ALIGN;
    else if (s == "CNT_PADDING_PG") return reg_CNT_PADDING_PG;
    else if (s == "CNT_POSEIDON_G") return reg_CNT_POSEIDON_G;
    else if (s == "STEP") return reg_STEP;
    else if (s == "HASHPOS") return reg_HASHPOS;
    else {
        zklog.error("string2Reg() invalid string = " + s);
        exitProcess();
        return reg_A;
    }
}

string reg2string(tReg reg)
{
    switch (reg)
    {
        case reg_empty:             return "";
        case reg_A:                 return "A";
        case reg_B:                 return "B";
        case reg_C:                 return "C";
        case reg_D:                 return "D";
        case reg_E:                 return "E";
        case reg_SR:                return "SR";
        case reg_CTX:               return "CTX";
        case reg_SP:                return "SP";
        case reg_PC:                return "PC";
        case reg_GAS:               return "GAS";
        case reg_zkPC:              return "zkPC";
        case reg_RR:                return "RR";
        case reg_CNT_ARITH:         return "CNT_ARITH";
        case reg_CNT_BINARY:        return "CNT_BINARY";
        case reg_CNT_KECCAK_F:      return "CNT_KECCAK_F";
        case reg_CNT_MEM_ALIGN:     return "CNT_MEM_ALIGN";
        case reg_CNT_PADDING_PG:    return "CNT_PADDING_PG";
        case reg_CNT_POSEIDON_G:    return "CNT_POSEIDON_G";
        case reg_STEP:              return "STEP";
        case reg_HASHPOS:           return "HASHPOS";
        default:                    return "unknown";
    }
}

void parseRomCommand (RomCommand &cmd, json tag)
{
    // Skipt if not present
    if (tag.is_null())
    {
        cmd.isPresent = false;
        return;
    }
    cmd.isPresent = true;

    // This must be a ROM command, not an array of them
    if (tag.is_array())
    {
        zklog.error("parseRomCommand() found tag is an array: " + tag.dump());
        exitProcess();
    }

    // op is a mandatory element
    cmd.op = string2Op(tag["op"]);

    // Parse optional elements
    if (tag.contains("varName")) cmd.varName = tag["varName"];
    if (tag.contains("regName")) cmd.reg = string2reg(tag["regName"]);
    if (tag.contains("funcName")) cmd.function = string2Function (tag["funcName"]);
    if (tag.contains("num")) { string aux = tag["num"]; cmd.num.set_str(aux, 10); }
    if (tag.contains("offset") && tag["offset"].is_number()) { cmd.offset = tag["offset"]; }
    if (tag.contains("values")) parseRomCommandArray(cmd.values, tag["values"]);
    if (tag.contains("params")) parseRomCommandArray(cmd.params, tag["params"]);

    // Build opAndVar string to be used in time statistics
    cmd.opAndFunction = op2String(cmd.op) + "[" + function2String(cmd.function) + "]";
}

void parseRomCommandArray (vector<RomCommand *> &values, json tag)
{
    // Skip if not present
    if (tag.is_null())
    {
        return;
    }

    // This must be a ROM command array, not one of them
    if (!tag.is_array())
    {
        zklog.error("parseRomCommandArray() found tag is not an array: " + tag.dump());
        exitProcess();
    }

    // Parse every command in the array
    for (uint64_t i=0; i<tag.size(); i++)
    {
        RomCommand *pRomCommand = new RomCommand();
        parseRomCommand(*pRomCommand, tag[i]);
        values.push_back(pRomCommand);
    }
}

void freeRomCommand (RomCommand &cmd)
{
    // Fee the ROM command arrays content
    freeRomCommandArray(cmd.values);
    freeRomCommandArray(cmd.params);
}

void freeRomCommandArray (vector<RomCommand *> &array)
{
    // Free all ROM commands
    for (vector<class RomCommand *>::iterator it = array.begin(); it != array.end(); it++ )
    {
        freeRomCommand(**it);
        delete(*it);
    }

    // Empty the array
    array.clear();
}

} // namespace