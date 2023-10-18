#include <iostream>
#include "main_sm/fork_6/main/rom.hpp"
#include "main_sm/fork_6/main/rom_command.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "zklog.hpp"

namespace fork_6
{

void Rom::load(Goldilocks &fr, json &romJson)
{
    // Load ROM program
    if (!romJson.contains("program"))
    {
        zklog.error("Rom::load() could not find program in rom json");
        exitProcess();
    }
    loadProgram(fr, romJson["program"]);

    // Load ROM labels
    if (!romJson.contains("labels"))
    {
        zklog.error("Rom::load() could not find labels in rom json");
        exitProcess();
    }
    loadLabels(fr, romJson["labels"]);

    // Get labels offsets
    if (config.dontLoadRomOffsets == false)
    {
        memLengthOffset              = getMemoryOffset("memLength");
        txDestAddrOffset             = getMemoryOffset("txDestAddr");
        txCalldataLenOffset          = getMemoryOffset("txCalldataLen");
        txGasLimitOffset             = getMemoryOffset("txGasLimit");
        txValueOffset                = getMemoryOffset("txValue");
        txNonceOffset                = getMemoryOffset("txNonce");
        txGasPriceOffset             = getMemoryOffset("txGasPrice");
        txGasPriceRLPOffset          = getMemoryOffset("txGasPriceRLP");
        txChainIdOffset              = getMemoryOffset("txChainId");
        txROffset                    = getMemoryOffset("txR");
        txSOffset                    = getMemoryOffset("txS");
        txVOffset                    = getMemoryOffset("txV");
        txSrcOriginAddrOffset        = getMemoryOffset("txSrcOriginAddr");
        retDataCTXOffset             = getMemoryOffset("retDataCTX");
        retDataOffsetOffset          = getMemoryOffset("retDataOffset");
        retDataLengthOffset          = getMemoryOffset("retDataLength");
        newAccInputHashOffset        = getMemoryOffset("newAccInputHash");
        oldNumBatchOffset            = getMemoryOffset("oldNumBatch");
        newNumBatchOffset            = getMemoryOffset("newNumBatch");
        newLocalExitRootOffset       = getMemoryOffset("newLocalExitRoot");
        depthOffset                  = getMemoryOffset("depth");
        gasRefundOffset              = getMemoryOffset("gasRefund");
        txSrcAddrOffset              = getMemoryOffset("txSrcAddr");
        gasCallOffset                = getMemoryOffset("gasCall");
        isPreEIP155Offset            = getMemoryOffset("isPreEIP155");
        isCreateContractOffset       = getMemoryOffset("isCreateContract");
        storageAddrOffset            = getMemoryOffset("storageAddr");
        bytecodeLengthOffset         = getMemoryOffset("bytecodeLength");
        originCTXOffset              = getMemoryOffset("originCTX");
        currentCTXOffset             = getMemoryOffset("currentCTX");
        gasCTXOffset                 = getMemoryOffset("gasCTX");
        lastCtxUsedOffset            = getMemoryOffset("lastCtxUsed");
        isCreateOffset               = getMemoryOffset("isCreate");
        effectivePercentageRLPOffset = getMemoryOffset("effectivePercentageRLP");
        calldataCTXOffset            = getMemoryOffset("calldataCTX");
        calldataOffsetOffset         = getMemoryOffset("calldataOffset");
    }

    // Load ROM integer constants
    constants.BATCH_DIFFICULTY                  = getConstant(romJson, "BATCH_DIFFICULTY");
    constants.TX_GAS_LIMIT                      = getConstant(romJson, "TX_GAS_LIMIT");
    constants.GLOBAL_EXIT_ROOT_STORAGE_POS      = getConstant(romJson, "GLOBAL_EXIT_ROOT_STORAGE_POS");
    constants.LOCAL_EXIT_ROOT_STORAGE_POS       = getConstant(romJson, "LOCAL_EXIT_ROOT_STORAGE_POS");
    constants.LAST_TX_STORAGE_POS               = getConstant(romJson, "LAST_TX_STORAGE_POS");
    constants.STATE_ROOT_STORAGE_POS            = getConstant(romJson, "STATE_ROOT_STORAGE_POS");
    constants.MAX_MEM_EXPANSION_BYTES           = getConstant(romJson, "MAX_MEM_EXPANSION_BYTES");
    constants.FORK_ID                           = getConstant(romJson, "FORK_ID");
    constants.MIN_VALUE_SHORT                   = getConstant(romJson, "MIN_VALUE_SHORT");
    constants.MIN_BYTES_LONG                    = getConstant(romJson, "MIN_BYTES_LONG");
    constants.SMT_KEY_BALANCE                   = getConstant(romJson, "SMT_KEY_BALANCE");
    constants.SMT_KEY_NONCE                     = getConstant(romJson, "SMT_KEY_NONCE");
    constants.SMT_KEY_SC_CODE                   = getConstant(romJson, "SMT_KEY_SC_CODE");
    constants.SMT_KEY_SC_STORAGE                = getConstant(romJson, "SMT_KEY_SC_STORAGE");
    constants.SMT_KEY_SC_LENGTH                 = getConstant(romJson, "SMT_KEY_SC_LENGTH");
    constants.SMT_KEY_TOUCHED_ADDR              = getConstant(romJson, "SMT_KEY_TOUCHED_ADDR");
    constants.SMT_KEY_TOUCHED_SLOTS             = getConstant(romJson, "SMT_KEY_TOUCHED_SLOTS");
    constants.BASE_TX_GAS                       = getConstant(romJson, "BASE_TX_GAS");
    constants.BASE_TX_DEPLOY_GAS                = getConstant(romJson, "BASE_TX_DEPLOY_GAS");
    constants.SLOAD_GAS                         = getConstant(romJson, "SLOAD_GAS");
    constants.GAS_QUICK_STEP                    = getConstant(romJson, "GAS_QUICK_STEP");
    constants.GAS_FASTEST_STEP                  = getConstant(romJson, "GAS_FASTEST_STEP");
    constants.GAS_FAST_STEP                     = getConstant(romJson, "GAS_FAST_STEP");
    constants.GAS_MID_STEP                      = getConstant(romJson, "GAS_MID_STEP");
    constants.GAS_SLOW_STEP                     = getConstant(romJson, "GAS_SLOW_STEP");
    constants.GAS_EXT_STEP                      = getConstant(romJson, "GAS_EXT_STEP");
    constants.CALL_VALUE_TRANSFER_GAS           = getConstant(romJson, "CALL_VALUE_TRANSFER_GAS");
    constants.CALL_NEW_ACCOUNT_GAS              = getConstant(romJson, "CALL_NEW_ACCOUNT_GAS");
    constants.CALL_STIPEND                      = getConstant(romJson, "CALL_STIPEND");
    constants.ECRECOVER_GAS                     = getConstant(romJson, "ECRECOVER_GAS");
    constants.IDENTITY_GAS                      = getConstant(romJson, "IDENTITY_GAS");
    constants.IDENTITY_WORD_GAS                 = getConstant(romJson, "IDENTITY_WORD_GAS");
    constants.KECCAK_GAS                        = getConstant(romJson, "KECCAK_GAS");
    constants.KECCAK_WORD_GAS                   = getConstant(romJson, "KECCAK_WORD_GAS");
    constants.LOG_GAS                           = getConstant(romJson, "LOG_GAS");
    constants.LOG_TOPIC_GAS                     = getConstant(romJson, "LOG_TOPIC_GAS");
    constants.JUMP_DEST_GAS                     = getConstant(romJson, "JUMP_DEST_GAS");
    constants.WARM_STORGE_READ_GAS              = getConstant(romJson, "WARM_STORGE_READ_GAS");
    constants.COLD_ACCOUNT_ACCESS_COST_REDUCED  = getConstant(romJson, "COLD_ACCOUNT_ACCESS_COST_REDUCED");
    constants.COLD_ACCOUNT_ACCESS_COST          = getConstant(romJson, "COLD_ACCOUNT_ACCESS_COST");
    constants.EXP_BYTE_GAS                      = getConstant(romJson, "EXP_BYTE_GAS");
    constants.RETURN_GAS_COST                   = getConstant(romJson, "RETURN_GAS_COST");
    constants.CREATE_GAS                        = getConstant(romJson, "CREATE_GAS");
    constants.CREATE_2_GAS                      = getConstant(romJson, "CREATE_2_GAS");
    constants.SENDALL_GAS                       = getConstant(romJson, "SENDALL_GAS");
    constants.LOG_DATA_GAS                      = getConstant(romJson, "LOG_DATA_GAS");
    constants.SSTORE_ENTRY_EIP_2200_GAS         = getConstant(romJson, "SSTORE_ENTRY_EIP_2200_GAS");
    constants.SSTORE_SET_EIP_2200_GAS           = getConstant(romJson, "SSTORE_SET_EIP_2200_GAS");
    constants.COLD_SLOAD_COST                   = getConstant(romJson, "COLD_SLOAD_COST");
    constants.COLD_SLOAD_COST_REDUCED           = getConstant(romJson, "COLD_SLOAD_COST_REDUCED");
    constants.SSTORE_DYNAMIC_GAS                = getConstant(romJson, "SSTORE_DYNAMIC_GAS");
    constants.SSTORE_SET_GAS                    = getConstant(romJson, "SSTORE_SET_GAS");
    constants.SSTORE_SET_GAS_REDUCED            = getConstant(romJson, "SSTORE_SET_GAS_REDUCED");
    constants.SSTORE_RESET_GAS                  = getConstant(romJson, "SSTORE_RESET_GAS");
    constants.SSTORE_RESET_GAS_REDUCED          = getConstant(romJson, "SSTORE_RESET_GAS_REDUCED");
    constants.SSTORE_CLEARS_SCHEDULE            = getConstant(romJson, "SSTORE_CLEARS_SCHEDULE");
    constants.MIN_STEPS_FINISH_BATCH            = getConstant(romJson, "MIN_STEPS_FINISH_BATCH");
    constants.TOTAL_STEPS_LIMIT                 = getConstant(romJson, "TOTAL_STEPS_LIMIT");
    constants.MAX_CNT_STEPS_LIMIT               = getConstant(romJson, "MAX_CNT_STEPS_LIMIT");
    constants.MAX_CNT_ARITH_LIMIT               = getConstant(romJson, "MAX_CNT_ARITH_LIMIT");
    constants.MAX_CNT_BINARY_LIMIT              = getConstant(romJson, "MAX_CNT_BINARY_LIMIT");
    constants.MAX_CNT_MEM_ALIGN_LIMIT           = getConstant(romJson, "MAX_CNT_MEM_ALIGN_LIMIT");
    constants.MAX_CNT_KECCAK_F_LIMIT            = getConstant(romJson, "MAX_CNT_KECCAK_F_LIMIT");
    constants.MAX_CNT_PADDING_PG_LIMIT          = getConstant(romJson, "MAX_CNT_PADDING_PG_LIMIT");
    constants.MAX_CNT_POSEIDON_G_LIMIT          = getConstant(romJson, "MAX_CNT_POSEIDON_G_LIMIT");
    constants.SAFE_RANGE                        = getConstant(romJson, "SAFE_RANGE");
    constants.MAX_CNT_STEPS                     = getConstant(romJson, "MAX_CNT_STEPS");
    constants.MAX_CNT_ARITH                     = getConstant(romJson, "MAX_CNT_ARITH");
    constants.MAX_CNT_BINARY                    = getConstant(romJson, "MAX_CNT_BINARY");
    constants.MAX_CNT_MEM_ALIGN                 = getConstant(romJson, "MAX_CNT_MEM_ALIGN");
    constants.MAX_CNT_KECCAK_F                  = getConstant(romJson, "MAX_CNT_KECCAK_F");
    constants.MAX_CNT_PADDING_PG                = getConstant(romJson, "MAX_CNT_PADDING_PG");
    constants.MAX_CNT_POSEIDON_G                = getConstant(romJson, "MAX_CNT_POSEIDON_G");
    constants.MAX_CNT_POSEIDON_SLOAD_SSTORE     = getConstant(romJson, "MAX_CNT_POSEIDON_SLOAD_SSTORE");
    constants.MIN_CNT_KECCAK_BATCH              = getConstant(romJson, "MIN_CNT_KECCAK_BATCH");
    constants.CODE_SIZE_LIMIT                   = getConstant(romJson, "CODE_SIZE_LIMIT");
    constants.BYTECODE_STARTS_EF                = getConstant(romJson, "BYTECODE_STARTS_EF");

    // Load ROM scalar constants
    constants.ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2 = getConstantL(romJson, "ADDRESS_GLOBAL_EXIT_ROOT_MANAGER_L2");
    constants.ADDRESS_SYSTEM                      = getConstantL(romJson, "ADDRESS_SYSTEM");
    constants.MAX_NONCE                           = getConstantL(romJson, "MAX_NONCE");
    constants.MAX_UINT_256                        = getConstantL(romJson, "MAX_UINT_256");
    constants.FPEC                                = getConstantL(romJson, "FPEC");
    constants.FPEC_MINUS_ONE                      = getConstantL(romJson, "FPEC_MINUS_ONE");
    constants.FNEC_DIV_TWO                        = getConstantL(romJson, "FNEC_DIV_TWO");
    constants.FPEC_C2_256                         = getConstantL(romJson, "FPEC_C2_256");
    constants.FPEC_NON_SQRT                       = getConstantL(romJson, "FPEC_NON_SQRT");
    constants.FNEC                                = getConstantL(romJson, "FNEC");
    constants.FNEC_MINUS_ONE                      = getConstantL(romJson, "FNEC_MINUS_ONE");
    constants.ECGX                                = getConstantL(romJson, "ECGX");
    constants.ECGY                                = getConstantL(romJson, "ECGY");
    constants.P2_160                              = getConstantL(romJson, "P2_160");
    constants.P2_96                               = getConstantL(romJson, "P2_96");
}

void Rom::loadProgram(Goldilocks &fr, json &romJson)
{
    // Check that rom is null
    if (line != NULL)
    {
        zklog.error("Rom::loadProgram() called with line!=NULL");
        exitProcess();
    }

    // Get size of ROM JSON file array
    if (!romJson.is_array())
    {
        zklog.error("Rom::loadProgram() ROM JSON file content is not an array");
        exitProcess();
    }
    size = romJson.size();
    zklog.info("ROM size: " + to_string(size) + " lines");

    // Allocate romSize tRomLine's
    line = (RomLine *)new RomLine[size];
    if (line==NULL)
    {
        zklog.error("Rom::loadProgram() failed allocating ROM memory for " + to_string(size) + " instructions");
        exitProcess();
    }

    // Parse all ROM insruction lines and store them in memory: every line #i into rom[i]
    for (uint64_t i=0; i<size; i++)
    {
        json l = romJson[i];
        string fileName = l["fileName"];

        size_t lastSlash = fileName.find_last_of("/");
        if (lastSlash == string::npos)
        {
            line[i].fileName = fileName;
        }
        else
        {
            line[i].fileName = fileName.substr(lastSlash+1);
        }

        line[i].line = l["line"];
        line[i].lineStr = l["lineStr"];

        parseRomCommandArray(line[i].cmdBefore, l["cmdBefore"]);
        parseRomCommandArray(line[i].cmdAfter, l["cmdAfter"]);
        parseRomCommand(line[i].freeInTag, l["freeInTag"]);
        
        if (l["CONST"].is_string())
        {
            line[i].bConstPresent = true;
            fr.fromString(line[i].CONST, l["CONST"]);
        }
        else
        {
            line[i].bConstPresent = false;
        }

        if (l["CONSTL"].is_string())
        {
            line[i].bConstLPresent = true;
            line[i].CONSTL.set_str(l["CONSTL"],10);
        }
        else
        {
            line[i].bConstLPresent = false;
        }

        if (l["jmpAddr"].is_number_unsigned())
        {
            line[i].bJmpAddrPresent = true;
            line[i].jmpAddr = fr.fromU64(l["jmpAddr"]);
        }
        else
        {
            line[i].bJmpAddrPresent = false;
        }

        if (l["elseAddr"].is_number_unsigned())
        {
            line[i].bElseAddrPresent = true;
            line[i].elseAddr = fr.fromU64(l["elseAddr"]);
            line[i].elseAddrLabel = l["elseAddrLabel"];
        }
        else
        {
            line[i].bElseAddrPresent = false;
        }

        if (l["offset"].is_number_integer())
        {
            line[i].bOffsetPresent = true;
            line[i].offset = l["offset"];
            if (l["offsetLabel"].is_string())
            {
                line[i].offsetLabel = l["offsetLabel"];
                memoryMap[line[i].offsetLabel] = line[i].offset;
            }
        }
        else
        {
            line[i].bOffsetPresent = false;
        }

        if (l["inA"].is_string()) fr.fromString(line[i].inA, l["inA"]); else line[i].inA = fr.zero();
        if (l["inB"].is_string()) fr.fromString(line[i].inB, l["inB"]); else line[i].inB = fr.zero();
        if (l["inC"].is_string()) fr.fromString(line[i].inC, l["inC"]); else line[i].inC = fr.zero();
        if (l["inD"].is_string()) fr.fromString(line[i].inD, l["inD"]); else line[i].inD = fr.zero();
        if (l["inE"].is_string()) fr.fromString(line[i].inE, l["inE"]); else line[i].inE = fr.zero();

        if (l["inSR"].is_string()) fr.fromString(line[i].inSR, l["inSR"]); else line[i].inSR = fr.zero();
        if (l["inCTX"].is_string()) fr.fromString(line[i].inCTX, l["inCTX"]); else line[i].inCTX = fr.zero();
        if (l["inSP"].is_string()) fr.fromString(line[i].inSP, l["inSP"]); else line[i].inSP = fr.zero();
        if (l["inPC"].is_string()) fr.fromString(line[i].inPC, l["inPC"]); else line[i].inPC = fr.zero();
        if (l["inGAS"].is_string()) fr.fromString(line[i].inGAS, l["inGAS"]); else line[i].inGAS = fr.zero();
        if (l["inSTEP"].is_string()) fr.fromString(line[i].inSTEP, l["inSTEP"]); else line[i].inSTEP = fr.zero();
        if (l["inFREE"].is_string()) fr.fromString(line[i].inFREE, l["inFREE"]); else line[i].inFREE = fr.zero();
        if (l["inRR"].is_string()) fr.fromString(line[i].inRR, l["inRR"]); else line[i].inRR = fr.zero();
        if (l["inHASHPOS"].is_string()) fr.fromString(line[i].inHASHPOS, l["inHASHPOS"]); else line[i].inHASHPOS = fr.zero();
        if (l["inCntArith"].is_string()) fr.fromString(line[i].inCntArith, l["inCntArith"]); else line[i].inCntArith = fr.zero();
        if (l["inCntBinary"].is_string()) fr.fromString(line[i].inCntBinary, l["inCntBinary"]); else line[i].inCntBinary = fr.zero();
        if (l["inCntMemAlign"].is_string()) fr.fromString(line[i].inCntMemAlign, l["inCntMemAlign"]); else line[i].inCntMemAlign = fr.zero();
        if (l["inCntKeccakF"].is_string()) fr.fromString(line[i].inCntKeccakF, l["inCntKeccakF"]); else line[i].inCntKeccakF = fr.zero();
        if (l["inCntPoseidonG"].is_string()) fr.fromString(line[i].inCntPoseidonG, l["inCntPoseidonG"]); else line[i].inCntPoseidonG = fr.zero();
        if (l["inCntPaddingPG"].is_string()) fr.fromString(line[i].inCntPaddingPG, l["inCntPaddingPG"]); else line[i].inCntPaddingPG = fr.zero();
        if (l["inROTL_C"].is_string()) fr.fromString(line[i].inROTL_C, l["inROTL_C"]); else line[i].inROTL_C = fr.zero();
        if (l["inRCX"].is_string()) fr.fromString(line[i].inRCX, l["inRCX"]); else line[i].inRCX = fr.zero();

        if (l["mOp"].is_number_integer()) line[i].mOp = l["mOp"]; else line[i].mOp = 0;
        if (l["mWR"].is_number_integer()) line[i].mWR = l["mWR"]; else line[i].mWR = 0;
        if (l["hashK"].is_number_integer()) line[i].hashK = l["hashK"]; else line[i].hashK = 0;
        if (l["hashK1"].is_number_integer()) line[i].hashK1 = l["hashK1"]; else line[i].hashK1 = 0;
        if (l["hashKLen"].is_number_integer()) line[i].hashKLen = l["hashKLen"]; else line[i].hashKLen = 0;
        if (l["hashKDigest"].is_number_integer()) line[i].hashKDigest = l["hashKDigest"]; else line[i].hashKDigest = 0;
        if (l["hashP"].is_number_integer()) line[i].hashP = l["hashP"]; else line[i].hashP = 0;
        if (l["hashP1"].is_number_integer()) line[i].hashP1 = l["hashP1"]; else line[i].hashP1 = 0;
        if (l["hashPLen"].is_number_integer()) line[i].hashPLen = l["hashPLen"]; else line[i].hashPLen = 0;
        if (l["hashPDigest"].is_number_integer()) line[i].hashPDigest = l["hashPDigest"]; else line[i].hashPDigest = 0;

        if (l["JMP"].is_number_integer()) line[i].JMP = l["JMP"]; else line[i].JMP = 0;
        if (l["JMPC"].is_number_integer()) line[i].JMPC = l["JMPC"]; else line[i].JMPC = 0;
        if (l["JMPN"].is_number_integer()) line[i].JMPN = l["JMPN"]; else line[i].JMPN = 0;
        if (l["JMPZ"].is_number_integer()) line[i].JMPZ = l["JMPZ"]; else line[i].JMPZ = 0;
        if (l["call"].is_number_integer()) line[i].call = l["call"]; else line[i].call = 0;
        if (l["return"].is_number_integer()) line[i].return_ = l["return"]; else line[i].return_ = 0;
        if (l["useJmpAddr"].is_number_integer()) line[i].useJmpAddr = l["useJmpAddr"]; else line[i].useJmpAddr = 0;
        if (l["useElseAddr"].is_number_integer()) line[i].useElseAddr = l["useElseAddr"]; else line[i].useElseAddr = 0;
        if (l["useCTX"].is_number_integer()) line[i].useCTX = l["useCTX"]; else line[i].useCTX = 0;
        if (l["isStack"].is_number_integer()) line[i].isStack = l["isStack"]; else line[i].isStack = 0;
        if (l["isMem"].is_number_integer()) line[i].isMem = l["isMem"]; else line[i].isMem = 0;

        if (l["incStack"].is_number_integer()) line[i].incStack = l["incStack"]; else line[i].incStack = 0;
        if (l["ind"].is_number_integer()) line[i].ind = l["ind"]; else line[i].ind = 0;
        if (l["indRR"].is_number_integer()) line[i].indRR = l["indRR"]; else line[i].indRR = 0;

        if (l["assert"].is_number_integer()) line[i].assert = l["assert"]; else line[i].assert = 0;
        
        if (l["setA"].is_number_integer()) line[i].setA = l["setA"]; else line[i].setA = 0;
        if (l["setB"].is_number_integer()) line[i].setB = l["setB"]; else line[i].setB = 0;
        if (l["setC"].is_number_integer()) line[i].setC = l["setC"]; else line[i].setC = 0;
        if (l["setD"].is_number_integer()) line[i].setD = l["setD"]; else line[i].setD = 0;
        if (l["setE"].is_number_integer()) line[i].setE = l["setE"]; else line[i].setE = 0;

        if (l["setSR"].is_number_integer()) line[i].setSR = l["setSR"]; else line[i].setSR = 0;
        if (l["setCTX"].is_number_integer()) line[i].setCTX = l["setCTX"]; else line[i].setCTX = 0;
        if (l["setSP"].is_number_integer()) line[i].setSP = l["setSP"]; else line[i].setSP = 0;
        if (l["setPC"].is_number_integer()) line[i].setPC = l["setPC"]; else line[i].setPC = 0;
        if (l["setGAS"].is_number_integer()) line[i].setGAS = l["setGAS"]; else line[i].setGAS = 0;
        if (l["setRR"].is_number_integer()) line[i].setRR = l["setRR"]; else line[i].setRR = 0;
        if (l["setHASHPOS"].is_number_integer()) line[i].setHASHPOS = l["setHASHPOS"]; else line[i].setHASHPOS = 0;
        if (l["setRCX"].is_number_integer()) line[i].setRCX = l["setRCX"]; else line[i].setRCX = 0;
 
        if (l["sRD"].is_number_integer()) line[i].sRD = l["sRD"]; else line[i].sRD = 0;
        if (l["sWR"].is_number_integer()) line[i].sWR = l["sWR"]; else line[i].sWR = 0;
        if (l["arithEq0"].is_number_integer()) line[i].arithEq0 = l["arithEq0"]; else line[i].arithEq0 = 0;
        if (l["arithEq1"].is_number_integer()) line[i].arithEq1 = l["arithEq1"]; else line[i].arithEq1 = 0;
        if (l["arithEq2"].is_number_integer()) line[i].arithEq2 = l["arithEq2"]; else line[i].arithEq2 = 0;
        if (l["bin"].is_number_integer()) line[i].bin = l["bin"]; else line[i].bin = 0;
        if (l["binOpcode"].is_number_integer()) line[i].binOpcode = l["binOpcode"]; else line[i].binOpcode = 0;
        if (l["memAlignRD"].is_number_integer()) line[i].memAlignRD = l["memAlignRD"]; else line[i].memAlignRD = 0;
        if (l["memAlignWR"].is_number_integer()) line[i].memAlignWR = l["memAlignWR"]; else line[i].memAlignWR = 0;
        if (l["memAlignWR8"].is_number_integer()) line[i].memAlignWR8 = l["memAlignWR8"]; else line[i].memAlignWR8 = 0;
        if (l["repeat"].is_number_integer()) line[i].repeat = l["repeat"]; else line[i].repeat = 0;
    }
}

void Rom::loadLabels(Goldilocks &fr, json &romJson)
{
    // Check that memoryMap is empty
    if (labels.size() != 0)
    {
        zklog.error("Rom::loadLabels() called with labels.size()=" + to_string(labels.size()));
        exitProcess();
    }

    // Check it is an object
    if (!romJson.is_object())
    {
        zklog.error("Rom::loadLabels() labels content is not an object");
        exitProcess();
    }

    json::const_iterator it;
    for (it = romJson.begin(); it != romJson.end(); it++)
    {
        if (!it.value().is_number())
        {
            zklog.error("Rom::loadLabels() labels value is not a number");
            exitProcess();
        }
        labels[it.key()] = it.value();
    }
}

uint64_t Rom::getLabel(const string &label) const
{
    unordered_map<string,uint64_t>::const_iterator it;
    it = labels.find(label);
    if (it==labels.end())
    {
        zklog.error("Rom::getLabel() could not find label=" + label);
        exitProcess();
    }
    return it->second;
}

uint64_t Rom::getMemoryOffset(const string &label) const
{
    unordered_map<string,uint64_t>::const_iterator it;
    it = memoryMap.find(label);
    if (it==memoryMap.end())
    {
        zklog.error("Rom::getMemoryOffset() could not find label=" + label);
        exitProcess();
    }
    return it->second;
}

uint64_t Rom::getConstant(json &romJson, const string &constantName)
{
    if (!romJson.contains("constants") ||
        !romJson["constants"].is_object())
    {
        zklog.error("Rom::getConstant() could not find constants in rom json");
        exitProcess();
    }
    if (!romJson["constants"].contains(constantName) ||
        !romJson["constants"][constantName].is_object() )
    {
        zklog.error("Rom::getConstant() could not find constant " + constantName + " in rom json");
        exitProcess();
    }
    if (!romJson["constants"][constantName].contains("value") ||
        !romJson["constants"][constantName]["value"].is_string() )
    {
        zklog.error("Rom::getConstant() could not find value for constant " + constantName + " in rom json");
        exitProcess();
    }
    string auxString;
    auxString = romJson["constants"][constantName]["value"];
    //cout << "Rom::getConstant() " << constantName << "=" << auxString << endl;
    return atoi(auxString.c_str());
}

mpz_class Rom::getConstantL(json &romJson, const string &constantName)
{
    if (!romJson.contains("constants") ||
        !romJson["constants"].is_object())
    {
        zklog.error("Rom::getConstantL() could not find constants in rom json");
        exitProcess();
    }
    if (!romJson["constants"].contains(constantName) ||
        !romJson["constants"][constantName].is_object() )
    {
        zklog.error("Rom::getConstantL() could not find constant " + constantName + " in rom json");
        exitProcess();
    }
    if (!romJson["constants"][constantName].contains("value") ||
        !romJson["constants"][constantName]["value"].is_string() )
    {
        zklog.error("Rom::getConstantL() could not find value for constant " + constantName + " in rom json");
        exitProcess();
    }
    string auxString;
    auxString = romJson["constants"][constantName]["value"];
    //cout << "Rom::getConstantL() " << constantName << "=" << auxString << endl;

    mpz_class value;
    value.set_str(auxString, 10);
    return value;
}

void Rom::unload(void)
{
    for (uint64_t i=0; i<size; i++)
    {
        freeRomCommandArray(line[i].cmdBefore);
        freeRomCommand(line[i].freeInTag);
        freeRomCommandArray(line[i].cmdAfter);
    }
    delete[] line;
    line = NULL;
}

} // namespace