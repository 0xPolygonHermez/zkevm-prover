#include <iostream>
#include "rom.hpp"
#include "rom_command.hpp"
#include "scalar.hpp"

void Rom::load(FiniteField &fr, json &romJson)
{
    // Check that rom is null
    if (line != NULL)
    {
        cerr << "Error: loadRom() called with line!=NULL" << endl;
        exit(-1);
    }

    // Get size of ROM JSON file array
    if (!romJson.is_array())
    {
        cerr << "Error: ROM JSON file content is not an array" << endl;
        exit(-1);
    }
    size = romJson.size();
    cout << "ROM size: " << size << " lines" << endl;

    // Allocate romSize tRomLine's
    line = (RomLine *)new RomLine[size];
    if (line==NULL)
    {
        cerr << "Error: failed allocating ROM memory for " << size << " instructions" << endl;
        exit(-1);
    }

    // Parse all ROM insruction lines and store them in memory: every line #i into rom[i]
    for (uint64_t i=0; i<size; i++)
    {
        json l = romJson[i];
        line[i].fileName = l["fileName"];
        line[i].line = l["line"];
        //cout << "Instruction " << i << " fileName:" << line[i].fileName << " line:" << line[i].line << endl;

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

        if (l["offset"].is_number_integer())
        {
            line[i].bOffsetPresent = true;
            line[i].offset = l["offset"];
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
        if (l["inMAXMEM"].is_string()) fr.fromString(line[i].inMAXMEM, l["inMAXMEM"]); else line[i].inMAXMEM = fr.zero();
        if (l["inSTEP"].is_string()) fr.fromString(line[i].inSTEP, l["inSTEP"]); else line[i].inSTEP = fr.zero();
        if (l["inFREE"].is_string()) fr.fromString(line[i].inFREE, l["inFREE"]); else line[i].inFREE = fr.zero();
        if (l["inRR"].is_string()) fr.fromString(line[i].inRR, l["inRR"]); else line[i].inRR = fr.zero();
        if (l["inHASHPOS"].is_string()) fr.fromString(line[i].inHASHPOS, l["inHASHPOS"]); else line[i].inHASHPOS = fr.zero();

        if (l["mRD"].is_number_integer()) line[i].mRD = l["mRD"]; else line[i].mRD = 0;
        if (l["mWR"].is_number_integer()) line[i].mWR = l["mWR"]; else line[i].mWR = 0;
        if (l["hashK"].is_number_integer()) line[i].hashK = l["hashK"]; else line[i].hashK = 0;
        if (l["hashKLen"].is_number_integer()) line[i].hashKLen = l["hashKLen"]; else line[i].hashKLen = 0;
        if (l["hashKDigest"].is_number_integer()) line[i].hashKDigest = l["hashKDigest"]; else line[i].hashKDigest = 0;
        if (l["hashP"].is_number_integer()) line[i].hashP = l["hashP"]; else line[i].hashP = 0;
        if (l["hashPLen"].is_number_integer()) line[i].hashPLen = l["hashPLen"]; else line[i].hashPLen = 0;
        if (l["hashPDigest"].is_number_integer()) line[i].hashPDigest = l["hashPDigest"]; else line[i].hashPDigest = 0;

        if (l["JMP"].is_number_integer()) line[i].JMP = l["JMP"]; else line[i].JMP = 0;
        if (l["JMPC"].is_number_integer()) line[i].JMPC = l["JMPC"]; else line[i].JMPC = 0;
        if (l["useCTX"].is_number_integer()) line[i].useCTX = l["useCTX"]; else line[i].useCTX = 0;
        if (l["isCode"].is_number_integer()) line[i].isCode = l["isCode"]; else line[i].isCode = 0;
        if (l["isStack"].is_number_integer()) line[i].isStack = l["isStack"]; else line[i].isStack = 0;
        if (l["isMem"].is_number_integer()) line[i].isMem = l["isMem"]; else line[i].isMem = 0;

        if (l["incCode"].is_number_integer()) line[i].incCode = l["incCode"]; else line[i].incCode = 0;
        if (l["incStack"].is_number_integer()) line[i].incStack = l["incStack"]; else line[i].incStack = 0;
        if (l["ind"].is_number_integer()) line[i].ind = l["ind"]; else line[i].ind = 0;

        if (l["ecRecover"].is_number_integer()) line[i].ecRecover = l["ecRecover"]; else line[i].ecRecover = 0;
        if (l["shl"].is_number_integer()) line[i].shl = l["shl"]; else line[i].shl = 0;
        if (l["shr"].is_number_integer()) line[i].shr = l["shr"]; else line[i].shr = 0;
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
        if (l["setMAXMEM"].is_number_integer()) line[i].setMAXMEM = l["setMAXMEM"]; else line[i].setMAXMEM = 0;
        if (l["setRR"].is_number_integer()) line[i].setRR = l["setRR"]; else line[i].setRR = 0;
        if (l["setHASHPOS"].is_number_integer()) line[i].setHASHPOS = l["setHASHPOS"]; else line[i].setHASHPOS = 0;
 
        if (l["sRD"].is_number_integer()) line[i].sRD = l["sRD"]; else line[i].sRD = 0;
        if (l["sWR"].is_number_integer()) line[i].sWR = l["sWR"]; else line[i].sWR = 0;
        if (l["arith"].is_number_integer()) line[i].arith = l["arith"]; else line[i].arith = 0;
        if (l["arithEq0"].is_number_integer()) line[i].arithEq0 = l["arithEq0"]; else line[i].arithEq0 = 0;
        if (l["arithEq1"].is_number_integer()) line[i].arithEq1 = l["arithEq1"]; else line[i].arithEq1 = 0;
        if (l["arithEq2"].is_number_integer()) line[i].arithEq2 = l["arithEq2"]; else line[i].arithEq2 = 0;
        if (l["arithEq3"].is_number_integer()) line[i].arithEq3 = l["arithEq3"]; else line[i].arithEq3 = 0;
        if (l["bin"].is_number_integer()) line[i].bin = l["bin"]; else line[i].bin = 0;
        if (l["binOpcode"].is_number_integer()) line[i].binOpcode = l["binOpcode"]; else line[i].binOpcode = 0;
        if (l["comparator"].is_number_integer()) line[i].comparator = l["comparator"]; else line[i].comparator = 0;
        if (l["opcodeRomMap"].is_number_integer()) line[i].opcodeRomMap = l["opcodeRomMap"]; else line[i].opcodeRomMap = 0;
    }
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