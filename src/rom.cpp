#include <iostream>
#include "rom.hpp"
#include "rom_command.hpp"
#include "scalar.hpp"

void Rom::load(RawFr &fr, json &romJson)
{
    // Check that rom is null
    if (romData != NULL)
    {
        cerr << "Error: loadRom() called with romData!=NULL" << endl;
        exit(-1);
    }

    // Get size of ROM JSON file array
    if (!romJson.is_array())
    {
        cerr << "Error: ROM JSON file content is not an array" << endl;
        exit(-1);
    }
    romSize = romJson.size();
    cout << "ROM size: " << romSize << " lines" << endl;

    // Allocate romSize tRomLine's
    romData = (RomLine *)new RomLine[romSize];
    if (romData==NULL)
    {
        cerr << "Error: failed allocating ROM memory for " << romSize << " instructions" << endl;
        exit(-1);
    }

    // Parse all ROM insruction lines and store them in memory: every line #i into rom[i]
    for (uint64_t i=0; i<romSize; i++)
    {
        json l = romJson[i];
        romData[i].fileName = l["fileName"];
        romData[i].line = l["line"];
        //cout << "Instruction " << i << " fileName:" << romData[i].fileName << " line:" << romData[i].line << endl;

        parseRomCommandArray(romData[i].cmdBefore, l["cmdBefore"]);
        parseRomCommandArray(romData[i].cmdAfter, l["cmdAfter"]);
        parseRomCommand(romData[i].freeInTag, l["freeInTag"]);

        if (l["CONST"].is_string())
        {
            romData[i].bConstPresent = true;
            fr.fromString(romData[i].CONST, l["CONST"]);
        }
        else
        {
            romData[i].bConstPresent = false;
        }

        if (l["offset"].is_number_integer())
        {
            romData[i].bOffsetPresent = true;
            romData[i].offset = l["offset"];
        }
        else
        {
            romData[i].bOffsetPresent = false;
        }

        if (l["inA"].is_string()) fr.fromString(romData[i].inA, l["inA"]); else romData[i].inA = fr.zero();
        if (l["inB"].is_string()) fr.fromString(romData[i].inB, l["inB"]); else romData[i].inB = fr.zero();
        if (l["inC"].is_string()) fr.fromString(romData[i].inC, l["inC"]); else romData[i].inC = fr.zero();
        if (l["inD"].is_string()) fr.fromString(romData[i].inD, l["inD"]); else romData[i].inD = fr.zero();
        if (l["inE"].is_string()) fr.fromString(romData[i].inE, l["inE"]); else romData[i].inE = fr.zero();

        if (l["inSR"].is_string()) fr.fromString(romData[i].inSR, l["inSR"]); else romData[i].inSR = fr.zero();
        if (l["inCTX"].is_string()) fr.fromString(romData[i].inCTX, l["inCTX"]); else romData[i].inCTX = fr.zero();
        if (l["inSP"].is_string()) fr.fromString(romData[i].inSP, l["inSP"]); else romData[i].inSP = fr.zero();
        if (l["inPC"].is_string()) fr.fromString(romData[i].inPC, l["inPC"]); else romData[i].inPC = fr.zero();
        if (l["inGAS"].is_string()) fr.fromString(romData[i].inGAS, l["inGAS"]); else romData[i].inGAS = fr.zero();
        if (l["inMAXMEM"].is_string()) fr.fromString(romData[i].inMAXMEM, l["inMAXMEM"]); else romData[i].inMAXMEM = fr.zero();
        if (l["inSTEP"].is_string()) fr.fromString(romData[i].inSTEP, l["inSTEP"]); else romData[i].inSTEP = fr.zero();
        if (l["inFREE"].is_string()) fr.fromString(romData[i].inFREE, l["inFREE"]); else romData[i].inFREE = fr.zero();

        if (l["mRD"].is_number_integer()) romData[i].mRD = l["mRD"]; else romData[i].mRD = 0;
        if (l["mWR"].is_number_integer()) romData[i].mWR = l["mWR"]; else romData[i].mWR = 0;
        if (l["hashRD"].is_number_integer()) romData[i].hashRD = l["hashRD"]; else romData[i].hashRD = 0;
        if (l["hashWR"].is_number_integer()) romData[i].hashWR = l["hashWR"]; else romData[i].hashWR = 0;
        if (l["hashE"].is_number_integer()) romData[i].hashE = l["hashE"]; else romData[i].hashE = 0;

        if (l["JMP"].is_number_integer()) romData[i].JMP = l["JMP"]; else romData[i].JMP = 0;
        if (l["JMPC"].is_number_integer()) romData[i].JMPC = l["JMPC"]; else romData[i].JMPC = 0;
        if (l["useCTX"].is_number_integer()) romData[i].useCTX = l["useCTX"]; else romData[i].useCTX = 0;
        if (l["isCode"].is_number_integer()) romData[i].isCode = l["isCode"]; else romData[i].isCode = 0;
        if (l["isStack"].is_number_integer()) romData[i].isStack = l["isStack"]; else romData[i].isStack = 0;
        if (l["isMem"].is_number_integer()) romData[i].isMem = l["isMem"]; else romData[i].isMem = 0;

        if (l["incCode"].is_number_integer()) romData[i].incCode = l["incCode"]; else romData[i].incCode = 0;
        if (l["incStack"].is_number_integer()) romData[i].incStack = l["incStack"]; else romData[i].incStack = 0;
        if (l["ind"].is_number_integer()) romData[i].ind = l["ind"]; else romData[i].ind = 0;

        if (l["ecRecover"].is_number_integer()) romData[i].ecRecover = l["ecRecover"]; else romData[i].ecRecover = 0;
        if (l["shl"].is_number_integer()) romData[i].shl = l["shl"]; else romData[i].shl = 0;
        if (l["shr"].is_number_integer()) romData[i].shr = l["shr"]; else romData[i].shr = 0;
        if (l["assert"].is_number_integer()) romData[i].assert = l["assert"]; else romData[i].assert = 0;
        
        if (l["setA"].is_number_integer()) romData[i].setA = l["setA"]; else romData[i].setA = 0;
        if (l["setB"].is_number_integer()) romData[i].setB = l["setB"]; else romData[i].setB = 0;
        if (l["setC"].is_number_integer()) romData[i].setC = l["setC"]; else romData[i].setC = 0;
        if (l["setD"].is_number_integer()) romData[i].setD = l["setD"]; else romData[i].setD = 0;
        if (l["setE"].is_number_integer()) romData[i].setE = l["setE"]; else romData[i].setE = 0;

        if (l["setSR"].is_number_integer()) romData[i].setSR = l["setSR"]; else romData[i].setSR = 0;
        if (l["setCTX"].is_number_integer()) romData[i].setCTX = l["setCTX"]; else romData[i].setCTX = 0;
        if (l["setSP"].is_number_integer()) romData[i].setSP = l["setSP"]; else romData[i].setSP = 0;
        if (l["setPC"].is_number_integer()) romData[i].setPC = l["setPC"]; else romData[i].setPC = 0;
        if (l["setGAS"].is_number_integer()) romData[i].setGAS = l["setGAS"]; else romData[i].setGAS = 0;
        if (l["setMAXMEM"].is_number_integer()) romData[i].setMAXMEM = l["setMAXMEM"]; else romData[i].setMAXMEM = 0;
 
        if (l["sRD"].is_number_integer()) romData[i].sRD = l["sRD"]; else romData[i].sRD = 0;
        if (l["sWR"].is_number_integer()) romData[i].sWR = l["sWR"]; else romData[i].sWR = 0;
        if (l["arith"].is_number_integer()) romData[i].arith = l["arith"]; else romData[i].arith = 0;
        if (l["bin"].is_number_integer()) romData[i].bin = l["bin"]; else romData[i].bin = 0;
        if (l["comparator"].is_number_integer()) romData[i].comparator = l["comparator"]; else romData[i].comparator = 0;
        if (l["opcodeRomMap"].is_number_integer()) romData[i].opcodeRomMap = l["opcodeRomMap"]; else romData[i].opcodeRomMap = 0;
    }
}

void Rom::unload(void)
{
    for (uint64_t i=0; i<romSize; i++)
    {
        freeRomCommandArray(romData[i].cmdBefore);
        freeRomCommand(romData[i].freeInTag);
        freeRomCommandArray(romData[i].cmdAfter);
    }
    delete[] romData;
    romData = NULL;
}