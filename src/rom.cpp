#include <iostream>
#include "rom.hpp"
#include "rom_command.hpp"

void Rom::loadRom(json &romJson)
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

        if (l["CONST"].is_number_integer())
        {
            romData[i].bConstPresent = true;
            romData[i].CONST = l["CONST"];
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

        romData[i].inA          = (l["inA"] == 1)           ? 1 : 0;
        romData[i].inB          = (l["inB"] == 1)           ? 1 : 0;
        romData[i].inC          = (l["inC"] == 1)           ? 1 : 0;
        romData[i].inD          = (l["inD"] == 1)           ? 1 : 0;
        romData[i].inE          = (l["inE"] == 1)           ? 1 : 0;
        romData[i].inSR         = (l["inSR"] == 1)          ? 1 : 0;
        romData[i].inCTX        = (l["inCTX"] == 1)         ? 1 : 0;
        romData[i].inSP         = (l["inSP"] == 1)          ? 1 : 0;
        romData[i].inPC         = (l["inPC"] == 1)          ? 1 : 0;
        romData[i].inGAS        = (l["inGAS"] == 1)         ? 1 : 0;
        romData[i].inMAXMEM     = (l["inMAXMEM"] == 1)      ? 1 : 0;
        romData[i].inSTEP       = (l["inSTEP"] == 1)        ? 1 : 0;
        romData[i].mRD          = (l["mRD"] == 1)           ? 1 : 0;
        romData[i].mWR          = (l["mWR"] == 1)           ? 1 : 0;
        romData[i].hashRD       = (l["hashRD"] == 1)        ? 1 : 0;
        romData[i].hashWR       = (l["hashWR"] == 1)        ? 1 : 0;
        romData[i].hashE        = (l["hashE"] == 1)         ? 1 : 0;
        romData[i].JMP          = (l["JMP"] == 1)           ? 1 : 0;
        romData[i].JMPC         = (l["JMPC"] == 1)          ? 1 : 0;
        romData[i].useCTX       = (l["useCTX"] == 1)        ? 1 : 0;
        romData[i].isCode       = (l["isCode"] == 1)        ? 1 : 0;
        romData[i].isStack      = (l["isStack"] == 1)       ? 1 : 0;
        romData[i].isMem        = (l["isMem"] == 1)         ? 1 : 0;
        romData[i].inc          = (l["inc"] == 1)           ? 1 : 0;
        romData[i].dec          = (l["dec"] == 1)           ? 1 : 0;
        romData[i].ind          = (l["ind"] == 1)           ? 1 : 0;
        romData[i].inFREE       = (l["inFREE"] == 1)        ? 1 : 0;
        romData[i].ecRecover    = (l["ecRecover"] == 1)     ? 1 : 0;
        romData[i].shl          = (l["shl"] == 1)           ? 1 : 0;
        romData[i].shr          = (l["shr"] == 1)           ? 1 : 0;
        romData[i].neg          = (l["neg"] == 1)           ? 1 : 0;
        romData[i].assert       = (l["assert"] == 1)        ? 1 : 0;
        romData[i].setA         = (l["setA"] == 1)          ? 1 : 0;
        romData[i].setB         = (l["setB"] == 1)          ? 1 : 0;
        romData[i].setC         = (l["setC"] == 1)          ? 1 : 0;
        romData[i].setD         = (l["setD"] == 1)          ? 1 : 0;
        romData[i].setE         = (l["setE"] == 1)          ? 1 : 0;
        romData[i].setSR        = (l["setSR"] == 1)         ? 1 : 0;
        romData[i].setCTX       = (l["setCTX"] == 1)        ? 1 : 0;
        romData[i].setSP        = (l["setSP"] == 1)         ? 1 : 0;
        romData[i].setPC        = (l["setPC"] == 1)         ? 1 : 0;
        romData[i].setGAS       = (l["setGAS"] == 1)        ? 1 : 0;
        romData[i].setMAXMEM    = (l["setMAXMEM"] == 1)     ? 1 : 0;
        romData[i].sRD          = (l["sRD"] == 1)           ? 1 : 0;
        romData[i].sWR          = (l["sWR"] == 1)           ? 1 : 0;
        romData[i].arith        = (l["arith"] == 1)         ? 1 : 0;
        romData[i].bin          = (l["bin"] == 1)           ? 1 : 0;
        romData[i].comparator   = (l["comparator"] == 1)    ? 1 : 0;
        romData[i].opcodeRomMap = (l["opcodeRomMap"] == 1)  ? 1 : 0;
    }
}

void Rom::unloadRom(void)
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