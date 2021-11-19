#include <iostream>
#include "rom.hpp"
#include "rom_command.hpp"

void loadRom(Context &ctx, json &romJson)
{

    // Get size of ROM JSON file array
    if (!romJson.is_array())
    {
        cerr << "Error: ROM JSON file content is not an array" << endl;
        exit(-1);
    }
    ctx.romSize = romJson.size();
    cout << "ROM size: " << ctx.romSize << endl;

    // Allocate romSize tRomLine's
    rom = (RomLine *)new RomLine[ctx.romSize];
    if (rom==NULL)
    {
        cerr << "Error: failed allocating ROM memory for " << ctx.romSize << " instructions" << endl;
        exit(-1);
    }

    // Parse all ROM insruction lines and store them in memory
    for (uint64_t i=0; i<ctx.romSize; i++)
    {
        json l = romJson[i];
        rom[i].fileName = l["fileName"];
        rom[i].line = l["line"];
        cout << "Instruction " << i << " fileName:" << rom[i].fileName << " line:" << rom[i].line << endl;

        parseRomCommandArray(rom[i].cmdBefore, l["cmdBefore"]);
        parseRomCommandArray(rom[i].cmdAfter, l["cmdAfter"]);
        parseRomCommand(rom[i].freeInTag, l["freeInTag"]);

        if (l["CONST"].is_number_unsigned())
        {
            rom[i].bConstPresent = true;
            rom[i].CONST = l["CONST"];
        }
        else
        {
            rom[i].bConstPresent = false;
        }

        if (l["offset"].is_number_integer())
        {
            rom[i].bOffsetPresent = true;
            rom[i].offset = l["offset"];
        }
        else
        {
            rom[i].bOffsetPresent = false;
        }

        rom[i].inA          = (l["inA"] == 1)           ? 1 : 0;
        rom[i].inB          = (l["inB"] == 1)           ? 1 : 0;
        rom[i].inC          = (l["inC"] == 1)           ? 1 : 0;
        rom[i].inD          = (l["inD"] == 1)           ? 1 : 0;
        rom[i].inE          = (l["inE"] == 1)           ? 1 : 0;
        rom[i].inSR         = (l["inSR"] == 1)          ? 1 : 0;
        rom[i].inCTX        = (l["inCTX"] == 1)         ? 1 : 0;
        rom[i].inSP         = (l["inSP"] == 1)          ? 1 : 0;
        rom[i].inPC         = (l["inPC"] == 1)          ? 1 : 0;
        rom[i].inGAS        = (l["inGAS"] == 1)         ? 1 : 0;
        rom[i].inMAXMEM     = (l["inMAXMEM"] == 1)      ? 1 : 0;
        rom[i].inSTEP       = (l["inSTEP"] == 1)        ? 1 : 0;
        rom[i].mRD          = (l["mRD"] == 1)           ? 1 : 0;
        rom[i].mWR          = (l["mWR"] == 1)           ? 1 : 0;
        rom[i].hashRD       = (l["hashRD"] == 1)        ? 1 : 0;
        rom[i].hashWR       = (l["hashWR"] == 1)        ? 1 : 0;
        rom[i].hashE        = (l["hashE"] == 1)         ? 1 : 0;
        rom[i].JMP          = (l["JMP"] == 1)           ? 1 : 0;
        rom[i].JMPC         = (l["JMPC"] == 1)          ? 1 : 0;
        rom[i].useCTX       = (l["useCTX"] == 1)        ? 1 : 0;
        rom[i].isCode       = (l["isCode"] == 1)        ? 1 : 0;
        rom[i].isStack      = (l["isStack"] == 1)       ? 1 : 0;
        rom[i].isMem        = (l["isMem"] == 1)         ? 1 : 0;
        rom[i].inc          = (l["inc"] == 1)           ? 1 : 0;
        rom[i].dec          = (l["dec"] == 1)           ? 1 : 0;
        rom[i].ind          = (l["ind"] == 1)           ? 1 : 0;
        rom[i].inFREE       = (l["inFREE"] == 1)        ? 1 : 0;
        rom[i].ecRecover    = (l["ecRecover"] == 1)     ? 1 : 0;
        rom[i].shl          = (l["shl"] == 1)           ? 1 : 0;
        rom[i].shr          = (l["shr"] == 1)           ? 1 : 0;
        rom[i].neg          = (l["neg"] == 1)           ? 1 : 0;
        rom[i].assert       = (l["assert"] == 1)        ? 1 : 0;
        rom[i].setA         = (l["setA"] == 1)          ? 1 : 0;
        rom[i].setB         = (l["setB"] == 1)          ? 1 : 0;
        rom[i].setC         = (l["setC"] == 1)          ? 1 : 0;
        rom[i].setD         = (l["setD"] == 1)          ? 1 : 0;
        rom[i].setE         = (l["setE"] == 1)          ? 1 : 0;
        rom[i].setSR        = (l["setSR"] == 1)         ? 1 : 0;
        rom[i].setCTX       = (l["setCTX"] == 1)        ? 1 : 0;
        rom[i].setSP        = (l["setSP"] == 1)         ? 1 : 0;
        rom[i].setPC        = (l["setPC"] == 1)         ? 1 : 0;
        rom[i].setGAS       = (l["setGAS"] == 1)        ? 1 : 0;
        rom[i].setMAXMEM    = (l["setMAXMEM"] == 1)     ? 1 : 0;
        rom[i].sRD          = (l["sRD"] == 1)           ? 1 : 0;
        rom[i].sWR          = (l["sWR"] == 1)           ? 1 : 0;
        rom[i].arith        = (l["arith"] == 1)         ? 1 : 0;
        rom[i].bin          = (l["bin"] == 1)           ? 1 : 0;
        rom[i].comparator   = (l["comparator"] == 1)    ? 1 : 0;
        rom[i].opcodeRomMap = (l["opcodeRomMap"] == 1)  ? 1 : 0;
    }
}

void unloadRom(Context &ctx)
{
    for (uint64_t i=0; i<ctx.romSize; i++)
    {
        freeRomCommandArray(rom[i].cmdBefore);
        freeRomCommand(rom[i].freeInTag);
        freeRomCommandArray(rom[i].cmdAfter);
    }
    delete[] rom;
}