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
    rom = (tRomLine *)new tRomLine[ctx.romSize];
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

        // cmdBefore
        parseRomCommandArray(l["cmdBefore"], rom[i].cmdBefore);

        // inXX elements
        if (l["inA"] == 1) rom[i].inA = true; // TODO: Should we store any in value, or just 1/true?
        if (l["inB"] == 1) rom[i].inB = true;
        if (l["inC"] == 1) rom[i].inC = true;
        if (l["inD"] == 1) rom[i].inD = true;
        if (l["inE"] == 1) rom[i].inE = true;
        if (l["inSR"] == 1) rom[i].inSR = true;
        if (l["inCTX"] == 1) rom[i].inCTX = true;
        if (l["inSP"] == 1) rom[i].inSP = true;
        if (l["inPC"] == 1) rom[i].inPC = true;
        if (l["inGAS"] == 1) rom[i].inGAS = true;
        if (l["inMAXMEM"] == 1) rom[i].inMAXMEM = true;
        if (l["inSTEP"] == 1) rom[i].inSTEP = true;
        
        // CONST element
        if (l["CONST"].is_number_unsigned())
        {
            rom[i].bConstPresent = true;
            rom[i].CONST = l["CONST"];
        }

        // Memory elements
        if (l["mRD"] == 1) rom[i].mRD = true;
        if (l["mWR"] == 1) rom[i].mWR = true;
        if (l["hashRD"] == 1) rom[i].hashRD = true;
        if (l["hashWR"] == 1) rom[i].hashWR = true;
        if (l["hashE"] == 1) rom[i].hashE = true;
        if (l["JMP"] == 1) rom[i].JMP = true;
        if (l["JMPC"] == 1) rom[i].JMPC = true;
        
        // Offset element
        if (l["offset"].is_number_integer())
        {
            rom[i].bOffsetPresent = true;
            rom[i].offset = l["offset"];
        }

        // Memory areas
        if (l["useCTX"] == 1) rom[i].useCTX = true;
        if (l["isCode"] == 1) rom[i].isCode = true;
        if (l["isStack"] == 1) rom[i].isStack = true;
        if (l["isMem"] == 1) rom[i].isMem = true;

        // Inc/Dec elements
        if (l["inc"] == 1) rom[i].inc = true;
        if (l["dec"] == 1) rom[i].dec = true;

        // Ind element
        if (l["ind"].is_number_integer())
        {
            rom[i].bIndPresent = true;
            rom[i].ind = l["ind"];
        }

        if (l["inFREE"] == 1) rom[i].inFREE = true;

        // freeInTag
        parseRomCommand(l["freeInTag"], rom[i].freeInTag);

        if (l["ecRecover"] == 1) rom[i].ecRecover = true;
        if (l["shl"] == 1) rom[i].shl = true;
        if (l["shr"] == 1) rom[i].shr = true;

        if (l["neg"] == 1) rom[i].neg = true;
        if (l["assert"] == 1) rom[i].assert = true;

        // setXX elements
        if (l["setA"] == 1) rom[i].setA = true;
        if (l["setB"] == 1) rom[i].setB = true;
        if (l["setC"] == 1) rom[i].setC = true;
        if (l["setD"] == 1) rom[i].setD = true;
        if (l["setE"] == 1) rom[i].setE = true;
        if (l["setSR"] == 1) rom[i].setSR = true;
        if (l["setCTX"] == 1) rom[i].setCTX = true;
        if (l["setSP"] == 1) rom[i].setSP = true;
        if (l["setPC"] == 1) rom[i].setPC = true;
        if (l["setGAS"] == 1) rom[i].setGAS = true;
        if (l["setMAXMEM"] == 1) rom[i].setMAXMEM = true;
        if (l["sRD"] == 1) rom[i].sRD = true;
        if (l["sWR"] == 1) rom[i].sWR = true;
        if (l["arith"] == 1) rom[i].arith = true;
        if (l["bin"] == 1) rom[i].bin = true;
        if (l["comparator"] == 1) rom[i].comparator = true;
        if (l["opcodeRomMap"] == 1) rom[i].opcodeRomMap = true;

        // cmdAfter
        parseRomCommandArray(l["cmdAfter"], rom[i].cmdAfter);

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