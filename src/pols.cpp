#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "pols.hpp"
#include "config.hpp"
#include "context.hpp"

uint64_t type2size (eElementType elementType)
{
    switch (elementType) {
        case et_bool:
        case et_s8:
        case et_u8: return 1;
        case et_s16:
        case et_u16: return 2;
        case et_s32:
        case et_u32: return 4;
        case et_s64:
        case et_u64: return 8;
        case et_field: return sizeof(RawFr::Element);
        case et_unknown:
        default:
            cerr << "Error: type2size() caled with invalid polynomial elementType: " << elementType << endl;
            exit(-1);
    }
}

eElementType string2et (const string &s)
{
    if (s=="bool") return et_bool;
    if (s=="s8") return et_s8;
    if (s=="u8") return et_u8;
    if (s=="s16") return et_s16;
    if (s=="u16") return et_u16;
    if (s=="s32") return et_s32;
    if (s=="u32") return et_u32;
    if (s=="s64") return et_s64;
    if (s=="u64") return et_u64;
    if (s=="field") return et_field;
    cerr << "Error: string2et() found unexpected element type: " << s << endl;
    exit(-1);
}

void Pols::load(const vector<PolJsonData> &v)
{
    // Check the polynomials are not mapped
    if (fileName.size() != 0)
    {
        cerr << "Error: Pols::load() called with an existing file name: " << fileName << endl;
        exit(-1);
    }

    // Reset orderedPols
    memset(&orderedPols, 0, sizeof(orderedPols));

    // Add one polynomial per vector entry
    for (uint64_t i=0; i<v.size(); i++)
    {
        addPol(v[i].name, v[i].id, v[i].elementType);
    }
    size = v.size();
}

void Pols::addPol(const string &name, const uint64_t id, const string &elementType)
{
    // Find the polynomial with this name
    Pol * pPol = find(name);

    // Check that the element type matches
    if ( string2et(elementType) != pPol->elementType )
    {
             cerr << "Error: addPol() found inconsistent element type for pol " << name << endl;
             exit(-1);
    }

    // Store name
    pPol->name = name;

    // Store ID
    pPol->id = id;

    // Check that the ordered slot is not already occupied
    if ( orderedPols[id] != NULL )
    {
        cerr << "Error: addPol() found id " << id << " has already been added" << endl;
        exit(-1);
    }

    // Keep a pointer to the polynomial in the right position of the ordered polynomials list
    orderedPols[id] = pPol;
}

Pol * Pols::find(const string &name)
{
    // Committed, output polynomials
         if (name=="main.A0")           return (Pol *)&A0;
    else if (name=="main.A1")           return (Pol *)&A1;
    else if (name=="main.A2")           return (Pol *)&A2;
    else if (name=="main.A3")           return (Pol *)&A3;
    else if (name=="main.B0")           return (Pol *)&B0;
    else if (name=="main.B1")           return (Pol *)&B1;
    else if (name=="main.B2")           return (Pol *)&B2;
    else if (name=="main.B3")           return (Pol *)&B3;
    else if (name=="main.C0")           return (Pol *)&C0;
    else if (name=="main.C1")           return (Pol *)&C1;
    else if (name=="main.C2")           return (Pol *)&C2;
    else if (name=="main.C3")           return (Pol *)&C3;
    else if (name=="main.D0")           return (Pol *)&D0;
    else if (name=="main.D1")           return (Pol *)&D1;
    else if (name=="main.D2")           return (Pol *)&D2;
    else if (name=="main.D3")           return (Pol *)&D3;
    else if (name=="main.E0")           return (Pol *)&E0;
    else if (name=="main.E1")           return (Pol *)&E1;
    else if (name=="main.E2")           return (Pol *)&E2;
    else if (name=="main.E3")           return (Pol *)&E3;
    else if (name=="main.FREE0")        return (Pol *)&FREE0;
    else if (name=="main.FREE1")        return (Pol *)&FREE1;
    else if (name=="main.FREE2")        return (Pol *)&FREE2;
    else if (name=="main.FREE3")        return (Pol *)&FREE3;
    else if (name=="main.CONST")        return (Pol *)&CONST;
    else if (name=="main.CTX")          return (Pol *)&CTX;
    else if (name=="main.GAS")          return (Pol *)&GAS;
    else if (name=="main.JMP")          return (Pol *)&JMP;
    else if (name=="main.JMPC")         return (Pol *)&JMPC;
    else if (name=="main.MAXMEM")       return (Pol *)&MAXMEM;
    else if (name=="main.PC")           return (Pol *)&PC;
    else if (name=="main.SP")           return (Pol *)&SP;
    else if (name=="main.SR")           return (Pol *)&SR;
    else if (name=="main.arith")        return (Pol *)&arith;
    else if (name=="main.assert")       return (Pol *)&assert;
    else if (name=="main.bin")          return (Pol *)&bin;
    else if (name=="main.comparator")   return (Pol *)&comparator;
    else if (name=="main.ecRecover")    return (Pol *)&ecRecover;
    else if (name=="main.hashE")        return (Pol *)&hashE;
    else if (name=="main.hashRD")       return (Pol *)&hashRD;
    else if (name=="main.hashWR")       return (Pol *)&hashWR;
    else if (name=="main.inA")          return (Pol *)&inA;
    else if (name=="main.inB")          return (Pol *)&inB;
    else if (name=="main.inC")          return (Pol *)&inC;
    else if (name=="main.inD")          return (Pol *)&inD;
    else if (name=="main.inE")          return (Pol *)&inE;
    else if (name=="main.inCTX")        return (Pol *)&inCTX;
    else if (name=="main.inFREE")       return (Pol *)&inFREE;
    else if (name=="main.inGAS")        return (Pol *)&inGAS;
    else if (name=="main.inMAXMEM")     return (Pol *)&inMAXMEM;
    else if (name=="main.inPC")         return (Pol *)&inPC;
    else if (name=="main.inSP")         return (Pol *)&inSP;
    else if (name=="main.inSR")         return (Pol *)&inSR;
    else if (name=="main.inSTEP")       return (Pol *)&inSTEP;
    else if (name=="main.inc")          return (Pol *)&inc;
    else if (name=="main.dec")          return (Pol *)&dec;
    else if (name=="main.ind")          return (Pol *)&ind;
    else if (name=="main.isCode")       return (Pol *)&isCode;
    else if (name=="main.isMaxMem")     return (Pol *)&isMaxMem;
    else if (name=="main.isMem")        return (Pol *)&isMem;
    else if (name=="main.isNeg")        return (Pol *)&isNeg;
    else if (name=="main.isStack")      return (Pol *)&isStack;
    else if (name=="main.mRD")          return (Pol *)&mRD;
    else if (name=="main.mWR")          return (Pol *)&mWR;
    else if (name=="main.neg")          return (Pol *)&neg;
    else if (name=="main.offset")       return (Pol *)&offset;
    else if (name=="main.opcodeRomMap") return (Pol *)&opcodeRomMap;
    else if (name=="main.sRD")          return (Pol *)&sRD;
    else if (name=="main.sWR")          return (Pol *)&sWR;
    else if (name=="main.setA")         return (Pol *)&setA;
    else if (name=="main.setB")         return (Pol *)&setB;
    else if (name=="main.setC")         return (Pol *)&setC;
    else if (name=="main.setD")         return (Pol *)&setD;
    else if (name=="main.setE")         return (Pol *)&setE;
    else if (name=="main.setCTX")       return (Pol *)&setCTX;
    else if (name=="main.setGAS")       return (Pol *)&setGAS;
    else if (name=="main.setMAXMEM")    return (Pol *)&setMAXMEM;
    else if (name=="main.setPC")        return (Pol *)&setPC;
    else if (name=="main.setSP")        return (Pol *)&setSP;
    else if (name=="main.setSR")        return (Pol *)&setSR;
    else if (name=="main.shl")          return (Pol *)&shl;
    else if (name=="main.shr")          return (Pol *)&shr;
    else if (name=="main.useCTX")       return (Pol *)&useCTX;
    else if (name=="main.zkPC")         return (Pol *)&zkPC;
    else if (name=="byte4.freeIN")      return (Pol *)&byte4_freeIN;
    else if (name=="byte4.out")         return (Pol *)&byte4_out;

    // Constant, input polynomials
    else if (name=="GLOBAL.BYTE2")      return (Pol *)&global_byte2;
    else if (name=="GLOBAL.L1")         return (Pol *)&global_L1;
    else if (name=="GLOBAL.ZH")         return (Pol *)&global_ZH;
    else if (name=="GLOBAL.ZHINV")      return (Pol *)&global_ZHINV;
    else if (name=="byte4.SET")         return (Pol *)&byte4_SET;
    else if (name=="main.ROM")          return (Pol *)&ROM;
    else if (name=="main.STEP")         return (Pol *)&STEP;

    // If not found, log an error
    else
    {
        cerr << "Error: Pols::find() could not find a polynomial for name: " << name << endl;
        exit(-1);
    }
}

void Pols::mapToOutputFile (const string &outputFileName)
{
    mapToFile(outputFileName, true);
}

void Pols::mapToInputFile (const string &inputFileName)
{
    mapToFile(inputFileName, false);
}

void Pols::mapToFile (const string &file_name, bool bOutput)
{
    // Check and store the file name
    if (fileName.size()!=0)
    {
        cerr << "Error: Pols::mapToFile() called with an existing file name: " << fileName << endl;
        exit(-1);
    }
    if (file_name.size()==0)
    {
        cerr << "Error: Pols::mapToFile() called with an empty file name" << endl;
        exit(-1);
    }
    fileName = file_name;

    // Ensure all pols[] pointers have been assigned to one PolXxxx instance,
    // and take advantage of the loop to calculate the size
    polsSize = 0;
    for (uint64_t i=0; i<(bOutput ? NPOLS : NCONSTPOLS); i++)
    {
        if (orderedPols[i] == NULL)
        {
            cerr << "Error: Pols::mapToFile() found slot pols[" << i << "] empty" << endl;
            exit(-1);
        }
        polsSize += orderedPols[i]->elementSize*NEVALUATIONS;
    }
    cout << "Pols::mapToFile() calculated total size=" << polsSize << endl;

    // If input, check the file size is the same as the expected polsSize
    if (!bOutput)
    {
        struct stat sb;
        if ( lstat(fileName.c_str(), &sb) == -1)
        {
            cerr << "Error: Pols::mapToFile() failed calling lstat() of file " << fileName << endl;
            exit(-1);
        }
        if ((uint64_t)sb.st_size != polsSize)
        {
            cerr << "Error: Pols::mapToFile() found size of file " << fileName << " to be " << sb.st_size << " B instead of " << polsSize << " B" << endl;
            exit(-1);
        }
    }

    int oflags;
    if (bOutput) oflags = O_CREAT|O_RDWR|O_TRUNC;
    else         oflags = O_RDWR;
    int fd = open(fileName.c_str(), oflags, 0666);
    if (fd < 0)
    {
        cerr << "Error: Pols::mapToFile() failed opening " << (bOutput ? "output" : "input") << " file: " << fileName << endl;
        exit(-1);
    }

    // If output, extend the file size to the required one
    if (bOutput)
    {
        // Seek the last byte of the file
        int result = lseek(fd, polsSize-1, SEEK_SET);
        if (result == -1)
        {
            cerr << "Error: Pols::mapToFile() failed calling lseek() of file: " << fileName << endl;
            exit(-1);
        }

        // Write a 0 at the last byte of the file, to set its size; content is all zeros
        result = write(fd, "", 1);
        if (result < 0)
        {
            cerr << "Error: Pols::mapToFile() failed calling write() of file: " << fileName << endl;
            exit(-1);
        }
    }

    // Map the file into memory
    pPolsMappedMemmory = (uint8_t *)mmap( NULL, polsSize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (pPolsMappedMemmory == MAP_FAILED)
    {
        cerr << "Error: Pols::mapToFile() failed calling mmap() of file: " << fileName << endl;
        exit(-1);
    }
    close(fd);

    // Map every individual pol to the corresponding memory area, in order
    uint64_t offset = 0;
    for (uint64_t i=0; i<(bOutput ? NPOLS : NCONSTPOLS); i++)
    {
        switch (orderedPols[i]->elementType) {
            case et_bool:  ((PolBool *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_s8:    ((PolS8 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_u8:    ((PolU8 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_s16:   ((PolS16 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_u16:   ((PolU16 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_s32:   ((PolS32 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_u32:   ((PolU32 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_s64:   ((PolS64 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_u64:   ((PolU64 *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            case et_field: ((PolFieldElement *)(orderedPols[i]))->map(pPolsMappedMemmory+offset); break;
            default:
                cerr << "Error: Pols::mapToFile() found invalid elementType in pol " << i << endl;
                exit(-1);
        }
#ifdef LOG_POLS
        cout << "Mapped pols[" << i << "] with id "<< orderedPols[i]->id<< " and name \"" << orderedPols[i]->name << "\" to memory offset "<< offset << endl;
#endif
        offset += orderedPols[i]->elementSize*NEVALUATIONS;
    }
}

void Pols::unmap (void)
{
    // Unmap every polynomial
    for (uint64_t i=0; i<NPOLS; i++)
    {
        if (orderedPols[i] == NULL) break;

        switch (orderedPols[i]->elementType) {
            case et_bool:  ((PolBool *)(orderedPols[i]))->unmap(); break;
            case et_s8:    ((PolS8 *)(orderedPols[i]))->unmap(); break;
            case et_u8:    ((PolU8 *)(orderedPols[i]))->unmap(); break;
            case et_s16:   ((PolS16 *)(orderedPols[i]))->unmap(); break;
            case et_u16:   ((PolU16 *)(orderedPols[i]))->unmap(); break;
            case et_s32:   ((PolS32 *)(orderedPols[i]))->unmap(); break;
            case et_u32:   ((PolU32 *)(orderedPols[i]))->unmap(); break;
            case et_s64:   ((PolS64 *)(orderedPols[i]))->unmap(); break;
            case et_u64:   ((PolU64 *)(orderedPols[i]))->unmap(); break;
            case et_field: ((PolFieldElement *)(orderedPols[i]))->unmap(); break;
            default:
                cerr << "Error: Pols::unmap() found invalid elementType in pol " << i << endl;
                exit(-1);
        }
    }

    // Unmap the global memory address and reset size
    int err = munmap(pPolsMappedMemmory, polsSize);
    if (err != 0)
    {
        cerr << "Error: Pols::unmap() failed calling munmap() of file: " << fileName << endl;
        exit(-1);
    }
    pPolsMappedMemmory = NULL;
    polsSize = 0;
    fileName = "";
}