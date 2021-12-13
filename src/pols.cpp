#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include "pols.hpp"
#include "config.hpp"
#include "context.hpp"

void Pols::parse (const json &pil, vector<PolJsonData> &v)
{
    // PIL JSON file must contain a nCommitments key at the root level
    if ( !pil.contains("nCommitments") ||
         !pil["nCommitments"].is_number_unsigned() )
    {
        cerr << "Error: createPold() nCommitments key not found in PIL JSON file" << endl;
        exit(-1);
    }
    uint64_t nCommitments;
    nCommitments = pil["nCommitments"];
    cout << nCommitments << endl;

    // PIL JSON file must contain a references structure at the root level
    if ( !pil.contains("references") ||
         !pil["references"].is_structured() )
    {
        cerr << "Error: createPold() references key not found in PIL JSON file" << endl;
        exit(-1);
    }

    // Iterate the PIL JSON references array
    json references = pil["references"];
    uint64_t addedPols = 0;
    for (json::iterator it = references.begin(); it != references.end(); ++it) {
        string key = it.key();
        json value = it.value();
        if ( value.is_object() &&
             value.contains("type") && 
             value["type"].is_string() &&
             value.contains("id") &&
             value["id"].is_number_unsigned() ) 
        {
            string type = it.value()["type"];
            uint64_t id = it.value()["id"];
            if (type=="cmP") {
                if (id>=NPOLS)
                {
                    cerr << "Error: createPold() polynomial " << key << " id(" << id << ") >= NPOLS(" << NPOLS << ")" << endl;
                    exit(-1);
                }
                string elementType = it.value()["elementType"];
                PolJsonData data;
                data.name = key;
                data.id = id;
                data.elementType = elementType;
                v.push_back(data);
#ifdef LOG_POLS
                cout << "Added polynomial " << addedPols << ": " << key << " with ID " << id << " and type " << type << endl;
#endif
                addedPols++;
            }
        }
    }
#ifdef LOG_POLS
    cout << "Added " << addedPols << " polynomials" << endl;
#endif
}

void Pols::load(const vector<PolJsonData> &v, const string &outputFile)
{
    // Reset orderedPols
    memset(&orderedPols, 0, sizeof(orderedPols));

    // Store output file name
    this->outputFile = outputFile;

    // Add one polynomial per vector entry
    for (uint64_t i=0; i<v.size(); i++)
    {
        addPol(v[i].name, v[i].id, v[i].elementType);
    }

    // Map
    map();
}


void Pols::unload(void)
{
    unmap();
}

void Pols::addPol(const string &name, const uint64_t id, const string &elementType)
{
    // Find the polynomial with this name
    Pol * pPol = find(name);

    // Check that the element type matches
    if ( (elementType == "bool") && pPol->elementType != et_bool ||
         (elementType == "s8") && pPol->elementType != et_s8 ||
         (elementType == "u8") && pPol->elementType != et_u8 ||
         (elementType == "s16") && pPol->elementType != et_s16 ||
         (elementType == "u16") && pPol->elementType != et_u16 ||
         (elementType == "s32") && pPol->elementType !=et_s32 ||
         (elementType == "u32") && pPol->elementType != et_u32 ||
         (elementType == "s64") && pPol->elementType != et_s64 ||
         (elementType == "u64") && pPol->elementType !=et_u64 ||
         (elementType == "field") && pPol->elementType != et_field )
    {
             cerr << "Error: addPol() found inconsistent element type for pol " << name << endl;
             exit(-1);
    }

    // Store ID
    pPol ->id = id;

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
    else
    {
        cerr << "Error: Pols::find() could not find a polynomial for name: " << name << endl;
        exit(-1);
    }
}

void Pols::map (void)
{
    // Ensure all pols[] pointers have been assigned to one PolXxxx instance,
    // and take advantage of the loop to calculate the size
    polsSize = 0;
    for (uint64_t i=0; i<NPOLS; i++)
    {
        if (orderedPols[i] == NULL)
        {
            cout << "Error: Pols::map() found slot pols[" << i << "] empty" << endl;
            exit(-1);
        }
        polsSize += orderedPols[i]->elementSize()*NEVALUATIONS;
    }
    cout << "Pols::map() calculated total size=" << polsSize << endl;

    int fd = open(outputFile.c_str(), O_CREAT|O_RDWR|O_TRUNC, 0666);
    if (fd < 0)
    {
        cout << "Error: Pols::map() failed opening output file: " << outputFile << endl;
        exit(-1);
    }

    // Seek the last byte of the file
    int result = lseek(fd, polsSize-1, SEEK_SET);
    if (result == -1)
    {
        cout << "Error: Pols::map() failed calling lseek() of file: " << outputFile << endl;
        exit(-1);
    }

    // Write a 0 at the last byte of the file, to set its size; content is all zeros
    result = write(fd, "", 1);
    if (result < 0)
    {
        cout << "Error: Pols::map() failed calling write() of file: " << outputFile << endl;
        exit(-1);
    }

    // Map the file into memory
    pPolsMappedMemmory = (uint8_t *)mmap( NULL, polsSize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (pPolsMappedMemmory == MAP_FAILED)
    {
        cout << "Error: Pols::map() failed calling mmap() of file: " << outputFile << endl;
        exit(-1);
    }
    close(fd);

    // Map every individual pol to the corresponding memory area, in order
    uint64_t offset = 0;
    for (uint64_t i=0; i<NPOLS; i++)
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
                cerr << "Error: Pols::map() found invalid elementType in pol " << i << endl;
                exit(-1);
        }
#ifdef LOG_POLS
        cout << "Mapped pols[" << i << "] with id "<< orderedPols[i]->id<< " to memory offset "<< offset << endl;
#endif
        offset += orderedPols[i]->elementSize()*NEVALUATIONS;
    }
}

void Pols::unmap (void)
{
    // Unmap every polynomial
    for (uint64_t i=0; i<NPOLS; i++)
    {
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
        cout << "Error: Pols::unmap() failed calling munmap() of file: " << outputFile << endl;
        exit(-1);
    }
    pPolsMappedMemmory = NULL;
    polsSize = 0;
}