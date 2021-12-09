#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include "pols.hpp"
#include "config.hpp"

void createPols (Context &ctx, json &pil);
void mapPols    (Context &ctx);
void unmapPols  (Context &ctx);

void loadPols (Context &ctx, json &pil)
{
    createPols(ctx, pil);
    mapPols(ctx);
}

void unloadPols( Context &ctx)
{
    unmapPols(ctx);
}

void registerPol (Context &ctx, Pol &p, uint64_t id)
{
    // Assign the id to the polynomial
    p.id = id;

    // Check that the ordered slot is not already occupied
    if ( ctx.orderedPols[id] != NULL )
    {
        cerr << "Error: registerPol() found id " << id << " has already been added" << endl;
        exit(-1);
    }

    // Keep a pointer to the polynomial in the right position of the ordered polynomials list
    ctx.orderedPols[id] = &p;
}

// TODO: check if map performance is better
/* Initializes the variable that contains the polynomial ID */
void addPol (Context &ctx, string &name, uint64_t id, string &elementType)
{
         if (name=="main.A0")           registerPol(ctx, ctx.pols.A0, id);
    else if (name=="main.A1")           registerPol(ctx, ctx.pols.A1, id);
    else if (name=="main.A2")           registerPol(ctx, ctx.pols.A2, id);
    else if (name=="main.A3")           registerPol(ctx, ctx.pols.A3, id);
    else if (name=="main.B0")           registerPol(ctx, ctx.pols.B0, id);
    else if (name=="main.B1")           registerPol(ctx, ctx.pols.B1, id);
    else if (name=="main.B2")           registerPol(ctx, ctx.pols.B2, id);
    else if (name=="main.B3")           registerPol(ctx, ctx.pols.B3, id);
    else if (name=="main.C0")           registerPol(ctx, ctx.pols.C0, id);
    else if (name=="main.C1")           registerPol(ctx, ctx.pols.C1, id);
    else if (name=="main.C2")           registerPol(ctx, ctx.pols.C2, id);
    else if (name=="main.C3")           registerPol(ctx, ctx.pols.C3, id);
    else if (name=="main.D0")           registerPol(ctx, ctx.pols.D0, id);
    else if (name=="main.D1")           registerPol(ctx, ctx.pols.D1, id);
    else if (name=="main.D2")           registerPol(ctx, ctx.pols.D2, id);
    else if (name=="main.D3")           registerPol(ctx, ctx.pols.D3, id);
    else if (name=="main.E0")           registerPol(ctx, ctx.pols.E0, id);
    else if (name=="main.E1")           registerPol(ctx, ctx.pols.E1, id);
    else if (name=="main.E2")           registerPol(ctx, ctx.pols.E2, id);
    else if (name=="main.E3")           registerPol(ctx, ctx.pols.E3, id);
    else if (name=="main.FREE0")        registerPol(ctx, ctx.pols.FREE0, id);
    else if (name=="main.FREE1")        registerPol(ctx, ctx.pols.FREE1, id);
    else if (name=="main.FREE2")        registerPol(ctx, ctx.pols.FREE2, id);
    else if (name=="main.FREE3")        registerPol(ctx, ctx.pols.FREE3, id);
    else if (name=="main.CONST")        registerPol(ctx, ctx.pols.CONST, id);
    else if (name=="main.CTX")          registerPol(ctx, ctx.pols.CTX, id);
    else if (name=="main.GAS")          registerPol(ctx, ctx.pols.GAS, id);
    else if (name=="main.JMP")          registerPol(ctx, ctx.pols.JMP, id);
    else if (name=="main.JMPC")         registerPol(ctx, ctx.pols.JMPC, id);
    else if (name=="main.MAXMEM")       registerPol(ctx, ctx.pols.MAXMEM, id);
    else if (name=="main.PC")           registerPol(ctx, ctx.pols.PC, id);
    else if (name=="main.SP")           registerPol(ctx, ctx.pols.SP, id);
    else if (name=="main.SR")           registerPol(ctx, ctx.pols.SR, id);
    else if (name=="main.arith")        registerPol(ctx, ctx.pols.arith, id);
    else if (name=="main.assert")       registerPol(ctx, ctx.pols.assert, id);
    else if (name=="main.bin")          registerPol(ctx, ctx.pols.bin, id);
    else if (name=="main.comparator")   registerPol(ctx, ctx.pols.comparator, id);
    else if (name=="main.ecRecover")    registerPol(ctx, ctx.pols.ecRecover, id);
    else if (name=="main.hashE")        registerPol(ctx, ctx.pols.hashE, id);
    else if (name=="main.hashRD")       registerPol(ctx, ctx.pols.hashRD, id);
    else if (name=="main.hashWR")       registerPol(ctx, ctx.pols.hashWR, id);
    else if (name=="main.inA")          registerPol(ctx, ctx.pols.inA, id);
    else if (name=="main.inB")          registerPol(ctx, ctx.pols.inB, id);
    else if (name=="main.inC")          registerPol(ctx, ctx.pols.inC, id);
    else if (name=="main.inD")          registerPol(ctx, ctx.pols.inD, id);
    else if (name=="main.inE")          registerPol(ctx, ctx.pols.inE, id);
    else if (name=="main.inCTX")        registerPol(ctx, ctx.pols.inCTX, id);
    else if (name=="main.inFREE")       registerPol(ctx, ctx.pols.inFREE, id);
    else if (name=="main.inGAS")        registerPol(ctx, ctx.pols.inGAS, id);
    else if (name=="main.inMAXMEM")     registerPol(ctx, ctx.pols.inMAXMEM, id);
    else if (name=="main.inPC")         registerPol(ctx, ctx.pols.inPC, id);
    else if (name=="main.inSP")         registerPol(ctx, ctx.pols.inSP, id);
    else if (name=="main.inSR")         registerPol(ctx, ctx.pols.inSR, id);
    else if (name=="main.inSTEP")       registerPol(ctx, ctx.pols.inSTEP, id);
    else if (name=="main.inc")          registerPol(ctx, ctx.pols.inc, id);
    else if (name=="main.dec")          registerPol(ctx, ctx.pols.dec, id);
    else if (name=="main.ind")          registerPol(ctx, ctx.pols.ind, id);
    else if (name=="main.isCode")       registerPol(ctx, ctx.pols.isCode, id);
    else if (name=="main.isMaxMem")     registerPol(ctx, ctx.pols.isMaxMem, id);
    else if (name=="main.isMem")        registerPol(ctx, ctx.pols.isMem, id);
    else if (name=="main.isNeg")        registerPol(ctx, ctx.pols.isNeg, id);
    else if (name=="main.isStack")      registerPol(ctx, ctx.pols.isStack, id);
    else if (name=="main.mRD")          registerPol(ctx, ctx.pols.mRD, id);
    else if (name=="main.mWR")          registerPol(ctx, ctx.pols.mWR, id);
    else if (name=="main.neg")          registerPol(ctx, ctx.pols.neg, id);
    else if (name=="main.offset")       registerPol(ctx, ctx.pols.offset, id);
    else if (name=="main.opcodeRomMap") registerPol(ctx, ctx.pols.opcodeRomMap, id);
    else if (name=="main.sRD")          registerPol(ctx, ctx.pols.sRD, id);
    else if (name=="main.sWR")          registerPol(ctx, ctx.pols.sWR, id);
    else if (name=="main.setA")         registerPol(ctx, ctx.pols.setA, id);
    else if (name=="main.setB")         registerPol(ctx, ctx.pols.setB, id);
    else if (name=="main.setC")         registerPol(ctx, ctx.pols.setC, id);
    else if (name=="main.setD")         registerPol(ctx, ctx.pols.setD, id);
    else if (name=="main.setE")         registerPol(ctx, ctx.pols.setE, id);
    else if (name=="main.setCTX")       registerPol(ctx, ctx.pols.setCTX, id);
    else if (name=="main.setGAS")       registerPol(ctx, ctx.pols.setGAS, id);
    else if (name=="main.setMAXMEM")    registerPol(ctx, ctx.pols.setMAXMEM, id);
    else if (name=="main.setPC")        registerPol(ctx, ctx.pols.setPC, id);
    else if (name=="main.setSP")        registerPol(ctx, ctx.pols.setSP, id);
    else if (name=="main.setSR")        registerPol(ctx, ctx.pols.setSR, id);
    else if (name=="main.shl")          registerPol(ctx, ctx.pols.shl, id);
    else if (name=="main.shr")          registerPol(ctx, ctx.pols.shr, id);
    else if (name=="main.useCTX")       registerPol(ctx, ctx.pols.useCTX, id);
    else if (name=="main.zkPC")         registerPol(ctx, ctx.pols.zkPC, id);
    else if (name=="byte4.freeIN")      registerPol(ctx, ctx.pols.byte4_freeIN, id);
    else if (name=="byte4.out")         registerPol(ctx, ctx.pols.byte4_out, id);
    else
    {
        cerr << "Error: addPol() could not find a polynomial for name: " << name << ", id: " << id << endl;
        exit(-1);
    }

    if ( (elementType == "bool") && ctx.orderedPols[id]->elementType != et_bool ||
         (elementType == "s8") && ctx.orderedPols[id]->elementType != et_s8 ||
         (elementType == "u8") && ctx.orderedPols[id]->elementType != et_u8 ||
         (elementType == "s16") && ctx.orderedPols[id]->elementType != et_s16 ||
         (elementType == "u16") && ctx.orderedPols[id]->elementType != et_u16 ||
         (elementType == "s32") && ctx.orderedPols[id]->elementType !=et_s32 ||
         (elementType == "u32") && ctx.orderedPols[id]->elementType != et_u32 ||
         (elementType == "s64") && ctx.orderedPols[id]->elementType != et_s64 ||
         (elementType == "u64") && ctx.orderedPols[id]->elementType !=et_u64 ||
         (elementType == "field") && ctx.orderedPols[id]->elementType != et_field )
    {
             cerr << "Error: addPol() found inconsistent element type for pol " << name << endl;
             exit(-1);
    }
}

/* 
    This function creates an array of polynomials and a mapping that maps the reference name in pil to the polynomial
*/
void createPols (Context &ctx, json &pil)
{
    // Reset orderedPols
    memset(&ctx.orderedPols, 0, sizeof(ctx.orderedPols));

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
                addPol(ctx, key, id, elementType);
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

void mapPols (Context &ctx)
{
    // Ensure all pols[] pointers have been assigned to one PolXxxx instance,
    // and take advantage of the loop to calculate the size
    ctx.polsSize = 0;
    for (uint64_t i=0; i<NPOLS; i++)
    {
        if (ctx.orderedPols[i] == NULL)
        {
            cout << "Error: mapPols() found slot ctx.pols[" << i << "] empty" << endl;
            exit(-1);
        }
        ctx.polsSize += ctx.orderedPols[i]->elementSize()*NEVALUATIONS;
    }
    cout << "mapPols() calculated total size=" << ctx.polsSize << endl;

    int fd = open(ctx.outputFile.c_str(), O_CREAT|O_RDWR|O_TRUNC, 0666);
    if (fd < 0)
    {
        cout << "Error: mapPols() failed opening output file: " << ctx.outputFile << endl;
        exit(-1);
    }

    // Seek the last byte of the file
    int result = lseek(fd, ctx.polsSize-1, SEEK_SET);
    if (result == -1)
    {
        cout << "Error: mapPols() failed calling lseek() of file: " << ctx.outputFile << endl;
        exit(-1);
    }

    // Write a 0 at the last byte of the file, to set its size
    result = write(fd, "", 1);
    if (result < 0)
    {
        cout << "Error: mapPols() failed calling write() of file: " << ctx.outputFile << endl;
        exit(-1);
    }

    // TODO: Should we write the whole content of the file to 0?

    ctx.pPolsMappedMemmory = (uint8_t *)mmap( NULL, ctx.polsSize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (ctx.pPolsMappedMemmory == MAP_FAILED)
    {
        cout << "Error: mapPols() failed calling mmap() of file: " << ctx.outputFile << endl;
        exit(-1);
    }
    close(fd);

    // Map every individual pol to the corresponding memory area, in order
    uint64_t offset = 0;
    for (uint64_t i=0; i<NPOLS; i++)
    {
        switch (ctx.orderedPols[i]->elementType) {
            case et_bool: ((PolBool *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s8: ((PolS8 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u8: ((PolU8 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s16: ((PolS16 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u16: ((PolU16 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s32: ((PolS32 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u32: ((PolU32 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s64: ((PolS64 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u64: ((PolU64 *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_field: ((PolFieldElement *)(ctx.orderedPols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            default:
                cerr << "Error: mapPols() found invalid elementType in pol " << i << endl;
                exit(-1);
        }
#ifdef LOG_POLS
        cout << "Mapped pols[" << i << "] with id "<< ctx.orderedPols[i]->id<< " to memory offset "<< offset << endl;
#endif
        offset += ctx.orderedPols[i]->elementSize()*NEVALUATIONS;
    }
}

void unmapPols (Context &ctx)
{
    int err = munmap(ctx.pPolsMappedMemmory, ctx.polsSize);
    if (err != 0)
    {
        cout << "Error: closePols() failed calling munmap() of file: " << ctx.outputFile << endl;
        exit(-1);
    }
    for (uint64_t i=0; i<NPOLS; i++)
    {
        switch (ctx.orderedPols[i]->elementType) {
            case et_bool: ((PolBool *)(ctx.orderedPols[i]))->unmap(); break;
            case et_s8: ((PolS8 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_u8: ((PolU8 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_s16: ((PolS16 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_u16: ((PolU16 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_s32: ((PolS32 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_u32: ((PolU32 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_s64: ((PolS64 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_u64: ((PolU64 *)(ctx.orderedPols[i]))->unmap(); break;
            case et_field: ((PolFieldElement *)(ctx.orderedPols[i]))->unmap(); break;
            default:
                cerr << "Error: unmapPols() found invalid elementType in pol " << i << endl;
                exit(-1);
        }
    }
    ctx.pPolsMappedMemmory = NULL;

    ctx.polsSize = 0;
}