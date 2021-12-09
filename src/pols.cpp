#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include "pols.hpp"
#include "config.hpp"

// TODO: check if map performance is better
/* Initializes the variable that contains the polynomial ID */
void addPol (Context &ctx, string &name, uint64_t id, string &elementType)
{
    if ( ctx.orderedPols[id] != NULL )
    {
        cerr << "Error: addPol() found id " << id << " has already been added" << endl;
        exit(-1);
    }

         if (name=="main.A0") { ctx.polynomials.A0.id = id; ctx.orderedPols[id]=&(ctx.polynomials.A0); }
    else if (name=="main.A1") { ctx.polynomials.A1.id = id; ctx.orderedPols[id]=&(ctx.polynomials.A1); }
    else if (name=="main.A2") { ctx.polynomials.A2.id = id; ctx.orderedPols[id]=&(ctx.polynomials.A2); }
    else if (name=="main.A3") { ctx.polynomials.A3.id = id; ctx.orderedPols[id]=&(ctx.polynomials.A3); }
    else if (name=="main.B0") { ctx.polynomials.B0.id = id; ctx.orderedPols[id]=&(ctx.polynomials.B0); }
    else if (name=="main.B1") { ctx.polynomials.B1.id = id; ctx.orderedPols[id]=&(ctx.polynomials.B1); }
    else if (name=="main.B2") { ctx.polynomials.B2.id = id; ctx.orderedPols[id]=&(ctx.polynomials.B2); }
    else if (name=="main.B3") { ctx.polynomials.B3.id = id; ctx.orderedPols[id]=&(ctx.polynomials.B3); }
    else if (name=="main.C0") { ctx.polynomials.C0.id = id; ctx.orderedPols[id]=&(ctx.polynomials.C0); }
    else if (name=="main.C1") { ctx.polynomials.C1.id = id; ctx.orderedPols[id]=&(ctx.polynomials.C1); }
    else if (name=="main.C2") { ctx.polynomials.C2.id = id; ctx.orderedPols[id]=&(ctx.polynomials.C2); }
    else if (name=="main.C3") { ctx.polynomials.C3.id = id; ctx.orderedPols[id]=&(ctx.polynomials.C3); }
    else if (name=="main.D0") { ctx.polynomials.D0.id = id; ctx.orderedPols[id]=&(ctx.polynomials.D0); }
    else if (name=="main.D1") { ctx.polynomials.D1.id = id; ctx.orderedPols[id]=&(ctx.polynomials.D1); }
    else if (name=="main.D2") { ctx.polynomials.D2.id = id; ctx.orderedPols[id]=&(ctx.polynomials.D2); }
    else if (name=="main.D3") { ctx.polynomials.D3.id = id; ctx.orderedPols[id]=&(ctx.polynomials.D3); }
    else if (name=="main.E0") { ctx.polynomials.E0.id = id; ctx.orderedPols[id]=&(ctx.polynomials.E0); }
    else if (name=="main.E1") { ctx.polynomials.E1.id = id; ctx.orderedPols[id]=&(ctx.polynomials.E1); }
    else if (name=="main.E2") { ctx.polynomials.E2.id = id; ctx.orderedPols[id]=&(ctx.polynomials.E2); }
    else if (name=="main.E3") { ctx.polynomials.E3.id = id; ctx.orderedPols[id]=&(ctx.polynomials.E3); }
    else if (name=="main.FREE0") { ctx.polynomials.FREE0.id = id; ctx.orderedPols[id]=&(ctx.polynomials.FREE0); }
    else if (name=="main.FREE1") { ctx.polynomials.FREE1.id = id; ctx.orderedPols[id]=&(ctx.polynomials.FREE1); }
    else if (name=="main.FREE2") { ctx.polynomials.FREE2.id = id; ctx.orderedPols[id]=&(ctx.polynomials.FREE2); }
    else if (name=="main.FREE3") { ctx.polynomials.FREE3.id = id; ctx.orderedPols[id]=&(ctx.polynomials.FREE3); }
    else if (name=="main.CONST") { ctx.polynomials.CONST.id = id; ctx.orderedPols[id]=&(ctx.polynomials.CONST); }
    else if (name=="main.CTX") { ctx.polynomials.CTX.id = id; ctx.orderedPols[id]=&(ctx.polynomials.CTX); }
    else if (name=="main.GAS") { ctx.polynomials.GAS.id = id; ctx.orderedPols[id]=&(ctx.polynomials.GAS); }
    else if (name=="main.JMP") { ctx.polynomials.JMP.id = id; ctx.orderedPols[id]=&(ctx.polynomials.JMP); }
    else if (name=="main.JMPC") { ctx.polynomials.JMPC.id = id; ctx.orderedPols[id]=&(ctx.polynomials.JMPC); }
    else if (name=="main.MAXMEM") { ctx.polynomials.MAXMEM.id = id; ctx.orderedPols[id]=&(ctx.polynomials.MAXMEM); }
    else if (name=="main.PC") { ctx.polynomials.PC.id = id; ctx.orderedPols[id]=&(ctx.polynomials.PC); }
    else if (name=="main.SP") { ctx.polynomials.SP.id = id; ctx.orderedPols[id]=&(ctx.polynomials.SP); }
    else if (name=="main.SR") { ctx.polynomials.SR.id = id; ctx.orderedPols[id]=&(ctx.polynomials.SR); }
    else if (name=="main.arith") { ctx.polynomials.arith.id = id; ctx.orderedPols[id]=&(ctx.polynomials.arith); }
    else if (name=="main.assert") { ctx.polynomials.assert.id = id; ctx.orderedPols[id]=&(ctx.polynomials.assert); }
    else if (name=="main.bin") { ctx.polynomials.bin.id = id; ctx.orderedPols[id]=&(ctx.polynomials.bin); }
    else if (name=="main.comparator") { ctx.polynomials.comparator.id = id; ctx.orderedPols[id]=&(ctx.polynomials.comparator); }
    else if (name=="main.ecRecover") { ctx.polynomials.ecRecover.id = id; ctx.orderedPols[id]=&(ctx.polynomials.ecRecover); }
    else if (name=="main.hashE") { ctx.polynomials.hashE.id = id; ctx.orderedPols[id]=&(ctx.polynomials.hashE); }
    else if (name=="main.hashRD") { ctx.polynomials.hashRD.id = id; ctx.orderedPols[id]=&(ctx.polynomials.hashRD); }
    else if (name=="main.hashWR") { ctx.polynomials.hashWR.id = id; ctx.orderedPols[id]=&(ctx.polynomials.hashWR); }
    else if (name=="main.inA") { ctx.polynomials.inA.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inA); }
    else if (name=="main.inB") { ctx.polynomials.inB.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inB); }
    else if (name=="main.inC") { ctx.polynomials.inC.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inC); }
    else if (name=="main.inD") { ctx.polynomials.inD.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inD); }
    else if (name=="main.inE") { ctx.polynomials.inE.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inE); }
    else if (name=="main.inCTX") { ctx.polynomials.inCTX.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inCTX); }
    else if (name=="main.inFREE") { ctx.polynomials.inFREE.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inFREE); }
    else if (name=="main.inGAS") { ctx.polynomials.inGAS.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inGAS); }
    else if (name=="main.inMAXMEM") { ctx.polynomials.inMAXMEM.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inMAXMEM); }
    else if (name=="main.inPC") { ctx.polynomials.inPC.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inPC); }
    else if (name=="main.inSP") { ctx.polynomials.inSP.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inSP); }
    else if (name=="main.inSR") { ctx.polynomials.inSR.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inSR); }
    else if (name=="main.inSTEP") { ctx.polynomials.inSTEP.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inSTEP); }
    else if (name=="main.inc") { ctx.polynomials.inc.id = id; ctx.orderedPols[id]=&(ctx.polynomials.inc); }
    else if (name=="main.dec") { ctx.polynomials.dec.id = id; ctx.orderedPols[id]=&(ctx.polynomials.dec); }
    else if (name=="main.ind") { ctx.polynomials.ind.id = id; ctx.orderedPols[id]=&(ctx.polynomials.ind); }
    else if (name=="main.isCode") { ctx.polynomials.isCode.id = id; ctx.orderedPols[id]=&(ctx.polynomials.isCode); }
    else if (name=="main.isMaxMem") { ctx.polynomials.isMaxMem.id = id; ctx.orderedPols[id]=&(ctx.polynomials.isMaxMem); }
    else if (name=="main.isMem") { ctx.polynomials.isMem.id = id; ctx.orderedPols[id]=&(ctx.polynomials.isMem); }
    else if (name=="main.isNeg") { ctx.polynomials.isNeg.id = id; ctx.orderedPols[id]=&(ctx.polynomials.isNeg); }
    else if (name=="main.isStack") { ctx.polynomials.isStack.id = id; ctx.orderedPols[id]=&(ctx.polynomials.isStack); }
    else if (name=="main.mRD") { ctx.polynomials.mRD.id = id; ctx.orderedPols[id]=&(ctx.polynomials.mRD); }
    else if (name=="main.mWR") { ctx.polynomials.mWR.id = id; ctx.orderedPols[id]=&(ctx.polynomials.mWR); }
    else if (name=="main.neg") { ctx.polynomials.neg.id = id; ctx.orderedPols[id]=&(ctx.polynomials.neg); }
    else if (name=="main.offset") { ctx.polynomials.offset.id = id; ctx.orderedPols[id]=&(ctx.polynomials.offset); }
    else if (name=="main.opcodeRomMap") { ctx.polynomials.opcodeRomMap.id = id; ctx.orderedPols[id]=&(ctx.polynomials.opcodeRomMap); }
    else if (name=="main.sRD") { ctx.polynomials.sRD.id = id; ctx.orderedPols[id]=&(ctx.polynomials.sRD); }
    else if (name=="main.sWR") { ctx.polynomials.sWR.id = id; ctx.orderedPols[id]=&(ctx.polynomials.sWR); }
    else if (name=="main.setA") { ctx.polynomials.setA.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setA); }
    else if (name=="main.setB") { ctx.polynomials.setB.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setB); }
    else if (name=="main.setC") { ctx.polynomials.setC.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setC); }
    else if (name=="main.setD") { ctx.polynomials.setD.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setD); }
    else if (name=="main.setE") { ctx.polynomials.setE.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setE); }
    else if (name=="main.setCTX") { ctx.polynomials.setCTX.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setCTX); }
    else if (name=="main.setGAS") { ctx.polynomials.setGAS.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setGAS); }
    else if (name=="main.setMAXMEM") { ctx.polynomials.setMAXMEM.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setMAXMEM); }
    else if (name=="main.setPC") { ctx.polynomials.setPC.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setPC); }
    else if (name=="main.setSP") { ctx.polynomials.setSP.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setSP); }
    else if (name=="main.setSR") { ctx.polynomials.setSR.id = id; ctx.orderedPols[id]=&(ctx.polynomials.setSR); }
    else if (name=="main.shl") { ctx.polynomials.shl.id = id; ctx.orderedPols[id]=&(ctx.polynomials.shl); }
    else if (name=="main.shr") { ctx.polynomials.shr.id = id; ctx.orderedPols[id]=&(ctx.polynomials.shr); }
    else if (name=="main.useCTX") { ctx.polynomials.useCTX.id = id; ctx.orderedPols[id]=&(ctx.polynomials.useCTX); }
    else if (name=="main.zkPC") { ctx.polynomials.zkPC.id = id; ctx.orderedPols[id]=&(ctx.polynomials.zkPC); }
    else if (name=="byte4.freeIN") { ctx.polynomials.byte4_freeIN.id = id; ctx.orderedPols[id]=&(ctx.polynomials.byte4_freeIN); }
    else if (name=="byte4.out") { ctx.polynomials.byte4_out.id = id; ctx.orderedPols[id]=&(ctx.polynomials.byte4_out); }
    else
    {
        cerr << "Error: addPol() could not find a polynomial for name: " << name << ", id: " << id << endl;
        exit(-1); // TODO: Should we kill the process?
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
    // PIL JSON file must contain a nCommitments key at the root level
    if ( !pil.contains("nCommitments") ||
         !pil["nCommitments"].is_number_unsigned() )
    {
        cerr << "Error: nCommitments key not found in PIL JSON file" << endl;
        exit(-1);
    }
    uint64_t nCommitments;
    nCommitments = pil["nCommitments"];
    cout << nCommitments << endl;

    // PIL JSON file must contain a references structure at the root level
    if ( !pil.contains("references") ||
         !pil["references"].is_structured() )
    {
        cerr << "Error: references key not found in PIL JSON file" << endl;
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
                    cerr << "Error: polynomial " << key << " id(" << id << ") >= NPOLS(" << NPOLS << ")" << endl;
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