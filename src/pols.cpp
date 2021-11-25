#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include "pols.hpp"

// TODO: check if map performance is better
/* Initializes the variable that contains the polynomial ID */
void addPol (Context &ctx, string &name, uint64_t id, string &elementType)
{
    if ( ctx.pols[id] != NULL )
    {
        cerr << "Error: addPol() found id " << id << " has already been added" << endl;
        exit(-1);
    }

         if (name=="main.A0") { ctx.A0.id = id; ctx.pols[id]=&(ctx.A0); }
    else if (name=="main.A1") { ctx.A1.id = id; ctx.pols[id]=&(ctx.A1); }
    else if (name=="main.A2") { ctx.A2.id = id; ctx.pols[id]=&(ctx.A2); }
    else if (name=="main.A3") { ctx.A3.id = id; ctx.pols[id]=&(ctx.A3); }
    else if (name=="main.B0") { ctx.B0.id = id; ctx.pols[id]=&(ctx.B0); }
    else if (name=="main.B1") { ctx.B1.id = id; ctx.pols[id]=&(ctx.B1); }
    else if (name=="main.B2") { ctx.B2.id = id; ctx.pols[id]=&(ctx.B2); }
    else if (name=="main.B3") { ctx.B3.id = id; ctx.pols[id]=&(ctx.B3); }
    else if (name=="main.C0") { ctx.C0.id = id; ctx.pols[id]=&(ctx.C0); }
    else if (name=="main.C1") { ctx.C1.id = id; ctx.pols[id]=&(ctx.C1); }
    else if (name=="main.C2") { ctx.C2.id = id; ctx.pols[id]=&(ctx.C2); }
    else if (name=="main.C3") { ctx.C3.id = id; ctx.pols[id]=&(ctx.C3); }
    else if (name=="main.D0") { ctx.D0.id = id; ctx.pols[id]=&(ctx.D0); }
    else if (name=="main.D1") { ctx.D1.id = id; ctx.pols[id]=&(ctx.D1); }
    else if (name=="main.D2") { ctx.D2.id = id; ctx.pols[id]=&(ctx.D2); }
    else if (name=="main.D3") { ctx.D3.id = id; ctx.pols[id]=&(ctx.D3); }
    else if (name=="main.E0") { ctx.E0.id = id; ctx.pols[id]=&(ctx.E0); }
    else if (name=="main.E1") { ctx.E1.id = id; ctx.pols[id]=&(ctx.E1); }
    else if (name=="main.E2") { ctx.E2.id = id; ctx.pols[id]=&(ctx.E2); }
    else if (name=="main.E3") { ctx.E3.id = id; ctx.pols[id]=&(ctx.E3); }
    else if (name=="main.FREE0") { ctx.FREE0.id = id; ctx.pols[id]=&(ctx.FREE0); }
    else if (name=="main.FREE1") { ctx.FREE1.id = id; ctx.pols[id]=&(ctx.FREE1); }
    else if (name=="main.FREE2") { ctx.FREE2.id = id; ctx.pols[id]=&(ctx.FREE2); }
    else if (name=="main.FREE3") { ctx.FREE3.id = id; ctx.pols[id]=&(ctx.FREE3); }
    else if (name=="main.CONST") { ctx.CONST.id = id; ctx.pols[id]=&(ctx.CONST); }
    else if (name=="main.CTX") { ctx.CTX.id = id; ctx.pols[id]=&(ctx.CTX); }
    else if (name=="main.GAS") { ctx.GAS.id = id; ctx.pols[id]=&(ctx.GAS); }
    else if (name=="main.JMP") { ctx.JMP.id = id; ctx.pols[id]=&(ctx.JMP); }
    else if (name=="main.JMPC") { ctx.JMPC.id = id; ctx.pols[id]=&(ctx.JMPC); }
    else if (name=="main.MAXMEM") { ctx.MAXMEM.id = id; ctx.pols[id]=&(ctx.MAXMEM); }
    else if (name=="main.PC") { ctx.PC.id = id; ctx.pols[id]=&(ctx.PC); }
    else if (name=="main.SP") { ctx.SP.id = id; ctx.pols[id]=&(ctx.SP); }
    else if (name=="main.SR") { ctx.SR.id = id; ctx.pols[id]=&(ctx.SR); }
    else if (name=="main.arith") { ctx.arith.id = id; ctx.pols[id]=&(ctx.arith); }
    else if (name=="main.assert") { ctx.assert.id = id; ctx.pols[id]=&(ctx.assert); }
    else if (name=="main.bin") { ctx.bin.id = id; ctx.pols[id]=&(ctx.bin); }
    else if (name=="main.comparator") { ctx.comparator.id = id; ctx.pols[id]=&(ctx.comparator); }
    else if (name=="main.ecRecover") { ctx.ecRecover.id = id; ctx.pols[id]=&(ctx.ecRecover); }
    else if (name=="main.hashE") { ctx.hashE.id = id; ctx.pols[id]=&(ctx.hashE); }
    else if (name=="main.hashRD") { ctx.hashRD.id = id; ctx.pols[id]=&(ctx.hashRD); }
    else if (name=="main.hashWR") { ctx.hashWR.id = id; ctx.pols[id]=&(ctx.hashWR); }
    else if (name=="main.inA") { ctx.inA.id = id; ctx.pols[id]=&(ctx.inA); }
    else if (name=="main.inB") { ctx.inB.id = id; ctx.pols[id]=&(ctx.inB); }
    else if (name=="main.inC") { ctx.inC.id = id; ctx.pols[id]=&(ctx.inC); }
    else if (name=="main.inD") { ctx.inD.id = id; ctx.pols[id]=&(ctx.inD); }
    else if (name=="main.inE") { ctx.inE.id = id; ctx.pols[id]=&(ctx.inE); }
    else if (name=="main.inCTX") { ctx.inCTX.id = id; ctx.pols[id]=&(ctx.inCTX); }
    else if (name=="main.inFREE") { ctx.inFREE.id = id; ctx.pols[id]=&(ctx.inFREE); }
    else if (name=="main.inGAS") { ctx.inGAS.id = id; ctx.pols[id]=&(ctx.inGAS); }
    else if (name=="main.inMAXMEM") { ctx.inMAXMEM.id = id; ctx.pols[id]=&(ctx.inMAXMEM); }
    else if (name=="main.inPC") { ctx.inPC.id = id; ctx.pols[id]=&(ctx.inPC); }
    else if (name=="main.inSP") { ctx.inSP.id = id; ctx.pols[id]=&(ctx.inSP); }
    else if (name=="main.inSR") { ctx.inSR.id = id; ctx.pols[id]=&(ctx.inSR); }
    else if (name=="main.inSTEP") { ctx.inSTEP.id = id; ctx.pols[id]=&(ctx.inSTEP); }
    else if (name=="main.inc") { ctx.inc.id = id; ctx.pols[id]=&(ctx.inc); }
    else if (name=="main.dec") { ctx.dec.id = id; ctx.pols[id]=&(ctx.dec); }
    else if (name=="main.ind") { ctx.ind.id = id; ctx.pols[id]=&(ctx.ind); }
    else if (name=="main.isCode") { ctx.isCode.id = id; ctx.pols[id]=&(ctx.isCode); }
    else if (name=="main.isMaxMem") { ctx.isMaxMem.id = id; ctx.pols[id]=&(ctx.isMaxMem); }
    else if (name=="main.isMem") { ctx.isMem.id = id; ctx.pols[id]=&(ctx.isMem); }
    else if (name=="main.isNeg") { ctx.isNeg.id = id; ctx.pols[id]=&(ctx.isNeg); }
    else if (name=="main.isStack") { ctx.isStack.id = id; ctx.pols[id]=&(ctx.isStack); }
    else if (name=="main.mRD") { ctx.mRD.id = id; ctx.pols[id]=&(ctx.mRD); }
    else if (name=="main.mWR") { ctx.mWR.id = id; ctx.pols[id]=&(ctx.mWR); }
    else if (name=="main.neg") { ctx.neg.id = id; ctx.pols[id]=&(ctx.neg); }
    else if (name=="main.offset") { ctx.offset.id = id; ctx.pols[id]=&(ctx.offset); }
    else if (name=="main.opcodeRomMap") { ctx.opcodeRomMap.id = id; ctx.pols[id]=&(ctx.opcodeRomMap); }
    else if (name=="main.sRD") { ctx.sRD.id = id; ctx.pols[id]=&(ctx.sRD); }
    else if (name=="main.sWR") { ctx.sWR.id = id; ctx.pols[id]=&(ctx.sWR); }
    else if (name=="main.setA") { ctx.setA.id = id; ctx.pols[id]=&(ctx.setA); }
    else if (name=="main.setB") { ctx.setB.id = id; ctx.pols[id]=&(ctx.setB); }
    else if (name=="main.setC") { ctx.setC.id = id; ctx.pols[id]=&(ctx.setC); }
    else if (name=="main.setD") { ctx.setD.id = id; ctx.pols[id]=&(ctx.setD); }
    else if (name=="main.setE") { ctx.setE.id = id; ctx.pols[id]=&(ctx.setE); }
    else if (name=="main.setCTX") { ctx.setCTX.id = id; ctx.pols[id]=&(ctx.setCTX); }
    else if (name=="main.setGAS") { ctx.setGAS.id = id; ctx.pols[id]=&(ctx.setGAS); }
    else if (name=="main.setMAXMEM") { ctx.setMAXMEM.id = id; ctx.pols[id]=&(ctx.setMAXMEM); }
    else if (name=="main.setPC") { ctx.setPC.id = id; ctx.pols[id]=&(ctx.setPC); }
    else if (name=="main.setSP") { ctx.setSP.id = id; ctx.pols[id]=&(ctx.setSP); }
    else if (name=="main.setSR") { ctx.setSR.id = id; ctx.pols[id]=&(ctx.setSR); }
    else if (name=="main.shl") { ctx.shl.id = id; ctx.pols[id]=&(ctx.shl); }
    else if (name=="main.shr") { ctx.shr.id = id; ctx.pols[id]=&(ctx.shr); }
    else if (name=="main.useCTX") { ctx.useCTX.id = id; ctx.pols[id]=&(ctx.useCTX); }
    else if (name=="main.zkPC") { ctx.zkPC.id = id; ctx.pols[id]=&(ctx.zkPC); }
    else if (name=="byte4.freeIN") { ctx.byte4_freeIN.id = id; ctx.pols[id]=&(ctx.byte4_freeIN); }
    else if (name=="byte4.out") { ctx.byte4_out.id = id; ctx.pols[id]=&(ctx.byte4_out); }
    else
    {
        cerr << "Error: addPol() could not find a polynomial for name: " << name << ", id: " << id << endl;
        exit(-1); // TODO: Should we kill the process?
    }

    if ( (elementType == "bool") && ctx.pols[id]->elementType != et_bool ||
         (elementType == "s8") && ctx.pols[id]->elementType != et_s8 ||
         (elementType == "u8") && ctx.pols[id]->elementType != et_u8 ||
         (elementType == "s16") && ctx.pols[id]->elementType != et_s16 ||
         (elementType == "u16") && ctx.pols[id]->elementType != et_u16 ||
         (elementType == "s32") && ctx.pols[id]->elementType !=et_s32 ||
         (elementType == "u32") && ctx.pols[id]->elementType != et_u32 ||
         (elementType == "s64") && ctx.pols[id]->elementType != et_s64 ||
         (elementType == "u64") && ctx.pols[id]->elementType !=et_u64 ||
         (elementType == "field") && ctx.pols[id]->elementType != et_field )
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
                addedPols++;
                //cout << "Added polynomial " << addedPols << ": " << key << " with ID " << id << endl;
            }
        }

    }
}

void mapPols (Context &ctx)
{
    // Ensure all pols[] pointers have been assigned to one PolXxxx instance,
    // and take advantage of the loop to calculate the size
    ctx.polsSize = 0;
    for (uint64_t i=0; i<NPOLS; i++)
    {
        if (ctx.pols[i] == NULL)
        {
            cout << "Error: mapPols() found slot ctx.pols[" << i << "] empty" << endl;
            exit(-1);
        }
        ctx.polsSize += ctx.pols[i]->elementSize()*NEVALUATIONS;
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
        switch (ctx.pols[i]->elementType) {
            case et_bool: ((PolBool *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s8: ((PolS8 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u8: ((PolU8 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s16: ((PolS16 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u16: ((PolU16 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s32: ((PolS32 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u32: ((PolU32 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_s64: ((PolS64 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_u64: ((PolU64 *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            case et_field: ((PolFieldElement *)(ctx.pols[i]))->map(ctx.pPolsMappedMemmory+offset); break;
            default:
                cerr << "Error: mapPols() found invalid elementType in pol " << i << endl;
                exit(-1);
        }
        offset += ctx.pols[i]->elementSize()*NEVALUATIONS;
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
        switch (ctx.pols[i]->elementType) {
            case et_bool: ((PolBool *)(ctx.pols[i]))->unmap(); break;
            case et_s8: ((PolS8 *)(ctx.pols[i]))->unmap(); break;
            case et_u8: ((PolU8 *)(ctx.pols[i]))->unmap(); break;
            case et_s16: ((PolS16 *)(ctx.pols[i]))->unmap(); break;
            case et_u16: ((PolU16 *)(ctx.pols[i]))->unmap(); break;
            case et_s32: ((PolS32 *)(ctx.pols[i]))->unmap(); break;
            case et_u32: ((PolU32 *)(ctx.pols[i]))->unmap(); break;
            case et_s64: ((PolS64 *)(ctx.pols[i]))->unmap(); break;
            case et_u64: ((PolU64 *)(ctx.pols[i]))->unmap(); break;
            case et_field: ((PolFieldElement *)(ctx.pols[i]))->unmap(); break;
            default:
                cerr << "Error: unmapPols() found invalid elementType in pol " << i << endl;
                exit(-1);
        }
    }
    ctx.pPolsMappedMemmory = NULL;

    ctx.polsSize = 0;
}