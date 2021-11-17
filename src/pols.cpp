#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include "pols.hpp"

uint64_t A0 = INVALID_ID;
uint64_t A1 = INVALID_ID;
uint64_t A2 = INVALID_ID;
uint64_t A3 = INVALID_ID;
uint64_t B0 = INVALID_ID;
uint64_t B1 = INVALID_ID;
uint64_t B2 = INVALID_ID;
uint64_t B3 = INVALID_ID;
uint64_t C0 = INVALID_ID;
uint64_t C1 = INVALID_ID;
uint64_t C2 = INVALID_ID;
uint64_t C3 = INVALID_ID;
uint64_t D0 = INVALID_ID;
uint64_t D1 = INVALID_ID;
uint64_t D2 = INVALID_ID;
uint64_t D3 = INVALID_ID;
uint64_t E0 = INVALID_ID;
uint64_t E1 = INVALID_ID;
uint64_t E2 = INVALID_ID;
uint64_t E3 = INVALID_ID;
uint64_t FREE0 = INVALID_ID;
uint64_t FREE1 = INVALID_ID;
uint64_t FREE2 = INVALID_ID;
uint64_t FREE3 = INVALID_ID;
uint64_t CONST = INVALID_ID;
uint64_t CTX = INVALID_ID;
uint64_t GAS = INVALID_ID;
uint64_t JMP = INVALID_ID;
uint64_t JMPC = INVALID_ID;
uint64_t MAXMEM = INVALID_ID;
uint64_t PC = INVALID_ID;
uint64_t SP = INVALID_ID;
uint64_t SR = INVALID_ID;
uint64_t arith = INVALID_ID;
uint64_t assert = INVALID_ID;
uint64_t bin = INVALID_ID;
uint64_t comparator = INVALID_ID;
uint64_t ecRecover = INVALID_ID;
uint64_t hashE = INVALID_ID;
uint64_t hashRD = INVALID_ID;
uint64_t hashWR = INVALID_ID;
uint64_t inA = INVALID_ID;
uint64_t inB = INVALID_ID;
uint64_t inC = INVALID_ID;
uint64_t inD = INVALID_ID;
uint64_t inE = INVALID_ID;
uint64_t inCTX = INVALID_ID;
uint64_t inFREE = INVALID_ID;
uint64_t inGAS = INVALID_ID;
uint64_t inMAXMEM = INVALID_ID;
uint64_t inPC = INVALID_ID;
uint64_t inSP = INVALID_ID;
uint64_t inSR = INVALID_ID;
uint64_t inSTEP = INVALID_ID;
uint64_t inc = INVALID_ID;
uint64_t dec2 = INVALID_ID;
uint64_t ind = INVALID_ID;
uint64_t isCode = INVALID_ID;
uint64_t isMaxMem = INVALID_ID;
uint64_t isMem = INVALID_ID;
uint64_t isNeg = INVALID_ID;
uint64_t isStack = INVALID_ID;
uint64_t mRD = INVALID_ID;
uint64_t mWR = INVALID_ID;
uint64_t neg = INVALID_ID;
uint64_t offset = INVALID_ID;
uint64_t opcodeRomMap = INVALID_ID;
uint64_t sRD = INVALID_ID;
uint64_t sWR = INVALID_ID;
uint64_t setA = INVALID_ID;
uint64_t setB = INVALID_ID;
uint64_t setC = INVALID_ID;
uint64_t setD = INVALID_ID;
uint64_t setE = INVALID_ID;
uint64_t setCTX = INVALID_ID;
uint64_t setGAS = INVALID_ID;
uint64_t setMAXMEM = INVALID_ID;
uint64_t setPC = INVALID_ID;
uint64_t setSP = INVALID_ID;
uint64_t setSR = INVALID_ID;
uint64_t shl = INVALID_ID;
uint64_t shr = INVALID_ID;
uint64_t useCTX = INVALID_ID;
uint64_t zkPC = INVALID_ID;
uint64_t byte4_freeIN = INVALID_ID;
uint64_t byte4_out = INVALID_ID;

// TODO: check if map performance is better
/* Initializes the variable that contains the polynomial ID */
void addPol(string &name, uint64_t id)
{
         if (name=="main.A0") A0 = id;
    else if (name=="main.A1") A1 = id;
    else if (name=="main.A2") A2 = id;
    else if (name=="main.A3") A3 = id;
    else if (name=="main.B0") B0 = id;
    else if (name=="main.B1") B1 = id;
    else if (name=="main.B2") B2 = id;
    else if (name=="main.B3") B3 = id;
    else if (name=="main.C0") C0 = id;
    else if (name=="main.C1") C1 = id;
    else if (name=="main.C2") C2 = id;
    else if (name=="main.C3") C3 = id;
    else if (name=="main.D0") D0 = id;
    else if (name=="main.D1") D1 = id;
    else if (name=="main.D2") D2 = id;
    else if (name=="main.D3") D3 = id;
    else if (name=="main.E0") E0 = id;
    else if (name=="main.E1") E1 = id;
    else if (name=="main.E2") E2 = id;
    else if (name=="main.E3") E3 = id;
    else if (name=="main.FREE0") FREE0 = id;
    else if (name=="main.FREE1") FREE1 = id;
    else if (name=="main.FREE2") FREE2 = id;
    else if (name=="main.FREE3") FREE3 = id;
    else if (name=="main.CONST") CONST = id;
    else if (name=="main.CTX") CTX = id;
    else if (name=="main.GAS") GAS = id;
    else if (name=="main.JMP") JMP = id;
    else if (name=="main.JMPC") JMPC = id;
    else if (name=="main.MAXMEM") MAXMEM = id;
    else if (name=="main.PC") PC = id;
    else if (name=="main.SP") SP = id;
    else if (name=="main.SR") SR = id;
    else if (name=="main.arith") arith = id;
    else if (name=="main.assert") assert = id;
    else if (name=="main.bin") bin = id;
    else if (name=="main.comparator") comparator = id;
    else if (name=="main.ecRecover") ecRecover = id;
    else if (name=="main.hashE") hashE = id;
    else if (name=="main.hashRD") hashRD = id;
    else if (name=="main.hashWR") hashWR = id;
    else if (name=="main.inA") inA = id;
    else if (name=="main.inB") inB = id;
    else if (name=="main.inC") inC = id;
    else if (name=="main.inD") inD = id;
    else if (name=="main.inE") inE = id;
    else if (name=="main.inCTX") inCTX = id;
    else if (name=="main.inFREE") inFREE = id;
    else if (name=="main.inGAS") inGAS = id;
    else if (name=="main.inMAXMEM") inMAXMEM = id;
    else if (name=="main.inPC") inPC = id;
    else if (name=="main.inSP") inSP = id;
    else if (name=="main.inSR") inSR = id;
    else if (name=="main.inSTEP") inSTEP = id;
    else if (name=="main.inc") inc = id;
    else if (name=="main.dec") dec2 = id;
    else if (name=="main.ind") ind = id;
    else if (name=="main.isCode") isCode = id;
    else if (name=="main.isMaxMem") isMaxMem = id;
    else if (name=="main.isMem") isMem = id;
    else if (name=="main.isNeg") isNeg = id;
    else if (name=="main.isStack") isStack = id;
    else if (name=="main.mRD") mRD = id;
    else if (name=="main.mWR") mWR = id;
    else if (name=="main.neg") neg = id;
    else if (name=="main.offset") offset = id;
    else if (name=="main.opcodeRomMap") opcodeRomMap = id;
    else if (name=="main.sRD") sRD = id;
    else if (name=="main.sWR") sWR = id;
    else if (name=="main.setA") setA = id;
    else if (name=="main.setB") setB = id;
    else if (name=="main.setC") setC = id;
    else if (name=="main.setD") setD = id;
    else if (name=="main.setE") setE = id;
    else if (name=="main.setCTX") setCTX = id;
    else if (name=="main.setGAS") setGAS = id;
    else if (name=="main.setMAXMEM") setMAXMEM = id;
    else if (name=="main.setPC") setPC = id;
    else if (name=="main.setSP") setSP = id;
    else if (name=="main.setSR") setSR = id;
    else if (name=="main.shl") shl = id;
    else if (name=="main.shr") shr = id;
    else if (name=="main.useCTX") useCTX = id;
    else if (name=="main.zkPC") zkPC = id;
    else if (name=="byte4.freeIN") byte4_freeIN = id;
    else if (name=="byte4.out") byte4_out = id;
    else
    {
        cerr << "Error: pol() could not find a polynomial for name: " << name << ", id: " << id << endl;
        exit(-1); // TODO: Should we kill the process?
    }
}

/* 
    This function creates an array of polynomials and a mapping that maps the reference name in pil to the polynomial
*/
void createPols(Context &ctx, json &pil)
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
                addPol(key,id);
                addedPols++;
                cout << "Added polynomial " << addedPols << ": " << key << " with ID " << id << endl;
            }
        }

    }
}

void mapPols(Context &ctx)
{
    int fd = open(ctx.outputFile.c_str(), O_CREAT|O_RDWR|O_TRUNC, 0666);
    if (fd < 0)
    {
        cout << "Error: closePols() failed opening output file: " << ctx.outputFile << endl;
        exit(-1);
    }

    // Seek the last byte of the file
    int result = lseek(fd, sizeof(tExecutorOutput)-1, SEEK_SET);
    if (result == -1)
    {
        cout << "Error: closePols() failed calling lseek() of file: " << ctx.outputFile << endl;
        exit(-1);
    }

    // Write a 0 at the last byte of the file, to set its size
    result = write(fd, "", 1);
    if (result < 0)
    {
        cout << "Error: closePols() failed calling write() of file: " << ctx.outputFile << endl;
        exit(-1);
    }

    // TODO: Should we write the whole content of the file to 0?

    ctx.pPols = (tExecutorOutput *)mmap( NULL, sizeof(tExecutorOutput), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (ctx.pPols == MAP_FAILED)
    {
        cout << "Error: closePols() failed calling mmap() of file: " << ctx.outputFile << endl;
        exit(-1);
    }
    close(fd);
}

void unmapPols (Context &ctx)
{
    int err = munmap(ctx.pPols, sizeof(tExecutorOutput));
    if (err != 0)
    {
        cout << "Error: closePols() failed calling munmap() of file: " << ctx.outputFile << endl;
        exit(-1);
    }
    ctx.pPols = NULL;
}