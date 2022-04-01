#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <gmpxx.h>

#include "config.hpp"
#include "executor.hpp"
#include "rom_line.hpp"
#include "rom_command.hpp"
#include "rom.hpp"
#include "context.hpp"
#include "pols.hpp"
#include "input.hpp"
#include "scalar.hpp"
#include "utils.hpp"
#include "eval_command.hpp"
#include "smt.hpp"
#include "ecrecover/ecrecover.hpp"
#include "ff/ff.hpp"

using namespace std;
using json = nlohmann::json;

#define MEM_OFFSET 0x300000000
#define STACK_OFFSET 0x200000000
#define CODE_OFFSET 0x100000000
#define CTX_OFFSET 0x400000000

void Executor::execute (const Input &input, Pols &cmPols, Database &db, Counters &counters, SmtActionList &smtActionList, MemoryAccessList &memoryAccessList, bool bFastMode)
{
    TimerStart(EXECUTE_INITIALIZATION);
    
#ifdef LOG_TIME
    uint64_t poseidonTime=0, poseidonTimes=0;
    uint64_t smtTime=0, smtTimes=0;
    uint64_t ecRecoverTime=0, ecRecoverTimes=0;
    uint64_t keccakTime=0, keccakTimes=0;
#endif

    // Create context and store a finite field reference in it
    Context ctx(fr, cmPols, input, db);
    ctx.prime = prime;

    /* Sets first evaluation of all polynomials to zero */
    initState(ctx);

#ifdef USE_LOCAL_STORAGE
    /* Copy input storage content into context storage */
    map< FieldElement, mpz_class, CompareFe>::iterator itsto;
    for (itsto=input.sto.begin(); itsto!=input.sto.end(); itsto++)
    {
        fe = itsto->first;
        ctx.sto[fe] = itsto->second;
    }
#endif

    if (input.db.size() > 0)
    {
        /* Copy input database content into context database */
        map< string, vector<FieldElement> >::const_iterator it;
        for (it=input.db.begin(); it!=input.db.end(); it++)
        {
            ctx.db.create(it->first, it->second);
        }
    }

    // opN are local, uncommitted polynomials
    FieldElement op0, op1, op2, op3, op4, op5, op6, op7;

    // Zero-knowledge program counter
    uint64_t zkPC = 0;

    TimerStopAndLog(EXECUTE_INITIALIZATION);

    TimerStart(EXECUTE_LOOP);

    uint64_t i;
    uint64_t nexti;
    for (uint64_t ii=0; ii<NEVALUATIONS; ii++)
    {
        if (bFastMode)
        {
            i = ii%2;
            nexti = (i+1)%2;
            pol(FREE0)[i] = fr.zero();
            pol(FREE1)[i] = fr.zero();
            pol(FREE2)[i] = fr.zero();
            pol(FREE3)[i] = fr.zero();
            pol(FREE4)[i] = fr.zero();
            pol(FREE5)[i] = fr.zero();
            pol(FREE6)[i] = fr.zero();
            pol(FREE7)[i] = fr.zero();
        }
        else
        {
            i = ii;
            // Calculate nexti to write the next evaluation register values according to setX
            // The registers of the evaluation 0 will be overwritten with the values from the last evaluation, closing the evaluation circle
            nexti = (i+1)%NEVALUATIONS;
        }
        zkPC = pol(zkPC)[i]; // This is the read line of ZK code
        ctx.zkPC = zkPC;

        // ctx.step is used inside evaluateCommand() to find the current value of the registers, e.g. pols(A0)[ctx.step]
        ctx.step = i;

#ifdef LOG_STEPS
        //cout << "--> Starting step: " << i << " with zkPC: " << zkPC << endl;
#endif

#ifdef LOG_FILENAME
        // Store fileName and line
        ctx.fileName = rom.line[zkPC].fileName;
        ctx.line = rom.line[zkPC].line;
#endif

        // Evaluate the list cmdBefore commands, and any children command, recursively
        for (uint64_t j=0; j<rom.line[zkPC].cmdBefore.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom.line[zkPC].cmdBefore[j], cr);
        }

        // Initialize the local registers to zero
        op0 = fr.zero();
        op1 = fr.zero();
        op2 = fr.zero();
        op3 = fr.zero();
        op4 = fr.zero();
        op5 = fr.zero();
        op6 = fr.zero();
        op7 = fr.zero();

        // inX adds the corresponding register values to the op local register set, multiplied by inX
        // In case several inXs are set to !=0, those values will be added together to opN
        // e.g. op0 = inX*X0 + inY*Y0 + inZ*Z0 +...

        // If inA, op = op + inA*A
        if (!fr.isZero(rom.line[zkPC].inA))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inA, pol(A0)[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inA, pol(A1)[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inA, pol(A2)[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inA, pol(A3)[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inA, pol(A4)[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inA, pol(A5)[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inA, pol(A6)[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inA, pol(A7)[i]));

            pol(inA)[i] = rom.line[zkPC].inA;

#ifdef LOG_INX
            cout << "inA op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inB, op = op + inB*B
        if (!fr.isZero(rom.line[zkPC].inB))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inB, pol(B0)[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inB, pol(B1)[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inB, pol(B2)[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inB, pol(B3)[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inB, pol(B4)[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inB, pol(B5)[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inB, pol(B6)[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inB, pol(B7)[i]));

            pol(inB)[i] = rom.line[zkPC].inB;
            
#ifdef LOG_INX
            cout << "inB op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inA, op = op + inA*A
        if (!fr.isZero(rom.line[zkPC].inC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inC, pol(C0)[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inC, pol(C1)[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inC, pol(C2)[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inC, pol(C3)[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inC, pol(C4)[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inC, pol(C5)[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inC, pol(C6)[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inC, pol(C7)[i]));

            pol(inC)[i] = rom.line[zkPC].inC;
            
#ifdef LOG_INX
            cout << "inC op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inD, op = op + inD*D
        if (!fr.isZero(rom.line[zkPC].inD))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inD, pol(D0)[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inD, pol(D1)[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inD, pol(D2)[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inD, pol(D3)[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inD, pol(D4)[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inD, pol(D5)[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inD, pol(D6)[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inD, pol(D7)[i]));

            pol(inD)[i] = rom.line[zkPC].inD;
            
#ifdef LOG_INX
            cout << "inD op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inE, op = op + inE*E
        if (!fr.isZero(rom.line[zkPC].inE))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inE, pol(E0)[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inE, pol(E1)[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inE, pol(E2)[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inE, pol(E3)[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inE, pol(E4)[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inE, pol(E5)[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inE, pol(E6)[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inE, pol(E7)[i]));

            pol(inE)[i] = rom.line[zkPC].inE;
            
#ifdef LOG_INX
            cout << "inE op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSR, op = op + inSR*SR
        if (!fr.isZero(rom.line[zkPC].inSR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSR, pol(SR0)[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inSR, pol(SR1)[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inSR, pol(SR2)[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inSR, pol(SR3)[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inSR, pol(SR4)[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inSR, pol(SR5)[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inSR, pol(SR6)[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inSR, pol(SR7)[i]));

            pol(inSR)[i] = rom.line[zkPC].inSR;
            
#ifdef LOG_INX
            cout << "inSR op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCTX, op = op + inCTX*CTX
        if (!fr.isZero(rom.line[zkPC].inCTX))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCTX, pol(CTX)[i]));
            pol(inCTX)[i] = rom.line[zkPC].inCTX;
#ifdef LOG_INX
            cout << "inCTX op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSP, op = op + inSP*SP
        if (!fr.isZero(rom.line[zkPC].inSP))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSP, pol(SP)[i]));
            pol(inSP)[i] = rom.line[zkPC].inSP;
#ifdef LOG_INX
            cout << "inSP op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inPC, op = op + inPC*PC
        if (!fr.isZero(rom.line[zkPC].inPC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inPC, pol(PC)[i]));
            pol(inPC)[i] = rom.line[zkPC].inPC;
#ifdef LOG_INX
            cout << "inPC op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inGAS, op = op + inGAS*GAS
        if (!fr.isZero(rom.line[zkPC].inGAS))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inGAS, pol(GAS)[i]));
            pol(inGAS)[i] = rom.line[zkPC].inGAS;
#ifdef LOG_INX
            cout << "inGAS op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inMAXMEM, op = op + inMAXMEM*MAXMEM
        if (!fr.isZero(rom.line[zkPC].inMAXMEM))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inMAXMEM, pol(MAXMEM)[i]));
            pol(inMAXMEM)[i] = rom.line[zkPC].inMAXMEM;
#ifdef LOG_INX
            cout << "inMAXMEM op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSTEP, op = op + inSTEP*STEP
        if (!fr.isZero(rom.line[zkPC].inSTEP))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSTEP, i));
            pol(inSTEP)[i] = rom.line[zkPC].inSTEP;
#ifdef LOG_INX
            cout << "inSTEP op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCONST, op = op + CONST
        if (rom.line[zkPC].bConstPresent)
        {
            op0 = fr.add(op0, rom.line[zkPC].CONST);
            pol(CONST)[i] = rom.line[zkPC].CONST;
#ifdef LOG_INX
            cout << "CONST op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        uint32_t addrRel = 0;
        uint64_t addr = 0;

        // If address is involved, load offset into addr
        if (rom.line[zkPC].mRD==1 || rom.line[zkPC].mWR==1 || rom.line[zkPC].hashRD==1 || rom.line[zkPC].hashWR==1 || rom.line[zkPC].hashE==1 || rom.line[zkPC].JMP==1 || rom.line[zkPC].JMPC==1) {
            if (rom.line[zkPC].ind == 1)
            {
                addrRel = fe2n(fr, prime, pol(E0)[i]);
            }
            if (rom.line[zkPC].bOffsetPresent && rom.line[zkPC].offset!=0)
            {
                // If offset is possitive, and the sum is too big, fail
                if (rom.line[zkPC].offset>0 && (uint64_t(addrRel)+uint64_t(rom.line[zkPC].offset))>=0x100000000)
                {
                    cerr << "Error: addrRel >= 0x100000000 ln: " << ctx.zkPC << endl;
                    exit(-1);                  
                }
                // If offset is negative, and its modulo is bigger than addrRel, fail
                if (rom.line[zkPC].offset<0 && (-rom.line[zkPC].offset)>addrRel)
                {
                    cerr << "Error: addrRel < 0 ln: " << ctx.zkPC << endl;
                    exit(-1);
                }
                addrRel += rom.line[zkPC].offset;
            }
            addr = addrRel;
#ifdef LOG_ADDR
            cout << "Any addr=" << addr << endl;
#endif
        }

        // If useCTX, addr = addr + CTX*CTX_OFFSET
        if (rom.line[zkPC].useCTX == 1) {
            addr += pol(CTX)[i]*CTX_OFFSET;
            pol(useCTX)[i] = 1;
#ifdef LOG_ADDR
            cout << "useCTX addr=" << addr << endl;
#endif
        }

        // If isCode, addr = addr + CODE_OFFSET
        if (rom.line[zkPC].isCode == 1) {
            addr += CODE_OFFSET;
            pol(isCode)[i] = 1;
#ifdef LOG_ADDR
            cout << "isCode addr=" << addr << endl;
#endif
        }

        // If isStack, addr = addr + STACK_OFFSET
        if (rom.line[zkPC].isStack == 1) {
            addr += STACK_OFFSET;
            pol(isStack)[i] = 1;
#ifdef LOG_ADDR
            cout << "isStack addr=" << addr << endl;
#endif
        }

        // If isMem, addr = addr + MEM_OFFSET
        if (rom.line[zkPC].isMem == 1) {
            addr += MEM_OFFSET;
            pol(isMem)[i] = 1;
#ifdef LOG_ADDR
            cout << "isMem addr=" << addr << endl;
#endif
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].incCode != 0) pol(incCode)[i] = rom.line[zkPC].incCode;
        if (rom.line[zkPC].incStack != 0) pol(incStack)[i] = rom.line[zkPC].incStack;
        if (rom.line[zkPC].ind == 1) pol(ind)[i] = 1;

        // If offset, record it in byte4
        if (rom.line[zkPC].bOffsetPresent && (rom.line[zkPC].offset!=0)) {
            pol(offset)[i] = rom.line[zkPC].offset;
        }

        // If inFREE, calculate the free value, and add it to op
        if (!fr.isZero(rom.line[zkPC].inFREE))
        {
            // freeInTag must be present
            if (rom.line[zkPC].freeInTag.isPresent == false) {
                cerr << "Error: Instruction with freeIn without freeInTag:" << ctx.zkPC << endl;
                exit(-1);
            }

            // Store free value here, and add it to op later
            FieldElement fi0;
            FieldElement fi1;
            FieldElement fi2;
            FieldElement fi3;
            FieldElement fi4;
            FieldElement fi5;
            FieldElement fi6;
            FieldElement fi7;

            // If there is no operation specified in freeInTag.op, then get the free value directly from the corresponding source
            if (rom.line[zkPC].freeInTag.op == "") {
                uint64_t nHits = 0;

                // If mRD (memory read) get fi=mem[addr], if it exsists
                if (rom.line[zkPC].mRD == 1)
                {
                    if (ctx.mem.find(addr) != ctx.mem.end()) {
#ifdef LOG_MEMORY
                        cout << "Memory read mRD: addr:" << addr << " " << printFea(ctx, ctx.mem[addr]) << endl;
#endif
                        fi0 = ctx.mem[addr].fe0;
                        fi1 = ctx.mem[addr].fe1;
                        fi2 = ctx.mem[addr].fe2;
                        fi3 = ctx.mem[addr].fe3;
                        fi4 = ctx.mem[addr].fe4;
                        fi5 = ctx.mem[addr].fe5;
                        fi6 = ctx.mem[addr].fe6;
                        fi7 = ctx.mem[addr].fe7;

                        MemoryAccess memoryAccess;
                        memoryAccess.bIsWrite = false;
                        memoryAccess.address = addr;
                        memoryAccess.pc = i;
                        memoryAccess.fe0 = fi0;
                        memoryAccess.fe1 = fi1;
                        memoryAccess.fe2 = fi2;
                        memoryAccess.fe3 = fi3;
                        memoryAccess.fe4 = fi4;
                        memoryAccess.fe5 = fi5;
                        memoryAccess.fe6 = fi6;
                        memoryAccess.fe7 = fi7;
                        memoryAccessList.access.push_back(memoryAccess);

                    } else {
                        fi0 = fr.zero();
                        fi1 = fr.zero();
                        fi2 = fr.zero();
                        fi3 = fr.zero();
                        fi4 = fr.zero();
                        fi5 = fr.zero();
                        fi6 = fr.zero();
                        fi7 = fr.zero();
                    }
                    nHits++;
                }

                // If sRD (storage read) get a poseidon hash, and read fi=sto[hash]
                if (rom.line[zkPC].sRD == 1)
                {
                    FieldElement keyV0[12];
                    keyV0[0] = pol(A0)[i];
                    keyV0[1] = pol(A1)[i];
                    keyV0[2] = pol(A2)[i];
                    keyV0[3] = pol(A3)[i];
                    keyV0[4] = pol(A4)[i];
                    keyV0[5] = pol(A5)[i];
                    keyV0[6] = pol(B0)[i];
                    keyV0[7] = pol(B1)[i];
                    keyV0[8] = 0;
                    keyV0[9] = 0;
                    keyV0[10] = 0;
                    keyV0[11] = 0;

                    FieldElement keyV1[12];
                    keyV1[0] = pol(C0)[i];
                    keyV1[1] = pol(C1)[i];
                    keyV1[2] = pol(C2)[i];
                    keyV1[3] = pol(C3)[i];
                    keyV1[4] = pol(C4)[i];
                    keyV1[5] = pol(C5)[i];
                    keyV1[6] = pol(C6)[i];
                    keyV1[7] = pol(C7)[i];
                    keyV1[8] = 0;
                    keyV1[9] = 0;
                    keyV1[10] = 0;
                    keyV1[11] = 0;

                    
#ifdef LOG_TIME
                    struct timeval t;
                    gettimeofday(&t, NULL);
#endif

                    // Call poseidon and get the hash key
                    poseidon.hash(keyV0);
                    poseidon.hash(keyV1);
                    keyV0[4] = keyV1[0];
                    keyV0[5] = keyV1[1];
                    keyV0[6] = keyV1[2];
                    keyV0[7] = keyV1[3];
                    keyV0[8] = 0;
                    keyV0[9] = 0;
                    keyV0[10] = 0;
                    keyV0[11] = 0;
                    poseidon.hash(keyV0);
                    ctx.lastSWrite.key[0] = keyV0[0];
                    ctx.lastSWrite.key[1] = keyV0[1];
                    ctx.lastSWrite.key[2] = keyV0[2];
                    ctx.lastSWrite.key[3] = keyV0[3];
#ifdef LOG_TIME
                    poseidonTime += TimeDiff(t);
                    poseidonTimes+=3;
#endif
                    // Increment counter
                    counters.hashPoseidon++;

#ifdef LOG_STORAGE
                    cout << "Storage read sRD got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif 

#ifdef USE_LOCAL_STORAGE
                    //printStorage(ctx);
                    // Check that storage entry exists
                    if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                    {
                        cerr << "Error: Storage not initialized, key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.zkPC << " step: " << ctx.step << endl;
                        exit(-1);
                    }

                    //cout << "STORAGE1 i:" << i << " hash:" << fr.toString(ctx.lastSWrite.key, 16) << " value:" << ctx.sto[ctx.lastSWrite.key].get_str(16) << endl;

                    //SmtGetResult smtGetResult;
                    //smt.get(ctx.fr, ctx.db, pol(SR)[i], ctx.lastSWrite.key, smtGetResult);
                    //cout << "STORAGE2 i:" << i << " hash:" << fr.toString(ctx.lastSWrite.key, 16) << " value:" << smtGetResult.value.get_str(16) << endl;

                    // Read the value from storage, and store it in fin
                    scalar2fea(fr, ctx.sto[ctx.lastSWrite.key], fi0, fi1, fi2, fi3);
#else
                    FieldElement oldRoot[4];
                    sr8to4(fr, pol(SR0)[i], pol(SR1)[i], pol(SR2)[i], pol(SR3)[i], pol(SR4)[i], pol(SR5)[i], pol(SR6)[i], pol(SR7)[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);
                    
                    SmtGetResult smtGetResult;
                    smt.get(ctx.db, oldRoot, ctx.lastSWrite.key, smtGetResult);
                    //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;

                    SmtAction smtAction;
                    smtAction.bIsSet = false;
                    smtAction.getResult = smtGetResult;
                    smtActionList.action.push_back(smtAction);
                    
                    scalar2fea(fr, smtGetResult.value, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
#endif

                    nHits++;
#ifdef LOG_STORAGE
                    cout << "Storage read sRD read from key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " value:" << fr.toString(fi3, 16) << ":" << fr.toString(fi2, 16) << ":" << fr.toString(fi1, 16) << ":" << fr.toString(fi0, 16) << endl;
#endif 
                }

                // If sWR (storage write) calculate the poseidon hash key, check its entry exists in storage, and update new root hash
                if (rom.line[zkPC].sWR == 1)
                {
                    // reset lastSWrite
                    ctx.lastSWrite.key[0] = fr.zero();
                    ctx.lastSWrite.key[1] = fr.zero();
                    ctx.lastSWrite.key[2] = fr.zero();
                    ctx.lastSWrite.key[3] = fr.zero();
                    ctx.lastSWrite.newRoot[0] = fr.zero();
                    ctx.lastSWrite.newRoot[1] = fr.zero();
                    ctx.lastSWrite.newRoot[2] = fr.zero();
                    ctx.lastSWrite.newRoot[3] = fr.zero();
                    ctx.lastSWrite.step = 0;
                    
                    FieldElement keyV0[12];
                    keyV0[0] = pol(A0)[i];
                    keyV0[1] = pol(A1)[i];
                    keyV0[2] = pol(A2)[i];
                    keyV0[3] = pol(A3)[i];
                    keyV0[4] = pol(A4)[i];
                    keyV0[5] = pol(A5)[i];
                    keyV0[6] = pol(B0)[i];
                    keyV0[7] = pol(B1)[i];
                    keyV0[8] = 0;
                    keyV0[9] = 0;
                    keyV0[10] = 0;
                    keyV0[11] = 0;

                    FieldElement keyV1[12];
                    keyV1[0] = pol(C0)[i];
                    keyV1[1] = pol(C1)[i];
                    keyV1[2] = pol(C2)[i];
                    keyV1[3] = pol(C3)[i];
                    keyV1[4] = pol(C4)[i];
                    keyV1[5] = pol(C5)[i];
                    keyV1[6] = pol(C6)[i];
                    keyV1[7] = pol(C7)[i];
                    keyV1[8] = 0;
                    keyV1[9] = 0;
                    keyV1[10] = 0;
                    keyV1[11] = 0;
#ifdef LOG_TIME
                    struct timeval t;
                    gettimeofday(&t, NULL);
#endif
                    // Call poseidon
                    poseidon.hash(keyV0);
                    poseidon.hash(keyV1);
                    keyV0[4] = keyV1[0];
                    keyV0[5] = keyV1[1];
                    keyV0[6] = keyV1[2];
                    keyV0[7] = keyV1[3];
                    keyV0[8] = 0;
                    keyV0[9] = 0;
                    keyV0[10] = 0;
                    keyV0[11] = 0;
                    poseidon.hash(keyV0);
                    ctx.lastSWrite.key[0] = keyV0[0];
                    ctx.lastSWrite.key[1] = keyV0[1];
                    ctx.lastSWrite.key[2] = keyV0[2];
                    ctx.lastSWrite.key[3] = keyV0[3];
#ifdef LOG_TIME
                    poseidonTime += TimeDiff(t);
                    poseidonTimes++;
#endif
                    // Increment counter
                    counters.hashPoseidon++;
#ifdef LOG_STORAGE
                    cout << "Storage write sWR got poseidon key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << endl;
#endif                    
#ifdef USE_LOCAL_STORAGE
                    // Check that storage entry exists
                    if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                    {
                        cerr << "Error: Storage write sWR not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.zkPC << endl;
                        exit(-1);
                    }
#endif

                    // Call SMT to get the new Merkel Tree root hash
                    SmtSetResult smtSetResult;
                    mpz_class scalarD;
                    fea2scalar(fr, scalarD, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i], pol(D4)[i], pol(D5)[i], pol(D6)[i], pol(D7)[i]);
#ifdef LOG_TIME
                    gettimeofday(&t, NULL);
#endif
                    FieldElement oldRoot[4];
                    sr8to4(fr, pol(SR0)[i], pol(SR1)[i], pol(SR2)[i], pol(SR3)[i], pol(SR4)[i], pol(SR5)[i], pol(SR6)[i], pol(SR7)[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);
                    
                    smt.set(ctx.db, oldRoot, ctx.lastSWrite.key, scalarD, smtSetResult);

                    SmtAction smtAction;
                    smtAction.bIsSet = true;
                    smtAction.setResult = smtSetResult;
                    smtActionList.action.push_back(smtAction);
#ifdef LOG_TIME
                    smtTime += TimeDiff(t);
                    smtTimes++;
#endif
                    ctx.lastSWrite.newRoot[0] = smtSetResult.newRoot[0];
                    ctx.lastSWrite.newRoot[1] = smtSetResult.newRoot[1];
                    ctx.lastSWrite.newRoot[2] = smtSetResult.newRoot[2];
                    ctx.lastSWrite.newRoot[3] = smtSetResult.newRoot[3];
                    ctx.lastSWrite.step = i;

                    sr4to8(fr, smtSetResult.newRoot[0], smtSetResult.newRoot[1], smtSetResult.newRoot[2], smtSetResult.newRoot[3], fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
#ifdef LOG_STORAGE
                    cout << "Storage write sWR stored at key: " << ctx.fr.toString(ctx.lastSWrite.key, 16) << " newRoot: " << fr.toString(res.newRoot, 16) << endl;
#endif
                }

                // If hashRD (hash read)
                if (rom.line[zkPC].hashRD == 1)
                {
                    // Check the entry addr exists in hash
                    if (ctx.hash.find(addr) == ctx.hash.end()) {
                        cerr << "Error: Hash address not initialized" << endl;
                        exit(-1);
                    }

                    // Read fi=hash[addr]
                    mpz_class auxScalar(ctx.hash[addr].hash);
                    scalar2fea(fr, auxScalar, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
#ifdef LOG_HASH
                    cout << "Hash read hashRD: addr:" << addr << " hash:" << auxScalar.get_str(16) << endl;
#endif
                }

                // If ecRecover, build the transaction signature, recover the address that generated it, and copy fi=recovered address
                if (rom.line[zkPC].ecRecover == 1) {

                    // Increment counter
                    counters.ecRecover++;
                    
                    mpz_class aux;
                    
                    // Get d=A
                    fea2scalar(fr, aux, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i], pol(A4)[i], pol(A5)[i], pol(A6)[i], pol(A7)[i]);
                    string d = NormalizeTo0xNFormat(aux.get_str(16),64);

                    // Signature string = 0x + r(32B) + s(32B) + v(1B) = 0x + 130chars
                    fea2scalar(fr, aux, pol(B0)[i], pol(B1)[i], pol(B2)[i], pol(B3)[i], pol(B4)[i], pol(B5)[i], pol(B6)[i], pol(B7)[i]);
                    string r = NormalizeToNFormat(aux.get_str(16),64);
                    fea2scalar(fr, aux, pol(C0)[i], pol(C1)[i], pol(C2)[i], pol(C3)[i], pol(C4)[i], pol(C5)[i], pol(C6)[i], pol(C7)[i]);
                    string s = NormalizeToNFormat(aux.get_str(16),64);
                    aux = fe2n(fr, prime, pol(D0)[i]);
                    string v = NormalizeToNFormat(aux.get_str(16),2);
                    string signature = "0x" + r + s + v;

                    /* Return the address associated with the public key signature from elliptic curve signature.
                       Signature parts: r: first 32 bytes of signature; s: second 32 bytes of signature; v: final 1 byte of signature.
                       Hash: d: 32 bytes. */
#ifdef LOG_TIME
                    struct timeval t;
                    gettimeofday(&t, NULL);
#endif
                    string ecResult = ecrecover(signature, d);
                    // TODO: Consider calling ecrecover in parallel to save time
#ifdef LOG_TIME
                    ecRecoverTime += TimeDiff(t);
                    ecRecoverTimes++;
#endif 
                    mpz_class raddr(ecResult);
                    scalar2fea(fr, raddr, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
                }

                // If shl, shift A, D bytes to the left, and discard highest bits
                if (rom.line[zkPC].shl == 1)
                {
                    // Read a=A
                    mpz_class a;
                    fea2scalar(fr, a, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i], pol(A4)[i], pol(A5)[i], pol(A6)[i], pol(A7)[i]);

                    // Read s=D
                    uint64_t s = fe2n(fr, prime, pol(D0)[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHL too big: " << ctx.zkPC << endl;
                        exit(-1);
                    }

                    // Calculate b = shift a, s bytes to the left
                    mpz_class band("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
                    mpz_class b;
                    b = (a << s*8) & band;

                    // Copy fi=b
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
                }

                // If shr, shift A, D bytes to the right
                if (rom.line[zkPC].shr == 1)
                {
                    // Read a=A
                    mpz_class a;
                    fea2scalar(fr, a, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i], pol(A4)[i], pol(A5)[i], pol(A6)[i], pol(A7)[i]);

                    // Read s=D
                    uint64_t s = fe2n(fr, prime, pol(D0)[i]);
                    if ((s>32) || (s<0)) {
                        cerr << "Error: SHR too big: " << ctx.zkPC << endl;
                        exit(-1);
                    }

                    // Calculate b = shift a, s bytes to the right
                    mpz_class b = a >> s*8;

                    // Copy fi=b
                    scalar2fea(fr, b, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                    nHits++;
                } 

                // Check that one and only one instruction has been requested
                if (nHits == 0) {
                    cerr << "Error: Empty freeIn without a valid instruction: " << ctx.zkPC << endl;
                    exit(-1);
                }
                if (nHits > 1) {
                    cerr << "Error: Only one instruction that requires freeIn is alllowed: " << ctx.zkPC << endl;
                }
            }
            // If freeInTag.op!="", then evaluate the requested command (recursively)
            else
            {
                // Call evalCommand()
                CommandResult cr;
                evalCommand(ctx, rom.line[zkPC].freeInTag, cr);

                // Copy fi=command result, depending on its type 
                if (cr.type == crt_fea) {
                    fi0 = cr.fea0;
                    fi1 = cr.fea1;
                    fi2 = cr.fea2;
                    fi3 = cr.fea3;
                    fi4 = cr.fea4;
                    fi5 = cr.fea5;
                    fi6 = cr.fea6;
                    fi7 = cr.fea7;
                } else if (cr.type == crt_fe) {
                    fi0 = cr.fe;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else if (cr.type == crt_scalar) {
                    scalar2fea(fr, cr.scalar, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                } else if (cr.type == crt_u16) {
                    fi0 = cr.u16;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else if (cr.type == crt_u32) {
                    fi0 = cr.u32;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else if (cr.type == crt_u64) {
                    fi0 = cr.u64;
                    fi1 = fr.zero();
                    fi2 = fr.zero();
                    fi3 = fr.zero();
                    fi4 = fr.zero();
                    fi5 = fr.zero();
                    fi6 = fr.zero();
                    fi7 = fr.zero();
                } else {
                    cerr << "Error: unexpected command result type: " << cr.type << endl;
                    exit(-1);
                }
            }

            // Store polynomial FREE=fi
            pol(FREE0)[i] = fi0;
            pol(FREE1)[i] = fi1;
            pol(FREE2)[i] = fi2;
            pol(FREE3)[i] = fi3;
            pol(FREE4)[i] = fi4;
            pol(FREE5)[i] = fi5;
            pol(FREE6)[i] = fi6;
            pol(FREE7)[i] = fi7;

            // op = op + inFREE*fi
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inFREE, fi0));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inFREE, fi1));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inFREE, fi2));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inFREE, fi3));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inFREE, fi4));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inFREE, fi5));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inFREE, fi6));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inFREE, fi7));

            // Copy ROM flags into the polynomials
            pol(inFREE)[i] = rom.line[zkPC].inFREE;
        }

        // If assert, check that A=op
        if (rom.line[zkPC].assert == 1)
        {
            if ( (!fr.eq(pol(A0)[i], op0)) ||
                 (!fr.eq(pol(A1)[i], op1)) ||
                 (!fr.eq(pol(A2)[i], op2)) ||
                 (!fr.eq(pol(A3)[i], op3)) ||
                 (!fr.eq(pol(A4)[i], op4)) ||
                 (!fr.eq(pol(A5)[i], op5)) ||
                 (!fr.eq(pol(A6)[i], op6)) ||
                 (!fr.eq(pol(A7)[i], op7)) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << ctx.zkPC << endl;
                cout << "A: " << fr.toString(pol(A7)[i], 16) << ":" << fr.toString(pol(A6)[i], 16) << ":" << fr.toString(pol(A5)[i], 16) << ":" << fr.toString(pol(A4)[i], 16) << ":" << fr.toString(pol(A3)[i], 16) << ":" << fr.toString(pol(A2)[i], 16) << ":" << fr.toString(pol(A1)[i], 16) << ":" << fr.toString(pol(A0)[i], 16) << endl;
                cout << "OP:" << fr.toString(op7, 16) << ":" << fr.toString(op6, 16) << ":" << fr.toString(op5, 16) << ":" << fr.toString(op4,16) << ":" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0,16) << endl;
                exit(-1);
            }
            pol(assert)[i] = 1;
#ifdef LOG_ASSERT
            cout << "assert" << endl;
#endif
        }

        // If setA, A'=op
        if (rom.line[zkPC].setA == 1) {
            pol(A0)[nexti] = op0;
            pol(A1)[nexti] = op1;
            pol(A2)[nexti] = op2;
            pol(A3)[nexti] = op3;
            pol(A4)[nexti] = op4;
            pol(A5)[nexti] = op5;
            pol(A6)[nexti] = op6;
            pol(A7)[nexti] = op7;
            pol(setA)[i] = 1;
#ifdef LOG_SETX
            cout << "setA A[nexti]=" << pol(A3)[nexti] << ":" << pol(A2)[nexti] << ":" << pol(A1)[nexti] << ":" << fr.toString(pol(A0)[nexti], 16) << endl;
#endif
        } else {
            pol(A0)[nexti] = pol(A0)[i];
            pol(A1)[nexti] = pol(A1)[i];
            pol(A2)[nexti] = pol(A2)[i];
            pol(A3)[nexti] = pol(A3)[i];
            pol(A4)[nexti] = pol(A4)[i];
            pol(A5)[nexti] = pol(A5)[i];
            pol(A6)[nexti] = pol(A6)[i];
            pol(A7)[nexti] = pol(A7)[i];
        }

        // If setB, B'=op
        if (rom.line[zkPC].setB == 1) {
            pol(B0)[nexti] = op0;
            pol(B1)[nexti] = op1;
            pol(B2)[nexti] = op2;
            pol(B3)[nexti] = op3;
            pol(B4)[nexti] = op4;
            pol(B5)[nexti] = op5;
            pol(B6)[nexti] = op6;
            pol(B7)[nexti] = op7;
            pol(setB)[i] = 1;
#ifdef LOG_SETX
            cout << "setB B[nexti]=" << pol(B3)[nexti] << ":" << pol(B2)[nexti] << ":" << pol(B1)[nexti] << ":" << fr.toString(pol(B0)[nexti], 16) << endl;
#endif
        } else {
            pol(B0)[nexti] = pol(B0)[i];
            pol(B1)[nexti] = pol(B1)[i];
            pol(B2)[nexti] = pol(B2)[i];
            pol(B3)[nexti] = pol(B3)[i];
            pol(B4)[nexti] = pol(B4)[i];
            pol(B5)[nexti] = pol(B5)[i];
            pol(B6)[nexti] = pol(B6)[i];
            pol(B7)[nexti] = pol(B7)[i];
        }

        // If setC, C'=op
        if (rom.line[zkPC].setC == 1) {
            pol(C0)[nexti] = op0;
            pol(C1)[nexti] = op1;
            pol(C2)[nexti] = op2;
            pol(C3)[nexti] = op3;
            pol(C4)[nexti] = op4;
            pol(C5)[nexti] = op5;
            pol(C6)[nexti] = op6;
            pol(C7)[nexti] = op7;
            pol(setC)[i] = 1;
#ifdef LOG_SETX
            cout << "setC C[nexti]=" << pol(C3)[nexti] << ":" << pol(C2)[nexti] << ":" << pol(C1)[nexti] << ":" << fr.toString(pol(C0)[nexti], 16) << endl;
#endif
        } else {
            pol(C0)[nexti] = pol(C0)[i];
            pol(C1)[nexti] = pol(C1)[i];
            pol(C2)[nexti] = pol(C2)[i];
            pol(C3)[nexti] = pol(C3)[i];
            pol(C4)[nexti] = pol(C4)[i];
            pol(C5)[nexti] = pol(C5)[i];
            pol(C6)[nexti] = pol(C6)[i];
            pol(C7)[nexti] = pol(C7)[i];
        }

        // If setD, D'=op
        if (rom.line[zkPC].setD == 1) {
            pol(D0)[nexti] = op0;
            pol(D1)[nexti] = op1;
            pol(D2)[nexti] = op2;
            pol(D3)[nexti] = op3;
            pol(D4)[nexti] = op4;
            pol(D5)[nexti] = op5;
            pol(D6)[nexti] = op6;
            pol(D7)[nexti] = op7;
            pol(setD)[i] = 1;
#ifdef LOG_SETX
            cout << "setD D[nexti]=" << pol(D3)[nexti] << ":" << pol(D2)[nexti] << ":" << pol(D1)[nexti] << ":" << fr.toString(pol(D0)[nexti], 16) << endl;
#endif
        } else {
            pol(D0)[nexti] = pol(D0)[i];
            pol(D1)[nexti] = pol(D1)[i];
            pol(D2)[nexti] = pol(D2)[i];
            pol(D3)[nexti] = pol(D3)[i];
            pol(D4)[nexti] = pol(D4)[i];
            pol(D5)[nexti] = pol(D5)[i];
            pol(D6)[nexti] = pol(D6)[i];
            pol(D7)[nexti] = pol(D7)[i];
        }
        
        // If setE, E'=op
        if (rom.line[zkPC].setE == 1) {
            pol(E0)[nexti] = op0;
            pol(E1)[nexti] = op1;
            pol(E2)[nexti] = op2;
            pol(E3)[nexti] = op3;
            pol(E4)[nexti] = op4;
            pol(E5)[nexti] = op5;
            pol(E6)[nexti] = op6;
            pol(E7)[nexti] = op7;
            pol(setE)[i] = 1;
#ifdef LOG_SETX
            cout << "setE E[nexti]=" << pol(E3)[nexti] << ":" << pol(E2)[nexti] << ":" << pol(E1)[nexti] << ":" << fr.toString(pol(E0)[nexti] ,16) << endl;
#endif
        } else {
            pol(E0)[nexti] = pol(E0)[i];
            pol(E1)[nexti] = pol(E1)[i];
            pol(E2)[nexti] = pol(E2)[i];
            pol(E3)[nexti] = pol(E3)[i];
            pol(E4)[nexti] = pol(E4)[i];
            pol(E5)[nexti] = pol(E5)[i];
            pol(E6)[nexti] = pol(E6)[i];
            pol(E7)[nexti] = pol(E7)[i];
        }

        // If setSR, SR'=op
        if (rom.line[zkPC].setSR == 1) {
            pol(SR0)[nexti] = op0;
            pol(SR1)[nexti] = op1;
            pol(SR2)[nexti] = op2;
            pol(SR3)[nexti] = op3;
            pol(SR4)[nexti] = op4;
            pol(SR5)[nexti] = op5;
            pol(SR6)[nexti] = op6;
            pol(SR7)[nexti] = op7;
            pol(setSR)[i] = 1;
#ifdef LOG_SETX
            cout << "setSR SR[nexti]=" << fr.toString(pol(SR)[nexti], 16) << endl;
#endif
        } else {
            pol(SR0)[nexti] = pol(SR0)[i];
            pol(SR1)[nexti] = pol(SR1)[i];
            pol(SR2)[nexti] = pol(SR2)[i];
            pol(SR3)[nexti] = pol(SR3)[i];
            pol(SR4)[nexti] = pol(SR4)[i];
            pol(SR5)[nexti] = pol(SR5)[i];
            pol(SR6)[nexti] = pol(SR6)[i];
            pol(SR7)[nexti] = pol(SR7)[i];
        }

        // If setCTX, CTX'=op
        if (rom.line[zkPC].setCTX == 1) {
            pol(CTX)[nexti] = fe2n(fr, prime, op0);
            pol(setCTX)[i] = 1;
#ifdef LOG_SETX
            cout << "setCTX CTX[nexti]=" << pol(CTX)[nexti] << endl;
#endif
        } else {
            pol(CTX)[nexti] = pol(CTX)[i];
        }

        // If setSP, SP'=op
        if (rom.line[zkPC].setSP == 1) {
            pol(SP)[nexti] = fe2n(fr, prime, op0);
            pol(setSP)[i] = 1;
#ifdef LOG_SETX
            cout << "setSP SP[nexti]=" << pol(SP)[nexti] << endl;
#endif
        } else {
            // SP' = SP + incStack
            if (rom.line[zkPC].incStack<0 || rom.line[zkPC].incStack>0xFFFF)
            {
                cerr << "Error: incStack cannot be added to an u16 polynomial: " << rom.line[zkPC].incStack << endl;
                exit(-1);
            }
            pol(SP)[nexti] = pol(SP)[i] + rom.line[zkPC].incStack;
        }

        // If setPC, PC'=op
        if (rom.line[zkPC].setPC == 1) {
            pol(PC)[nexti] = fe2n(fr, prime, op0);
            pol(setPC)[i] = 1;
#ifdef LOG_SETX
            cout << "setPC PC[nexti]=" << pol(PC)[nexti] << endl;
#endif
        } else {
            // PC' = PC + incCode
            if (rom.line[zkPC].incCode<0 || rom.line[zkPC].incCode>0xFFFF)
            {
                cerr << "Error: incCode cannot be added to an u16 polynomial: " << rom.line[zkPC].incCode << endl;
                exit(-1);
            }
            pol(PC)[nexti] = pol(PC)[i] + rom.line[zkPC].incCode;
        }

        // If JMPC, jump conditionally based on op value
        if (rom.line[zkPC].JMPC == 1) {
#ifdef LOG_JMP
            cout << "JMPC: op0=" << fr.toString(op0) << endl;
#endif
            int64_t o = fe2n(fr, prime, op0);
#ifdef LOG_JMP
            cout << "JMPC: o=" << o << endl;
#endif
            // If op<0, jump to addr: zkPC'=addr
            if (o < 0) {
                pol(isNeg)[i] = 1;
                pol(zkPC)[nexti] = addr;
                ctx.byte4[0x100000000 + o] = true;
#ifdef LOG_JMP
               cout << "Next zkPC(1)=" << pol(zkPC)[nexti] << endl;
#endif
            }
            // If op>=0, simply increase zkPC'=zkPC+1
            else
            {
                pol(zkPC)[nexti] = pol(zkPC)[i] + 1;
#ifdef LOG_JMP
                cout << "Next zkPC(2)=" << pol(zkPC)[nexti] << endl;
#endif
                ctx.byte4[o] = true;
            }
            pol(JMPC)[i] = 1;
        }
        // If JMP, directly jump zkPC'=addr
        else if (rom.line[zkPC].JMP == 1)
        {
            pol(zkPC)[nexti] = addr;
#ifdef LOG_JMP
            cout << "Next zkPC(3)=" << pol(zkPC)[nexti] << endl;
#endif
            pol(JMP)[i] = 1;
        }
        // Else, simply increase zkPC'=zkPC+1
        else
        {
            pol(zkPC)[nexti] = pol(zkPC)[i] + 1;
        }

        // Calculate the new max mem address, if any
        uint32_t maxMemCalculated = 0;
        uint32_t mm = pol(MAXMEM)[i];
        if (rom.line[zkPC].isMem==1)
        {
            if (addrRel>mm) {
                pol(isMaxMem)[i] = 1;
                maxMemCalculated = addrRel;
                ctx.byte4[maxMemCalculated - mm] = true;
            } else {
                maxMemCalculated = mm;
                ctx.byte4[0] = true;
            }
        } else {
            maxMemCalculated = mm;
        }

        // If setMAXMEM, MAXMEM'=op
        if (rom.line[zkPC].setMAXMEM == 1) {
            pol(MAXMEM)[nexti] = fe2n(fr, prime, op0);
            pol(setMAXMEM)[i] = 1;
#ifdef LOG_SETX
            cout << "setMAXMEM MAXMEM[nexti]=" << pol(MAXMEM)[nexti] << endl;
#endif
        } else {
            pol(MAXMEM)[nexti] = maxMemCalculated;
        }

        // If setGAS, GAS'=op
        if (rom.line[zkPC].setGAS == 1) {
            pol(GAS)[nexti] = fe2n(fr, prime, op0);
            pol(setGAS)[i] = 1;
#ifdef LOG_SETX
            cout << "setGAS GAS[nexti]=" << pol(GAS)[nexti] << endl;
#endif
        } else {
            pol(GAS)[nexti] = pol(GAS)[i];
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].mRD == 1) pol(mRD)[i] = 1;

        // If mWR, mem[addr]=op
        if (rom.line[zkPC].mWR == 1) {
            ctx.mem[addr].fe0 = op0;
            ctx.mem[addr].fe1 = op1;
            ctx.mem[addr].fe2 = op2;
            ctx.mem[addr].fe3 = op3;
            ctx.mem[addr].fe4 = op4;
            ctx.mem[addr].fe5 = op5;
            ctx.mem[addr].fe6 = op6;
            ctx.mem[addr].fe7 = op7;
            pol(mWR)[i] = 1;

            MemoryAccess memoryAccess;
            memoryAccess.bIsWrite = true;
            memoryAccess.address = addr;
            memoryAccess.pc = i;
            memoryAccess.fe0 = op0;
            memoryAccess.fe1 = op1;
            memoryAccess.fe2 = op2;
            memoryAccess.fe3 = op3;
            memoryAccess.fe4 = op4;
            memoryAccess.fe5 = op5;
            memoryAccess.fe6 = op6;
            memoryAccess.fe7 = op7;
            memoryAccessList.access.push_back(memoryAccess);

#ifdef LOG_MEMORY
            cout << "Memory write mWR: addr:" << addr << " " << printFea(ctx, ctx.mem[addr]) << endl;
#endif
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].sRD == 1) pol(sRD)[i] = 1;

        if (rom.line[zkPC].sWR == 1)
        {
            if (ctx.lastSWrite.step != i)
            {
                FieldElement keyV0[12];
                keyV0[0] = pol(A0)[i];
                keyV0[1] = pol(A1)[i];
                keyV0[2] = pol(A2)[i];
                keyV0[3] = pol(A3)[i];
                keyV0[4] = pol(A4)[i];
                keyV0[5] = pol(A5)[i];
                keyV0[6] = pol(B0)[i];
                keyV0[7] = pol(B1)[i];
                keyV0[8] = 0;
                keyV0[9] = 0;
                keyV0[10] = 0;
                keyV0[11] = 0;

                FieldElement keyV1[12];
                keyV1[0] = pol(C0)[i];
                keyV1[1] = pol(C1)[i];
                keyV1[2] = pol(C2)[i];
                keyV1[3] = pol(C3)[i];
                keyV1[4] = pol(C4)[i];
                keyV1[5] = pol(C5)[i];
                keyV1[6] = pol(C6)[i];
                keyV1[7] = pol(C7)[i];
                keyV1[8] = 0;
                keyV1[9] = 0;
                keyV1[10] = 0;
                keyV1[11] = 0;
                
#ifdef LOG_TIME
                struct timeval t;
                gettimeofday(&t, NULL);
#endif
                // Call poseidon to get the hash
                poseidon.hash(keyV0);
                poseidon.hash(keyV1);
                keyV0[4] = keyV1[0];
                keyV0[5] = keyV1[1];
                keyV0[6] = keyV1[2];
                keyV0[7] = keyV1[3];
                keyV0[8] = 0;
                keyV0[9] = 0;
                keyV0[10] = 0;
                keyV0[11] = 0;
                poseidon.hash(keyV0);
                ctx.lastSWrite.key[0] = keyV0[0];
                ctx.lastSWrite.key[1] = keyV0[1];
                ctx.lastSWrite.key[2] = keyV0[2];
                ctx.lastSWrite.key[3] = keyV0[3];
#ifdef LOG_TIME
                poseidonTime += TimeDiff(t);
                poseidonTimes++;
#endif
                // Increment counter
                counters.hashPoseidon++;

#ifdef USE_LOCAL_STORAGE
                // Check that storage entry exists
                if (ctx.sto.find(ctx.lastSWrite.key) == ctx.sto.end())
                {
                    cerr << "Error: Storage not initialized key: " << fr.toString(ctx.lastSWrite.key, 16) << " line: " << ctx.zkPC << endl;
                    exit(-1);
                }
#endif

                // Call SMT to get the new Merkel Tree root hash
                SmtSetResult res;
                mpz_class scalarD;
                fea2scalar(fr, scalarD, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i], pol(D4)[i], pol(D5)[i], pol(D6)[i], pol(D7)[i]);
#ifdef LOG_TIME
                gettimeofday(&t, NULL);
#endif
                FieldElement oldRoot[4];
                sr8to4(fr, pol(SR0)[i], pol(SR1)[i], pol(SR2)[i], pol(SR3)[i], pol(SR4)[i], pol(SR5)[i], pol(SR6)[i], pol(SR7)[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

                smt.set(ctx.db, oldRoot, ctx.lastSWrite.key, scalarD, res);

                SmtAction smtAction;
                smtAction.bIsSet = true;
                smtAction.setResult = res;
                smtActionList.action.push_back(smtAction);
#ifdef LOG_TIME
                smtTime += TimeDiff(t);
                smtTimes++;
#endif
                // Store it in lastSWrite
                ctx.lastSWrite.newRoot[0] = res.newRoot[0];
                ctx.lastSWrite.newRoot[1] = res.newRoot[1];
                ctx.lastSWrite.newRoot[2] = res.newRoot[2];
                ctx.lastSWrite.newRoot[3] = res.newRoot[3];
                ctx.lastSWrite.step = i;
            }

            // Check that the new root hash equals op0
            FieldElement oldRoot[4];
            sr8to4(fr, op0, op1, op2, op3, op4, op5, op6, op7, oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

            if ( !fr.eq(ctx.lastSWrite.newRoot, oldRoot) )
            {
                cerr << "Error: Storage write does not match; i: " << i << " zkPC: " << ctx.zkPC << 
                    " ctx.lastSWrite.newRoot: " << fr.toString(ctx.lastSWrite.newRoot[3], 16) << ":" << fr.toString(ctx.lastSWrite.newRoot[2], 16) << ":" << fr.toString(ctx.lastSWrite.newRoot[1], 16) << ":" << fr.toString(ctx.lastSWrite.newRoot[0], 16) <<
                    " oldRoot: " << fr.toString(oldRoot[3], 16) << ":" << fr.toString(oldRoot[2], 16) << ":" << fr.toString(oldRoot[1], 16) << ":" << fr.toString(oldRoot[0], 16) << endl;
                exit(-1);
            }

#ifdef USE_LOCAL_STORAGE
            // Store sto[poseidon_hash]=D
            mpz_class auxScalar;
            fea2scalar(fr, auxScalar, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i]);
            ctx.sto[ctx.lastSWrite.key] = auxScalar;
#endif

            // Copy ROM flags into the polynomials
            pol(sWR)[i] = 1;
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].hashRD == 1) pol(hashRD)[i] = 1;

        if (rom.line[zkPC].hashWR == 1) {

            // Get the size of the hash from D0
            int64_t size = fe2n(fr, prime, pol(D0)[i]);
            if ((size<0) || (size>32)) {
                cerr << "Error: Invalid size for hash.  Size:" << size << " Line:" << ctx.zkPC << endl;
                exit(-1);
            }

            // Get contents of opN into a
            mpz_class a;
            fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7);

            // If there is no entry in the hash database for this address, then create a new one
            if (ctx.hash.find(addr) == ctx.hash.end())
            {
                HashValue hashValue;
                ctx.hash[addr] = hashValue;
            }

            // Fill the hash data vector with chunks of the scalar value
            mpz_class band(0xFF);
            mpz_class result;
            for (int64_t j=0; j<size; j++) {
                result = (a >> (size-j-1)*8) & band;
                uint64_t uiResult = result.get_ui();
                ctx.hash[addr].data.push_back((uint8_t)uiResult);
            }

            // Copy ROM flags into the polynomials
            pol(hashWR)[i] = 1;

#ifdef LOG_HASH
            cout << "Hash write  hashWR: addr:" << addr << " hash:" << ctx.hash[addr].hash << " size:" << ctx.hash[addr].data.size() << " data:";
            for (uint64_t k=0; k<ctx.hash[addr].data.size(); k++) cout << byte2string(ctx.hash[addr].data[k]) << ":";
            cout << endl;
#endif
        }

        // If hashE, calculate hash[addr] using keccak256
        if (rom.line[zkPC].hashE == 1)
        {
#ifdef LOG_TIME
            struct timeval t;
            gettimeofday(&t, NULL);
#endif
            ctx.hash[addr].hash = keccak256(ctx.hash[addr].data.data(), ctx.hash[addr].data.size());
#ifdef LOG_TIME
            keccakTime += TimeDiff(t);
            keccakTimes++;
#endif
            // Increment counter
            counters.hashKeccak++;

            pol(hashE)[i] = 1;
#ifdef LOG_HASH
            cout << "Hash calculate hashE: addr:" << addr << " hash:" << ctx.hash[addr].hash << " size:" << ctx.hash[addr].data.size() << " data:";
            for (uint64_t k=0; k<ctx.hash[addr].data.size(); k++) cout << byte2string(ctx.hash[addr].data[k]) << ":";
            cout << endl;
#endif            
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].ecRecover == 1) pol(ecRecover)[i] = 1;

        // If arith, check that A*B + C = D<<256 + op, using scalars (result can be a big number)
        if (rom.line[zkPC].arith == 1)
        {
            counters.arith++;
            
            // Convert to scalar
            mpz_class A, B, C, D, op;
            fea2scalar(fr, A, pol(A0)[i], pol(A1)[i], pol(A2)[i], pol(A3)[i], pol(A4)[i], pol(A5)[i], pol(A6)[i], pol(A7)[i]);
            fea2scalar(fr, B, pol(B0)[i], pol(B1)[i], pol(B2)[i], pol(B3)[i], pol(B4)[i], pol(B5)[i], pol(B6)[i], pol(B7)[i]);
            fea2scalar(fr, C, pol(C0)[i], pol(C1)[i], pol(C2)[i], pol(C3)[i], pol(C4)[i], pol(C5)[i], pol(C6)[i], pol(C7)[i]);
            fea2scalar(fr, D, pol(D0)[i], pol(D1)[i], pol(D2)[i], pol(D3)[i], pol(D4)[i], pol(D5)[i], pol(D6)[i], pol(D7)[i]);
            fea2scalar(fr, op, op0, op1, op2, op3, op4, op5, op6, op7);

            // Check the condition
            if ( (A*B) + C != (D<<256) + op ) {
                cerr << "Error: Arithmetic does not match: " << ctx.zkPC << endl;
                mpz_class left = (A*B) + C;
                mpz_class right = (D<<256) + op;
                cerr << "(A*B) + C = " << left.get_str(16) << endl;
                cerr << "(D<<256) + op = " << right.get_str(16) << endl;
                exit(-1);
            }

            // Copy ROM flags into the polynomials
            pol(arith)[i] = 1;
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].shl == 1) pol(shl)[i] = 1;
        if (rom.line[zkPC].shr == 1) pol(shr)[i] = 1;
        if (rom.line[zkPC].bin == 1) pol(bin)[i] = 1;
        if (rom.line[zkPC].comparator == 1) pol(comparator)[i] = 1;
        if (rom.line[zkPC].opcodeRomMap == 1) pol(opcodeRomMap)[i] = 1;

        // Evaluate the list cmdAfter commands, and any children command, recursively
        for (uint64_t j=0; j<rom.line[zkPC].cmdAfter.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom.line[zkPC].cmdAfter[j], cr);
        }

#ifdef LOG_STEPS
        cout << "<-- Completed step: " << ii << " zkPC: " << zkPC << " op0: " << fr.toString(op0,16) << " A0: " << fr.toString(pol(A0)[i],16) << " FREE0: " << fr.toString(pol(FREE0)[i],16) << endl;
#endif
    }

    TimerStopAndLog(EXECUTE_LOOP);

    TimerStart(EXECUTE_CLEANUP);

    //printRegs(ctx);
    //printVars(ctx);
    //printMem(ctx);
    //printStorage(ctx);
    //printDb(ctx);

    // Check that all registers are set to 0
    if (!bFastMode) // In fast mode, last nexti was not 0 but 1, and pols have only 2 evaluations
    {
        checkFinalState(ctx);

        // Based on the content of byte4[], fill the byte4_freeIn and byte4_out polynomials
        uint64_t p = 0;
        uint64_t last = 0;

        // Check that we have enough room in polynomials
        if (ctx.byte4.size()*2 > NEVALUATIONS)
        {
            cerr << "Error: Too many byte4 entries" << endl;
            exit(-1);
        }
        
        // Generate polynomials content out of byte4 content
        for (map<uint32_t,bool>::iterator it=ctx.byte4.begin(); it!=ctx.byte4.end(); it++)
        {
            uint32_t num = it->first;
            pol(byte4_freeIN)[p] = num >> 16;
            pol(byte4_out)[p] = last;
            p++;
            pol(byte4_freeIN)[p] = num & 0xFFFF;
            pol(byte4_out)[p] = num >> 16;
            p++;
            last = num;
        }
        pol(byte4_freeIN)[p] = 0;
        pol(byte4_out)[p] = last;
        p++;
        pol(byte4_freeIN)[p] = 0;
        pol(byte4_out)[p] = 0;
        p++;
    }
    TimerStopAndLog(EXECUTE_CLEANUP);

#ifdef LOG_TIME
    cout << "TIMER STATISTICS: Poseidon time: " << double(poseidonTime)/1000 << " ms, called " << poseidonTimes << " times, so " << poseidonTime/zkmax(poseidonTimes,(uint64_t)1) << " us/time" << endl;
    cout << "TIMER STATISTICS: ecRecover time: " << double(ecRecoverTime)/1000 << " ms, called " << ecRecoverTimes << " times, so " << ecRecoverTime/zkmax(ecRecoverTimes,(uint64_t)1) << " us/time" << endl;
    cout << "TIMER STATISTICS: SMT time: " << double(smtTime)/1000 << " ms, called " << smtTimes << " times, so " << smtTime/zkmax(smtTimes,(uint64_t)1) << " us/time" << endl;
    cout << "TIMER STATISTICS: Keccak time: " << double(keccakTime) << " ms, called " << keccakTimes << " times, so " << keccakTime/zkmax(keccakTimes,(uint64_t)1) << " us/time" << endl;
#endif
}

/* Sets first evaluation of all polynomials to zero */
void Executor::initState(Context &ctx)
{
    // Register value initial parameters
    pol(A0)[0] = fr.zero();
    pol(A1)[0] = fr.zero();
    pol(A2)[0] = fr.zero();
    pol(A3)[0] = fr.zero();
    pol(A4)[0] = fr.zero();
    pol(A5)[0] = fr.zero();
    pol(A6)[0] = fr.zero();
    pol(B0)[0] = fr.zero();
    pol(B1)[0] = fr.zero();
    pol(B2)[0] = fr.zero();
    pol(B3)[0] = fr.zero();
    pol(B4)[0] = fr.zero();
    pol(B5)[0] = fr.zero();
    pol(B6)[0] = fr.zero();
    pol(B7)[0] = fr.zero();
    pol(C0)[0] = fr.zero();
    pol(C1)[0] = fr.zero();
    pol(C2)[0] = fr.zero();
    pol(C3)[0] = fr.zero();
    pol(C4)[0] = fr.zero();
    pol(C5)[0] = fr.zero();
    pol(C6)[0] = fr.zero();
    pol(C7)[0] = fr.zero();
    pol(D0)[0] = fr.zero();
    pol(D1)[0] = fr.zero();
    pol(D2)[0] = fr.zero();
    pol(D3)[0] = fr.zero();
    pol(D4)[0] = fr.zero();
    pol(D5)[0] = fr.zero();
    pol(D6)[0] = fr.zero();
    pol(D7)[0] = fr.zero();
    pol(E0)[0] = fr.zero();
    pol(E1)[0] = fr.zero();
    pol(E2)[0] = fr.zero();
    pol(E3)[0] = fr.zero();
    pol(E4)[0] = fr.zero();
    pol(E5)[0] = fr.zero();
    pol(E6)[0] = fr.zero();
    pol(E7)[0] = fr.zero();
    pol(SR0)[0] = fr.zero();
    pol(SR1)[0] = fr.zero();
    pol(SR2)[0] = fr.zero();
    pol(SR3)[0] = fr.zero();
    pol(SR4)[0] = fr.zero();
    pol(SR5)[0] = fr.zero();
    pol(SR6)[0] = fr.zero();
    pol(SR7)[0] = fr.zero();
    pol(CTX)[0] = 0;
    pol(SP)[0] = 0;
    pol(PC)[0] = 0;
    pol(MAXMEM)[0] = 0;
    pol(GAS)[0] = 0;
    pol(zkPC)[0] = 0;
}

// Check that last evaluation (which is in fact the first one) is zero
void Executor::checkFinalState(Context &ctx)
{
    if ( 
        (!fr.isZero(pol(A0)[0])) ||
        (!fr.isZero(pol(A1)[0])) ||
        (!fr.isZero(pol(A2)[0])) ||
        (!fr.isZero(pol(A3)[0])) ||
        (!fr.isZero(pol(A4)[0])) ||
        (!fr.isZero(pol(A5)[0])) ||
        (!fr.isZero(pol(A6)[0])) ||
        (!fr.isZero(pol(A7)[0])) ||
        (!fr.isZero(pol(B0)[0])) ||
        (!fr.isZero(pol(B1)[0])) ||
        (!fr.isZero(pol(B2)[0])) ||
        (!fr.isZero(pol(B3)[0])) ||
        (!fr.isZero(pol(B4)[0])) ||
        (!fr.isZero(pol(B5)[0])) ||
        (!fr.isZero(pol(B6)[0])) ||
        (!fr.isZero(pol(B7)[0])) ||
        (!fr.isZero(pol(C0)[0])) ||
        (!fr.isZero(pol(C1)[0])) ||
        (!fr.isZero(pol(C2)[0])) ||
        (!fr.isZero(pol(C3)[0])) ||
        (!fr.isZero(pol(C4)[0])) ||
        (!fr.isZero(pol(C5)[0])) ||
        (!fr.isZero(pol(C6)[0])) ||
        (!fr.isZero(pol(C7)[0])) ||
        (!fr.isZero(pol(D0)[0])) ||
        (!fr.isZero(pol(D1)[0])) ||
        (!fr.isZero(pol(D2)[0])) ||
        (!fr.isZero(pol(D3)[0])) ||
        (!fr.isZero(pol(D4)[0])) ||
        (!fr.isZero(pol(D5)[0])) ||
        (!fr.isZero(pol(D6)[0])) ||
        (!fr.isZero(pol(D7)[0])) ||
        (!fr.isZero(pol(E0)[0])) ||
        (!fr.isZero(pol(E1)[0])) ||
        (!fr.isZero(pol(E2)[0])) ||
        (!fr.isZero(pol(E3)[0])) ||
        (!fr.isZero(pol(E4)[0])) ||
        (!fr.isZero(pol(E5)[0])) ||
        (!fr.isZero(pol(E6)[0])) ||
        (!fr.isZero(pol(E7)[0])) ||
        (!fr.isZero(pol(SR0)[0])) ||
        (!fr.isZero(pol(SR1)[0])) ||
        (!fr.isZero(pol(SR2)[0])) ||
        (!fr.isZero(pol(SR3)[0])) ||
        (!fr.isZero(pol(SR4)[0])) ||
        (!fr.isZero(pol(SR5)[0])) ||
        (!fr.isZero(pol(SR6)[0])) ||
        (!fr.isZero(pol(SR7)[0])) ||
        (pol(CTX)[0]!=0) ||
        (pol(SP)[0]!=0) ||
        (pol(PC)[0]!=0) ||
        (pol(MAXMEM)[0]!=0) ||
        (pol(GAS)[0]!=0) ||
        (pol(zkPC)[0]!=0)
    ) {
        cerr << "Error: Program terminated with registers not set to zero" << endl;
        exit(-1);
    }
    else{
        //cout << "checkFinalState() succeeded" << endl;
    }
}
