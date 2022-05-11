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
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "poseidon_linear.hpp"

using namespace std;
using json = nlohmann::json;

#define MEM_OFFSET 0x300000000
#define STACK_OFFSET 0x200000000
#define CODE_OFFSET 0x100000000
#define CTX_OFFSET 0x400000000

void Executor::execute (const Input &input, MainCommitPols &pols, Byte4CommitPols &byte4Pols, Database &db, Counters &counters, vector<SmtAction> &smtActionList, MemoryAccessList &memoryAccessList, bool bFastMode)
{
    TimerStart(EXECUTE_INITIALIZATION);
    
#ifdef LOG_TIME
    uint64_t poseidonTime=0, poseidonTimes=0;
    uint64_t smtTime=0, smtTimes=0;
    uint64_t ecRecoverTime=0, ecRecoverTimes=0;
    uint64_t keccakTime=0, keccakTimes=0;
#endif

    RawFec fec;
    RawFnec fnec;

    // Create context and store a finite field reference in it
    Context ctx(fr, pols, byte4Pols, input, db);

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
    uint64_t degree = pols.degree();
    for (uint64_t ii=0; ii<degree; ii++)
    {
        if (bFastMode)
        {
            i = ii%2;
            nexti = (i+1)%2;
            pols.FREE0[i] = fr.zero();
            pols.FREE1[i] = fr.zero();
            pols.FREE2[i] = fr.zero();
            pols.FREE3[i] = fr.zero();
            pols.FREE4[i] = fr.zero();
            pols.FREE5[i] = fr.zero();
            pols.FREE6[i] = fr.zero();
            pols.FREE7[i] = fr.zero();
        }
        else
        {
            i = ii;
            // Calculate nexti to write the next evaluation register values according to setX
            // The registers of the evaluation 0 will be overwritten with the values from the last evaluation, closing the evaluation circle
            nexti = (i+1)%NEVALUATIONS;
        }
        zkPC = pols.zkPC[i]; // This is the read line of ZK code
        ctx.zkPC = zkPC;

        // ctx.step is used inside evaluateCommand() to find the current value of the registers, e.g. pols(A0)[ctx.step]
        ctx.step = i;

        uint64_t incHashPos = 0;

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
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inA, pols.A0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inA, pols.A1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inA, pols.A2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inA, pols.A3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inA, pols.A4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inA, pols.A5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inA, pols.A6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inA, pols.A7[i]));

            pols.inA[i] = rom.line[zkPC].inA;

#ifdef LOG_INX
            cout << "inA op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inB, op = op + inB*B
        if (!fr.isZero(rom.line[zkPC].inB))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inB, pols.B0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inB, pols.B1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inB, pols.B2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inB, pols.B3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inB, pols.B4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inB, pols.B5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inB, pols.B6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inB, pols.B7[i]));

            pols.inB[i] = rom.line[zkPC].inB;
            
#ifdef LOG_INX
            cout << "inB op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inA, op = op + inA*A
        if (!fr.isZero(rom.line[zkPC].inC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inC, pols.C0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inC, pols.C1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inC, pols.C2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inC, pols.C3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inC, pols.C4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inC, pols.C5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inC, pols.C6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inC, pols.C7[i]));

            pols.inC[i] = rom.line[zkPC].inC;
            
#ifdef LOG_INX
            cout << "inC op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inD, op = op + inD*D
        if (!fr.isZero(rom.line[zkPC].inD))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inD, pols.D0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inD, pols.D1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inD, pols.D2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inD, pols.D3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inD, pols.D4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inD, pols.D5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inD, pols.D6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inD, pols.D7[i]));

            pols.inD[i] = rom.line[zkPC].inD;
            
#ifdef LOG_INX
            cout << "inD op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inE, op = op + inE*E
        if (!fr.isZero(rom.line[zkPC].inE))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inE, pols.E0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inE, pols.E1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inE, pols.E2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inE, pols.E3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inE, pols.E4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inE, pols.E5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inE, pols.E6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inE, pols.E7[i]));

            pols.inE[i] = rom.line[zkPC].inE;
            
#ifdef LOG_INX
            cout << "inE op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSR, op = op + inSR*SR
        if (!fr.isZero(rom.line[zkPC].inSR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSR, pols.SR0[i]));
            op1 = fr.add(op1, fr.mul(rom.line[zkPC].inSR, pols.SR1[i]));
            op2 = fr.add(op2, fr.mul(rom.line[zkPC].inSR, pols.SR2[i]));
            op3 = fr.add(op3, fr.mul(rom.line[zkPC].inSR, pols.SR3[i]));
            op4 = fr.add(op4, fr.mul(rom.line[zkPC].inSR, pols.SR4[i]));
            op5 = fr.add(op5, fr.mul(rom.line[zkPC].inSR, pols.SR5[i]));
            op6 = fr.add(op6, fr.mul(rom.line[zkPC].inSR, pols.SR6[i]));
            op7 = fr.add(op7, fr.mul(rom.line[zkPC].inSR, pols.SR7[i]));

            pols.inSR[i] = rom.line[zkPC].inSR;
            
#ifdef LOG_INX
            cout << "inSR op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCTX, op = op + inCTX*CTX
        if (!fr.isZero(rom.line[zkPC].inCTX))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inCTX, pols.CTX[i]));
            pols.inCTX[i] = rom.line[zkPC].inCTX;
#ifdef LOG_INX
            cout << "inCTX op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSP, op = op + inSP*SP
        if (!fr.isZero(rom.line[zkPC].inSP))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSP, pols.SP[i]));
            pols.inSP[i] = rom.line[zkPC].inSP;
#ifdef LOG_INX
            cout << "inSP op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inPC, op = op + inPC*PC
        if (!fr.isZero(rom.line[zkPC].inPC))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inPC, pols.PC[i]));
            pols.inPC[i] = rom.line[zkPC].inPC;
#ifdef LOG_INX
            cout << "inPC op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inGAS, op = op + inGAS*GAS
        if (!fr.isZero(rom.line[zkPC].inGAS))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inGAS, pols.GAS[i]));
            pols.inGAS[i] = rom.line[zkPC].inGAS;
#ifdef LOG_INX
            cout << "inGAS op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inMAXMEM, op = op + inMAXMEM*MAXMEM
        if (!fr.isZero(rom.line[zkPC].inMAXMEM))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inMAXMEM, pols.MAXMEM[i]));
            pols.inMAXMEM[i] = rom.line[zkPC].inMAXMEM;
#ifdef LOG_INX
            cout << "inMAXMEM op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inSTEP, op = op + inSTEP*STEP
        if (!fr.isZero(rom.line[zkPC].inSTEP))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inSTEP, i));
            pols.inSTEP[i] = rom.line[zkPC].inSTEP;
#ifdef LOG_INX
            cout << "inSTEP op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inRR, op = op + inRR*RR
        if (!fr.isZero(rom.line[zkPC].inRR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inRR, pols.RR[i]));
            pols.inRR[i] = rom.line[zkPC].inRR;
#ifdef LOG_INX
            cout << "inRR op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inHASHPOS, op = op + inHASHPOS*HASHPOS
        if (!fr.isZero(rom.line[zkPC].inRR))
        {
            op0 = fr.add(op0, fr.mul(rom.line[zkPC].inHASHPOS, pols.HASHPOS[i]));
            pols.inHASHPOS[i] = rom.line[zkPC].inHASHPOS;
#ifdef LOG_INX
            cout << "inHASHPOS op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        // If inCONST, op = op + CONST
        if (rom.line[zkPC].bConstLPresent)
        {
            scalar2fea(fr, rom.line[zkPC].CONSTL, op0, op1, op2, op3, op4, op5, op6, op7);
            pols.CONST0[i] = op0;
            pols.CONST1[i] = op1;
            pols.CONST2[i] = op2;
            pols.CONST3[i] = op3;
            pols.CONST4[i] = op4;
            pols.CONST5[i] = op5;
            pols.CONST6[i] = op6;
            pols.CONST7[i] = op7;
#ifdef LOG_INX
            cout << "CONSTL op=" << rom.line[zkPC].CONSTL.get_str(16) << endl;
#endif
        }
        else if (rom.line[zkPC].bConstPresent)
        {
            op0 = fr.add(op0, rom.line[zkPC].CONST);
            pols.CONST0[i] = rom.line[zkPC].CONST;
#ifdef LOG_INX
            cout << "CONST op=" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0, 16) << endl;
#endif
        }

        uint32_t addrRel = 0;
        uint64_t addr = 0;

        // If address is involved, load offset into addr
        if (rom.line[zkPC].mRD==1 || rom.line[zkPC].mWR==1 || rom.line[zkPC].hashK==1 || rom.line[zkPC].hashKLen==1 || rom.line[zkPC].hashKDigest==1 || rom.line[zkPC].hashP==1 || rom.line[zkPC].hashPLen==1 || rom.line[zkPC].hashPDigest==1 || rom.line[zkPC].JMP==1 || rom.line[zkPC].JMPC==1) {
            if (rom.line[zkPC].ind == 1)
            {
                if (rom.line[zkPC].JMP==1 || rom.line[zkPC].JMPC==1) addrRel = fe2n(fr, pols.RR[i]);
                else addrRel = fe2n(fr, pols.E0[i]);
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
            addr += pols.CTX[i]*CTX_OFFSET;
            pols.useCTX[i] = 1;
#ifdef LOG_ADDR
            cout << "useCTX addr=" << addr << endl;
#endif
        }

        // If isCode, addr = addr + CODE_OFFSET
        if (rom.line[zkPC].isCode == 1) {
            addr += CODE_OFFSET;
            pols.isCode[i] = 1;
#ifdef LOG_ADDR
            cout << "isCode addr=" << addr << endl;
#endif
        }

        // If isStack, addr = addr + STACK_OFFSET
        if (rom.line[zkPC].isStack == 1) {
            addr += STACK_OFFSET;
            pols.isStack[i] = 1;
#ifdef LOG_ADDR
            cout << "isStack addr=" << addr << endl;
#endif
        }

        // If isMem, addr = addr + MEM_OFFSET
        if (rom.line[zkPC].isMem == 1) {
            addr += MEM_OFFSET;
            pols.isMem[i] = 1;
#ifdef LOG_ADDR
            cout << "isMem addr=" << addr << endl;
#endif
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].incCode != 0) pols.incCode[i] = rom.line[zkPC].incCode;
        if (rom.line[zkPC].incStack != 0) pols.incStack[i] = rom.line[zkPC].incStack;
        if (rom.line[zkPC].ind == 1) pols.ind[i] = 1;

        // If offset, record it in byte4
        if (rom.line[zkPC].bOffsetPresent && (rom.line[zkPC].offset!=0)) {
            pols.offset[i] = rom.line[zkPC].offset;
        }

        // If inFREE, calculate the free value, and add it to op
        if (!fr.isZero(rom.line[zkPC].inFREE))
        {
            // freeInTag must be present
            if (rom.line[zkPC].freeInTag.isPresent == false) {
                cerr << "Error: Instruction with freeIn without freeInTag: zkPC=" << ctx.zkPC << endl;
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
                    keyV0[0] = pols.A0[i];
                    keyV0[1] = pols.A1[i];
                    keyV0[2] = pols.A2[i];
                    keyV0[3] = pols.A3[i];
                    keyV0[4] = pols.A4[i];
                    keyV0[5] = pols.A5[i];
                    keyV0[6] = pols.B0[i];
                    keyV0[7] = pols.B1[i];
                    keyV0[8] = 0;
                    keyV0[9] = 0;
                    keyV0[10] = 0;
                    keyV0[11] = 0;

                    FieldElement keyV1[12];
                    keyV1[0] = pols.C0[i];
                    keyV1[1] = pols.C1[i];
                    keyV1[2] = pols.C2[i];
                    keyV1[3] = pols.C3[i];
                    keyV1[4] = pols.C4[i];
                    keyV1[5] = pols.C5[i];
                    keyV1[6] = pols.C6[i];
                    keyV1[7] = pols.C7[i];
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
                    //smt.get(ctx.fr, ctx.db, pols.SR[i], ctx.lastSWrite.key, smtGetResult);
                    //cout << "STORAGE2 i:" << i << " hash:" << fr.toString(ctx.lastSWrite.key, 16) << " value:" << smtGetResult.value.get_str(16) << endl;

                    // Read the value from storage, and store it in fin
                    scalar2fea(fr, ctx.sto[ctx.lastSWrite.key], fi0, fi1, fi2, fi3);
#else
                    FieldElement oldRoot[4];
                    sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);
                    
                    SmtGetResult smtGetResult;
                    smt.get(ctx.db, oldRoot, ctx.lastSWrite.key, smtGetResult);
                    //cout << "smt.get() returns value=" << smtGetResult.value.get_str(16) << endl;

                    SmtAction smtAction;
                    smtAction.bIsSet = false;
                    smtAction.getResult = smtGetResult;
                    smtActionList.push_back(smtAction);
                    
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
                    keyV0[0] = pols.A0[i];
                    keyV0[1] = pols.A1[i];
                    keyV0[2] = pols.A2[i];
                    keyV0[3] = pols.A3[i];
                    keyV0[4] = pols.A4[i];
                    keyV0[5] = pols.A5[i];
                    keyV0[6] = pols.B0[i];
                    keyV0[7] = pols.B1[i];
                    keyV0[8] = 0;
                    keyV0[9] = 0;
                    keyV0[10] = 0;
                    keyV0[11] = 0;

                    FieldElement keyV1[12];
                    keyV1[0] = pols.C0[i];
                    keyV1[1] = pols.C1[i];
                    keyV1[2] = pols.C2[i];
                    keyV1[3] = pols.C3[i];
                    keyV1[4] = pols.C4[i];
                    keyV1[5] = pols.C5[i];
                    keyV1[6] = pols.C6[i];
                    keyV1[7] = pols.C7[i];
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
                    fea2scalar(fr, scalarD, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
#ifdef LOG_TIME
                    gettimeofday(&t, NULL);
#endif
                    FieldElement oldRoot[4];
                    sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);
                    
                    smt.set(ctx.db, oldRoot, ctx.lastSWrite.key, scalarD, smtSetResult);

                    SmtAction smtAction;
                    smtAction.bIsSet = true;
                    smtAction.setResult = smtSetResult;
                    smtActionList.push_back(smtAction);
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
#if 0
                // If hashRD (hash read)
                if (rom.line[zkPC].hashK == 1) // TODO: Review, it was hashRD
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
#endif
                if (rom.line[zkPC].hashK == 1)
                {
                    // If there is no entry in the hash database for this address, then create a new one
                    if (ctx.hashK.find(addr) == ctx.hashK.end())
                    {
                        HashValue hashValue;
                        ctx.hashK[addr] = hashValue;
                    }
                    
                    // Get the size of the hash from D0
                    int64_t iSize = fe2n(fr, pols.D0[i]);
                    if ((iSize<0) || (iSize>32)) {
                        cerr << "Error: Invalid size for hashK:  Size:" << iSize << " Line:" << ctx.zkPC << endl;
                        exit(-1);
                    }
                    uint64_t size = iSize;

                    // Get the positon of the hash from HASHPOS
                    int64_t iPos = fe2n(fr, pols.HASHPOS[i]);
                    if (iPos < 0)
                    {
                        cerr << "Error: invalid pos for HashK: pos:" << iPos << " Line:" << ctx.zkPC << endl;
                        exit(-1);
                    }
                    uint64_t pos = iPos;

                    // Check that pos+size do not exceed data size
                    if ( (pos+size) > ctx.hashK[addr].data.size())
                    {
                        cerr << "Error: hashK invalid size of hash: pos=" << pos << " size=" << size << " data.size=" << ctx.hashK[addr].data.size() << endl;
                        exit(-1);
                    }

                    // Copy data into fi
                    mpz_class s;
                    for (uint64_t j=0; j<size; j++)
                    {
                        s = s<<8 + ctx.hashK[addr].data[pos+j];
                    }
                    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

                if (rom.line[zkPC].hashKDigest == 1)
                {
                    // If there is no entry in the hash database for this address, this is an error
                    if (ctx.hashK.find(addr) == ctx.hashK.end())
                    {
                        cerr << "Error: hashKDigest: digest not defined" << endl;
                        exit(-1);
                    }

                    // If digest was not calculated, this is an error
                    if (ctx.hashK[addr].digest.size() == 0)
                    {
                        cerr << "Error: hashKDigest: digest not calculated.  Call hashKLen to finish digest." << endl;
                        exit(-1);
                    }

                    // Copy digest into fi
                    mpz_class dg;
                    dg.set_str(ctx.hashK[addr].digest, 16);
                    scalar2fea(fr, dg, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

                if (rom.line[zkPC].hashP == 1)
                {
                    // If there is no entry in the hash database for this address, then create a new one
                    if (ctx.hashP.find(addr) == ctx.hashP.end())
                    {
                        HashValue hashValue;
                        ctx.hashP[addr] = hashValue;
                    }
                    
                    // Get the size of the hash from D0
                    int64_t iSize = fe2n(fr, pols.D0[i]);
                    if ((iSize<0) || (iSize>32)) {
                        cerr << "Error: Invalid size for hashP:  Size:" << iSize << " Line:" << ctx.zkPC << endl;
                        exit(-1);
                    }
                    uint64_t size = iSize;

                    // Get the positon of the hash from HASHPOS
                    int64_t iPos = fe2n(fr, pols.HASHPOS[i]);
                    if (iPos < 0)
                    {
                        cerr << "Error: invalid pos for HashP: pos:" << iPos << " Line:" << ctx.zkPC << endl;
                        exit(-1);
                    }
                    uint64_t pos = iPos;

                    // Check that pos+size do not exceed data size
                    if ( (pos+size) > ctx.hashP[addr].data.size())
                    {
                        cerr << "Error: hashP invalid size of hash: pos=" << pos << " size=" << size << " data.size=" << ctx.hashK[addr].data.size() << endl;
                        exit(-1);
                    }

                    // Copy data into fi
                    mpz_class s;
                    for (uint64_t j=0; j<size; j++)
                    {
                        s = s<<8 + ctx.hashP[addr].data[pos+j];
                    }
                    scalar2fea(fr, s, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

                if (rom.line[zkPC].hashPDigest == 1)
                {
                    // If there is no entry in the hash database for this address, this is an error
                    if (ctx.hashP.find(addr) == ctx.hashP.end())
                    {
                        cerr << "Error: hashPDigest: digest not defined" << endl;
                        exit(-1);
                    }

                    // If digest was not calculated, this is an error
                    if (ctx.hashP[addr].digest.size() == 0)
                    {
                        cerr << "Error: hashPDigest: digest not calculated.  Call hashKLen to finish digest." << endl;
                        exit(-1);
                    }

                    // Copy digest into fi
                    mpz_class dg;
                    dg.set_str(ctx.hashP[addr].digest, 16);
                    scalar2fea(fr, dg, fi0, fi1, fi2, fi3, fi4 ,fi5 ,fi6 ,fi7);

                    nHits++;
                }

#if 0
                // If ecRecover, build the transaction signature, recover the address that generated it, and copy fi=recovered address
                if (rom.line[zkPC].ecRecover == 1) {

                    // Increment counter
                    counters.ecRecover++;
                    
                    mpz_class aux;
                    
                    // Get d=A
                    fea2scalar(fr, aux, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                    string d = NormalizeTo0xNFormat(aux.get_str(16),64);

                    // Signature string = 0x + r(32B) + s(32B) + v(1B) = 0x + 130chars
                    fea2scalar(fr, aux, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                    string r = NormalizeToNFormat(aux.get_str(16),64);
                    fea2scalar(fr, aux, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
                    string s = NormalizeToNFormat(aux.get_str(16),64);
                    aux = fe2n(fr, pols.D0[i]);
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
#endif
                // TODO: Define nBits properly const { nBits } = require("../starkstruct.js");
                uint64_t nBits = 0;

                if (rom.line[zkPC].bin == 1)
                {
                    if (rom.line[zkPC].binOpcode == 0) // NOP
                    {
                        fi0 = fr.zero();
                        fi1 = fr.zero();
                        fi2 = fr.zero();
                        fi3 = fr.zero();
                        fi4 = fr.zero();
                        fi5 = fr.zero();
                        fi6 = fr.zero();
                        fi7 = fr.zero();
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 1) // ADD
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a + b) & Mask256;
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 2) // SUB
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a - b + twoTo256) & Mask256;
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 3) // LT
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a < b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 4) // GT
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a > b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 5) // SLT
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        if (a >= twoTo255) a = a - twoTo256;
                        if (b >= twoTo255) b = b - twoTo256;
                        c = (a < b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 6) // SGT
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        if (a >= twoTo255) a = a - twoTo256;
                        if (b >= twoTo255) b = b - twoTo256;
                        c = (a > b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 7) // EQ
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a == b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 8) // ISZERO
                    {
                        mpz_class a, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        c = (a == 0);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 9) // AND
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a & b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 10) // OR
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a | b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 11) // XOR
                    {
                        mpz_class a, b, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                        c = (a ^ b);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else if (rom.line[zkPC].binOpcode == 12) // NOT
                    {
                        mpz_class a, c;
                        fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                        c = (a ^ Mask256);
                        scalar2fea(fr, c, fi0, fi1, fi2, fi3, fi4, fi5, fi6, fi7);
                        nBits++;
                    }
                    else
                    {
                        cerr << "Error: Invalid binary operation: opcode=" << rom.line[zkPC].binOpcode << endl;
                        exit(-1);
                    }
                }

                // If shl, shift A, D bytes to the left, and discard highest bits
                if (rom.line[zkPC].shl == 1)
                {
                    // Read a=A
                    mpz_class a;
                    fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);

                    // Read s=D
                    uint64_t s = fe2n(fr, pols.D0[i]);
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
                    fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);

                    // Read s=D
                    uint64_t s = fe2n(fr, pols.D0[i]);
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
            pols.FREE0[i] = fi0;
            pols.FREE1[i] = fi1;
            pols.FREE2[i] = fi2;
            pols.FREE3[i] = fi3;
            pols.FREE4[i] = fi4;
            pols.FREE5[i] = fi5;
            pols.FREE6[i] = fi6;
            pols.FREE7[i] = fi7;

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
            pols.inFREE[i] = rom.line[zkPC].inFREE;
        }

        // If assert, check that A=op
        if (rom.line[zkPC].assert == 1)
        {
            if ( (!fr.eq(pols.A0[i], op0)) ||
                 (!fr.eq(pols.A1[i], op1)) ||
                 (!fr.eq(pols.A2[i], op2)) ||
                 (!fr.eq(pols.A3[i], op3)) ||
                 (!fr.eq(pols.A4[i], op4)) ||
                 (!fr.eq(pols.A5[i], op5)) ||
                 (!fr.eq(pols.A6[i], op6)) ||
                 (!fr.eq(pols.A7[i], op7)) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << ctx.zkPC << endl;
                cout << "A: " << fr.toString(pols.A7[i], 16) << ":" << fr.toString(pols.A6[i], 16) << ":" << fr.toString(pols.A5[i], 16) << ":" << fr.toString(pols.A4[i], 16) << ":" << fr.toString(pols.A3[i], 16) << ":" << fr.toString(pols.A2[i], 16) << ":" << fr.toString(pols.A1[i], 16) << ":" << fr.toString(pols.A0[i], 16) << endl;
                cout << "OP:" << fr.toString(op7, 16) << ":" << fr.toString(op6, 16) << ":" << fr.toString(op5, 16) << ":" << fr.toString(op4,16) << ":" << fr.toString(op3, 16) << ":" << fr.toString(op2, 16) << ":" << fr.toString(op1, 16) << ":" << fr.toString(op0,16) << endl;
                exit(-1);
            }
            pols.assert[i] = 1;
#ifdef LOG_ASSERT
            cout << "assert" << endl;
#endif
        }

        // If setA, A'=op
        if (rom.line[zkPC].setA == 1) {
            pols.A0[nexti] = op0;
            pols.A1[nexti] = op1;
            pols.A2[nexti] = op2;
            pols.A3[nexti] = op3;
            pols.A4[nexti] = op4;
            pols.A5[nexti] = op5;
            pols.A6[nexti] = op6;
            pols.A7[nexti] = op7;
            pols.setA[i] = 1;
#ifdef LOG_SETX
            cout << "setA A[nexti]=" << pols.A3[nexti] << ":" << pols.A2[nexti] << ":" << pols.A1[nexti] << ":" << fr.toString(pols.A0[nexti], 16) << endl;
#endif
        } else {
            pols.A0[nexti] = pols.A0[i];
            pols.A1[nexti] = pols.A1[i];
            pols.A2[nexti] = pols.A2[i];
            pols.A3[nexti] = pols.A3[i];
            pols.A4[nexti] = pols.A4[i];
            pols.A5[nexti] = pols.A5[i];
            pols.A6[nexti] = pols.A6[i];
            pols.A7[nexti] = pols.A7[i];
        }

        // If setB, B'=op
        if (rom.line[zkPC].setB == 1) {
            pols.B0[nexti] = op0;
            pols.B1[nexti] = op1;
            pols.B2[nexti] = op2;
            pols.B3[nexti] = op3;
            pols.B4[nexti] = op4;
            pols.B5[nexti] = op5;
            pols.B6[nexti] = op6;
            pols.B7[nexti] = op7;
            pols.setB[i] = 1;
#ifdef LOG_SETX
            cout << "setB B[nexti]=" << pols.B3[nexti] << ":" << pols.B2[nexti] << ":" << pols.B1[nexti] << ":" << fr.toString(pols.B0[nexti], 16) << endl;
#endif
        } else {
            pols.B0[nexti] = pols.B0[i];
            pols.B1[nexti] = pols.B1[i];
            pols.B2[nexti] = pols.B2[i];
            pols.B3[nexti] = pols.B3[i];
            pols.B4[nexti] = pols.B4[i];
            pols.B5[nexti] = pols.B5[i];
            pols.B6[nexti] = pols.B6[i];
            pols.B7[nexti] = pols.B7[i];
        }

        // If setC, C'=op
        if (rom.line[zkPC].setC == 1) {
            pols.C0[nexti] = op0;
            pols.C1[nexti] = op1;
            pols.C2[nexti] = op2;
            pols.C3[nexti] = op3;
            pols.C4[nexti] = op4;
            pols.C5[nexti] = op5;
            pols.C6[nexti] = op6;
            pols.C7[nexti] = op7;
            pols.setC[i] = 1;
#ifdef LOG_SETX
            cout << "setC C[nexti]=" << pols.C3[nexti] << ":" << pols.C2[nexti] << ":" << pols.C1[nexti] << ":" << fr.toString(pols.C0[nexti], 16) << endl;
#endif
        } else {
            pols.C0[nexti] = pols.C0[i];
            pols.C1[nexti] = pols.C1[i];
            pols.C2[nexti] = pols.C2[i];
            pols.C3[nexti] = pols.C3[i];
            pols.C4[nexti] = pols.C4[i];
            pols.C5[nexti] = pols.C5[i];
            pols.C6[nexti] = pols.C6[i];
            pols.C7[nexti] = pols.C7[i];
        }

        // If setD, D'=op
        if (rom.line[zkPC].setD == 1) {
            pols.D0[nexti] = op0;
            pols.D1[nexti] = op1;
            pols.D2[nexti] = op2;
            pols.D3[nexti] = op3;
            pols.D4[nexti] = op4;
            pols.D5[nexti] = op5;
            pols.D6[nexti] = op6;
            pols.D7[nexti] = op7;
            pols.setD[i] = 1;
#ifdef LOG_SETX
            cout << "setD D[nexti]=" << pols.D3[nexti] << ":" << pols.D2[nexti] << ":" << pols.D1[nexti] << ":" << fr.toString(pols.D0[nexti], 16) << endl;
#endif
        } else {
            pols.D0[nexti] = pols.D0[i];
            pols.D1[nexti] = pols.D1[i];
            pols.D2[nexti] = pols.D2[i];
            pols.D3[nexti] = pols.D3[i];
            pols.D4[nexti] = pols.D4[i];
            pols.D5[nexti] = pols.D5[i];
            pols.D6[nexti] = pols.D6[i];
            pols.D7[nexti] = pols.D7[i];
        }
        
        // If setE, E'=op
        if (rom.line[zkPC].setE == 1) {
            pols.E0[nexti] = op0;
            pols.E1[nexti] = op1;
            pols.E2[nexti] = op2;
            pols.E3[nexti] = op3;
            pols.E4[nexti] = op4;
            pols.E5[nexti] = op5;
            pols.E6[nexti] = op6;
            pols.E7[nexti] = op7;
            pols.setE[i] = 1;
#ifdef LOG_SETX
            cout << "setE E[nexti]=" << pols.E3[nexti] << ":" << pols.E2[nexti] << ":" << pols.E1[nexti] << ":" << fr.toString(pols.E0[nexti] ,16) << endl;
#endif
        } else {
            pols.E0[nexti] = pols.E0[i];
            pols.E1[nexti] = pols.E1[i];
            pols.E2[nexti] = pols.E2[i];
            pols.E3[nexti] = pols.E3[i];
            pols.E4[nexti] = pols.E4[i];
            pols.E5[nexti] = pols.E5[i];
            pols.E6[nexti] = pols.E6[i];
            pols.E7[nexti] = pols.E7[i];
        }

        // If setSR, SR'=op
        if (rom.line[zkPC].setSR == 1) {
            pols.SR0[nexti] = op0;
            pols.SR1[nexti] = op1;
            pols.SR2[nexti] = op2;
            pols.SR3[nexti] = op3;
            pols.SR4[nexti] = op4;
            pols.SR5[nexti] = op5;
            pols.SR6[nexti] = op6;
            pols.SR7[nexti] = op7;
            pols.setSR[i] = 1;
#ifdef LOG_SETX
            cout << "setSR SR[nexti]=" << fr.toString(pols.SR[nexti], 16) << endl;
#endif
        } else {
            pols.SR0[nexti] = pols.SR0[i];
            pols.SR1[nexti] = pols.SR1[i];
            pols.SR2[nexti] = pols.SR2[i];
            pols.SR3[nexti] = pols.SR3[i];
            pols.SR4[nexti] = pols.SR4[i];
            pols.SR5[nexti] = pols.SR5[i];
            pols.SR6[nexti] = pols.SR6[i];
            pols.SR7[nexti] = pols.SR7[i];
        }

        // If setCTX, CTX'=op
        if (rom.line[zkPC].setCTX == 1) {
            pols.CTX[nexti] = fe2n(fr, op0);
            pols.setCTX[i] = 1;
#ifdef LOG_SETX
            cout << "setCTX CTX[nexti]=" << pols.CTX[nexti] << endl;
#endif
        } else {
            pols.CTX[nexti] = pols.CTX[i];
        }

        // If setSP, SP'=op
        if (rom.line[zkPC].setSP == 1) {
            pols.SP[nexti] = fe2n(fr, op0);
            pols.setSP[i] = 1;
#ifdef LOG_SETX
            cout << "setSP SP[nexti]=" << pols.SP[nexti] << endl;
#endif
        } else {
            // SP' = SP + incStack
            if (rom.line[zkPC].incStack<0 || rom.line[zkPC].incStack>0xFFFF)
            {
                cerr << "Error: incStack cannot be added to an u16 polynomial: " << rom.line[zkPC].incStack << endl;
                exit(-1);
            }
            pols.SP[nexti] = pols.SP[i] + rom.line[zkPC].incStack;
        }

        // If setPC, PC'=op
        if (rom.line[zkPC].setPC == 1) {
            pols.PC[nexti] = fe2n(fr, op0);
            pols.setPC[i] = 1;
#ifdef LOG_SETX
            cout << "setPC PC[nexti]=" << pols.PC[nexti] << endl;
#endif
        } else {
            // PC' = PC + incCode
            if (rom.line[zkPC].incCode<0 || rom.line[zkPC].incCode>0xFFFF)
            {
                cerr << "Error: incCode cannot be added to an u16 polynomial: " << rom.line[zkPC].incCode << endl;
                exit(-1);
            }
            pols.PC[nexti] = pols.PC[i] + rom.line[zkPC].incCode;
        }

        // If setRR, RR'=op0
        if (rom.line[zkPC].setRR == 1) {
            pols.RR[nexti] = fe2n(fr, op0);
            pols.setRR[i] = 1;
        } else {
            pols.RR[nexti] = pols.RR[i];
        }

        // If JMPC, jump conditionally based on op value
        if (rom.line[zkPC].JMPC == 1) {
#ifdef LOG_JMP
            cout << "JMPC: op0=" << fr.toString(op0) << endl;
#endif
            int64_t o = fe2n(fr, op0);
#ifdef LOG_JMP
            cout << "JMPC: o=" << o << endl;
#endif
            // If op<0, jump to addr: zkPC'=addr
            if (o < 0) {
                pols.isNeg[i] = 1;
                pols.zkPC[nexti] = addr;
                ctx.byte4[0x100000000 + o] = true;
#ifdef LOG_JMP
               cout << "Next zkPC(1)=" << pols.zkPC[nexti] << endl;
#endif
            }
            // If op>=0, simply increase zkPC'=zkPC+1
            else
            {
                pols.zkPC[nexti] = pols.zkPC[i] + 1;
#ifdef LOG_JMP
                cout << "Next zkPC(2)=" << pols.zkPC[nexti] << endl;
#endif
                ctx.byte4[o] = true;
            }
            pols.JMPC[i] = 1;
        }
        // If JMP, directly jump zkPC'=addr
        else if (rom.line[zkPC].JMP == 1)
        {
            pols.zkPC[nexti] = addr;
#ifdef LOG_JMP
            cout << "Next zkPC(3)=" << pols.zkPC[nexti] << endl;
#endif
            pols.JMP[i] = 1;
        }
        // Else, simply increase zkPC'=zkPC+1
        else
        {
            pols.zkPC[nexti] = pols.zkPC[i] + 1;
        }

        // Calculate the new max mem address, if any
        uint32_t maxMemCalculated = 0;
        uint32_t mm = pols.MAXMEM[i];
        if (rom.line[zkPC].isMem==1)
        {
            if (addrRel>mm) {
                pols.isMaxMem[i] = 1;
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
            pols.MAXMEM[nexti] = fe2n(fr, op0);
            pols.setMAXMEM[i] = 1;
#ifdef LOG_SETX
            cout << "setMAXMEM MAXMEM[nexti]=" << pols.MAXMEM[nexti] << endl;
#endif
        } else {
            pols.MAXMEM[nexti] = maxMemCalculated;
        }

        // If setGAS, GAS'=op
        if (rom.line[zkPC].setGAS == 1) {
            pols.GAS[nexti] = fe2n(fr, op0);
            pols.setGAS[i] = 1;
#ifdef LOG_SETX
            cout << "setGAS GAS[nexti]=" << pols.GAS[nexti] << endl;
#endif
        } else {
            pols.GAS[nexti] = pols.GAS[i];
        }

        // Copy ROM flags into the polynomials
        if (rom.line[zkPC].mRD == 1)
        {
            pols.mOp[i] = 1;
            //pols.mWR[i] = 0; No need to set to 0, since this is the default value
        }

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
            pols.mOp[i] = 1;
            pols.mWR[i] = 1;

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
        if (rom.line[zkPC].sRD == 1) pols.sRD[i] = 1;

        if (rom.line[zkPC].sWR == 1)
        {
            if (ctx.lastSWrite.step != i)
            {
                FieldElement keyV0[12];
                keyV0[0] = pols.A0[i];
                keyV0[1] = pols.A1[i];
                keyV0[2] = pols.A2[i];
                keyV0[3] = pols.A3[i];
                keyV0[4] = pols.A4[i];
                keyV0[5] = pols.A5[i];
                keyV0[6] = pols.B0[i];
                keyV0[7] = pols.B1[i];
                keyV0[8] = 0;
                keyV0[9] = 0;
                keyV0[10] = 0;
                keyV0[11] = 0;

                FieldElement keyV1[12];
                keyV1[0] = pols.C0[i];
                keyV1[1] = pols.C1[i];
                keyV1[2] = pols.C2[i];
                keyV1[3] = pols.C3[i];
                keyV1[4] = pols.C4[i];
                keyV1[5] = pols.C5[i];
                keyV1[6] = pols.C6[i];
                keyV1[7] = pols.C7[i];
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
                fea2scalar(fr, scalarD, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
#ifdef LOG_TIME
                gettimeofday(&t, NULL);
#endif
                FieldElement oldRoot[4];
                sr8to4(fr, pols.SR0[i], pols.SR1[i], pols.SR2[i], pols.SR3[i], pols.SR4[i], pols.SR5[i], pols.SR6[i], pols.SR7[i], oldRoot[0], oldRoot[1], oldRoot[2], oldRoot[3]);

                smt.set(ctx.db, oldRoot, ctx.lastSWrite.key, scalarD, res);

                SmtAction smtAction;
                smtAction.bIsSet = true;
                smtAction.setResult = res;
                smtActionList.push_back(smtAction);
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
            fea2scalar(fr, auxScalar, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i]);
            ctx.sto[ctx.lastSWrite.key] = auxScalar;
#endif

            // Copy ROM flags into the polynomials
            pols.sWR[i] = 1;
        }

        if (rom.line[zkPC].hashK == 1)
        {
            pols.hashK[i] = 1;

            // If there is no entry in the hash database for this address, then create a new one
            if (ctx.hashK.find(addr) == ctx.hashK.end())
            {
                HashValue hashValue;
                ctx.hashK[addr] = hashValue;
            }
            
            // Get the size of the hash from D0
            int64_t iSize = fe2n(fr, pols.D0[i]);
            if ((iSize<0) || (iSize>32)) {
                cerr << "Error: Invalid size for hashK:  Size:" << iSize << " Line:" << ctx.zkPC << endl;
                exit(-1);
            }
            uint64_t size = iSize;

            // Get the positon of the hash from HASHPOS
            int64_t iPos = fe2n(fr, pols.HASHPOS[i]);
            if (iPos < 0)
            {
                cerr << "Error: invalid pos for HashK: pos:" << iPos << " Line:" << ctx.zkPC << endl;
                exit(-1);
            }
            uint64_t pos = iPos;

            // Get contents of opN into a
            mpz_class a;
            fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7);

            // Fill the hash data vector with chunks of the scalar value
            mpz_class result;
            for (uint64_t j=0; j<size; j++) {
                result = (a >> (size-j-1)*8) & Mask8;
                uint8_t bm = result.get_ui();
                if (ctx.hashK[addr].data.size() == (pos+j))
                {
                    ctx.hashK[addr].data.push_back(bm);
                }
                else if (ctx.hashK[addr].data.size() < (pos+j))
                {
                    cerr << "Error: hashK: trying to insert data in a position:" << (pos+j) << " higher than current data size:" << ctx.hashK[addr].data.size() << endl;
                    exit(-1);
                }
                else
                {
                    uint8_t bh;
                    bh = ctx.hashK[addr].data[pos+j];
                    if (bm != bh)
                    {
                        cerr << "Error: HashK bytes do not match: addr=" << addr << " pos+j=" << pos+j << " is bm=" << bm << " and it should be bh=" << bh << endl;
                        exit(-1);
                    }
                }
            }

            // Record the read operation
            HashRead hashRead;
            hashRead.pos = pos;
            hashRead.len = size;
            ctx.hashK[addr].reads.push_back(hashRead);

            // Store the size
            incHashPos = size;
        }

        if (rom.line[zkPC].hashKLen == 1)
        {
            pols.hashKLen[i] = 1;

            uint64_t lm = fe2n(fr, op0);
            uint64_t lh = ctx.hashK[addr].data.size();
            if (lm != lh)
            {
                cerr << "Error: HashK length does not match match addr=" << addr << " is lm=" << lm << " and it should be lh=" << lh << endl;
                exit(-1);
            }
            if (ctx.hashK[addr].digest.size() == 0)
            {
#ifdef LOG_TIME
                struct timeval t;
                gettimeofday(&t, NULL);
#endif
                ctx.hashK[addr].digest = keccak256(ctx.hashK[addr].data.data(), ctx.hashK[addr].data.size());
#ifdef LOG_TIME
                keccakTime += TimeDiff(t);
                keccakTimes++;
#endif
                // Increment counter
                counters.hashKeccak++;

#ifdef LOG_HASH
                cout << "Hash calculate hashKLen: addr:" << addr << " hash:" << ctx.hashK[addr].digest << " size:" << ctx.hashK[addr].data.size() << " data:";
                for (uint64_t k=0; k<ctx.hashK[addr].data.size(); k++) cout << byte2string(ctx.hashK[addr].data[k]) << ":";
                cout << endl;
#endif   
            }
        }

        if (rom.line[zkPC].hashKDigest == 1)
        {
            pols.hashKDigest[i] = 1;

            // Get contents of op into dg
            mpz_class dg;
            fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7);

            // Check the digest has been calculated
            if (ctx.hashK[addr].digest.size() == 0)
            {
                cerr << "Error: hashKDigest: Cannot load keccak from DB" << endl;
                exit(-1);
            }

            // Check that digest equals op
            mpz_class digestScalar;
            digestScalar.set_str(ctx.hashK[addr].digest, 10);
            if (dg != digestScalar)
            {
                cerr << "Error: hashKDigest: Digest does not match op" << endl;
                exit(-1);
            }
        }
            
        if (rom.line[zkPC].hashP == 1)
        {
            pols.hashP[i] = 1;

            // If there is no entry in the hash database for this address, then create a new one
            if (ctx.hashP.find(addr) == ctx.hashP.end())
            {
                HashValue hashValue;
                ctx.hashP[addr] = hashValue;
            }
            
            // Get the size of the hash from D0
            int64_t iSize = fe2n(fr, pols.D0[i]);
            if ((iSize<0) || (iSize>32)) {
                cerr << "Error: Invalid size for hashP:  Size:" << iSize << " Line:" << ctx.zkPC << endl;
                exit(-1);
            }
            uint64_t size = iSize;

            // Get the positon of the hash from HASHPOS
            int64_t iPos = fe2n(fr, pols.HASHPOS[i]);
            if (iPos < 0)
            {
                cerr << "Error: invalid pos for HashP: pos:" << iPos << " Line:" << ctx.zkPC << endl;
                exit(-1);
            }
            uint64_t pos = iPos;

            // Get contents of opN into a
            mpz_class a;
            fea2scalar(fr, a, op0, op1, op2, op3, op4, op5, op6, op7);

            // Fill the hash data vector with chunks of the scalar value
            mpz_class result;
            for (uint64_t j=0; j<size; j++) {
                result = (a >> (size-j-1)*8) & Mask8;
                uint8_t bm = result.get_ui();
                if (ctx.hashP[addr].data.size() == (pos+j))
                {
                    ctx.hashP[addr].data.push_back(bm);
                }
                else if (ctx.hashP[addr].data.size() < (pos+j))
                {
                    cerr << "Error: hashP: trying to insert data in a position:" << (pos+j) << " higher than current data size:" << ctx.hashP[addr].data.size() << endl;
                    exit(-1);
                }
                else
                {
                    uint8_t bh;
                    bh = ctx.hashP[addr].data[pos+j];
                    if (bm != bh)
                    {
                        cerr << "Error: HashP bytes do not match: addr=" << addr << " pos+j=" << pos+j << " is bm=" << bm << " and it should be bh=" << bh << endl;
                        exit(-1);
                    }
                }
            }
            
            // Record the read operation
            HashRead hashRead;
            hashRead.pos = pos;
            hashRead.len = size;
            ctx.hashP[addr].reads.push_back(hashRead);

            // Store the size
            incHashPos = size;
        }

        if (rom.line[zkPC].hashPLen == 1)
        {
            pols.hashPLen[i] = 1;

            uint64_t lm = fe2n(fr, op0);
            uint64_t lh = ctx.hashP[addr].data.size();
            if (lm != lh)
            {
                cerr << "Error: HashK length does not match match addr=" << addr << " is lm=" << lm << " and it should be lh=" << lh << endl;
                exit(-1);
            }
            if (ctx.hashP[addr].digest.size() == 0)
            {
#ifdef LOG_TIME
                struct timeval t;
                gettimeofday(&t, NULL);
#endif
                PoseidonLinear(poseidon, ctx.hashP[addr].data, ctx.hashP[addr].digest);
                ctx.db.setProgram(ctx.hashP[addr].digest, ctx.hashP[addr].data);
#ifdef LOG_TIME
                poseidonTime += TimeDiff(t);
                poseidonTimes++;
#endif
                // Increment counter
                counters.hashPoseidon++;

#ifdef LOG_HASH
                cout << "Hash calculate hashPLen: addr:" << addr << " hash:" << ctx.hashP[addr].digest << " size:" << ctx.hashP[addr].data.size() << " data:";
                for (uint64_t k=0; k<ctx.hashP[addr].data.size(); k++) cout << byte2string(ctx.hashP[addr].data[k]) << ":";
                cout << endl;
#endif   
            }
        }

        if (rom.line[zkPC].hashPDigest == 1)
        {
            pols.hashPDigest[i] = 1;

            // Get contents of op into dg
            mpz_class dg;
            fea2scalar(fr, dg, op0, op1, op2, op3, op4, op5, op6, op7);

            if (ctx.hashP.find(addr) == ctx.hashP.end())
            {
                HashValue hashValue;
                hashValue.digest = dg.get_str(10);
                ctx.db.getProgram(hashValue.digest, hashValue.data);
                ctx.hashP[addr] = hashValue;
            }

            // Check that digest equals op
            mpz_class digestScalar;
            digestScalar.set_str(ctx.hashP[addr].digest, 10);
            if (dg != digestScalar)
            {
                cerr << "Error: hashPDigest: Digest does not match op" << endl;
                exit(-1);
            }
        }

        // Copy ROM flags into the polynomials
        //if (rom.line[zkPC].ecRecover == 1) pols.ecRecover[i] = 1; TODO: Check if this is correct

        // If arith, check that A*B + C = D<<256 + op, using scalars (result can be a big number)
        if (rom.line[zkPC].arith == 1)
        {
            counters.arith++;

            if (rom.line[zkPC].arithEq0==1 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==0)
            {            
                // Convert to scalar
                mpz_class A, B, C, D, op;
                fea2scalar(fr, A, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, B, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, C, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
                fea2scalar(fr, D, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
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
                pols.arith[i] = 1;
                pols.arithEq0[i] = 1;
            }
            else
            {
                // Convert to scalar
                mpz_class x1, y1, x2, y2, x3, y3;
                fea2scalar(fr, x1, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, y1, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, x2, pols.C0[i], pols.C1[i], pols.C2[i], pols.C3[i], pols.C4[i], pols.C5[i], pols.C6[i], pols.C7[i]);
                fea2scalar(fr, y2, pols.D0[i], pols.D1[i], pols.D2[i], pols.D3[i], pols.D4[i], pols.D5[i], pols.D6[i], pols.D7[i]);
                fea2scalar(fr, x3, pols.D0[i], pols.E1[i], pols.E2[i], pols.E3[i], pols.E4[i], pols.E5[i], pols.E6[i], pols.E7[i]);
                fea2scalar(fr, y3, op0, op1, op2, op3, op4, op5, op6, op7);

                bool dbl = false;
                if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==1 && rom.line[zkPC].arithEq2==0 && rom.line[zkPC].arithEq3==1)
                {
                    dbl = false;
                }
                else if (rom.line[zkPC].arithEq0==0 && rom.line[zkPC].arithEq1==0 && rom.line[zkPC].arithEq2==1 && rom.line[zkPC].arithEq3==1)
                {
                    dbl = true;
                }
                else
                {
                    cerr << "Error: Invalid arithmetic op" << endl;
                    exit(-1);
                }

                RawFec::Element s;
                if (dbl)
                {
                    // s = 3*(x1^2)/(2*y1)
                    RawFec::Element fecX1, fecY1, numerator, denominator;
                    fec.fromString(fecX1, x1.get_str());
                    fec.fromString(fecY1, y1.get_str());

                    // numerator = 3*(x1^2)
                    fec.mul(numerator, fecX1, fecX1);
                    fec.fromUI(denominator, 3);
                    fec.mul(numerator, numerator, denominator);

                    // denominator = 2*y1 = y1+y1
                    fec.add(denominator, fecY1, fecY1);

                    // s = numerator/denominator
                    fec.div(s, numerator, denominator);

                    // TODO: y1 == 0 => division by zero ==> how manage? Feli
                }
                else
                {
                    // s = (y2-y1)/(x2-x1)
                    RawFec::Element fecX1, fecY1, fecX2, fecY2, numerator, denominator;
                    fec.fromString(fecX1, x1.get_str());
                    fec.fromString(fecY1, y1.get_str());
                    fec.fromString(fecX2, x2.get_str());
                    fec.fromString(fecY2, y2.get_str());

                    // numerator = y2-y1
                    fec.sub(numerator, fecY2, fecY1);

                    // denominator = x2-x1
                    fec.add(denominator, fecX2, fecX1);

                    // s = numerator/denominator
                    fec.div(s, numerator, denominator);

                    // TODO: x2-x1 == 0 => division by zero ==> how manage? Feli
                }

                RawFec::Element fecX1, fecS, minuend, subtrahend;
                mpz_class _x3, _y3;
                
                // Calculate _x3 = s*s - x1 +(x1 if dbl, x2 otherwise)
                fec.fromString(fecX1, x1.get_str());
                if (dbl)
                {
                    fec.add(subtrahend, fecX1, fecX1 );
                }
                else
                {
                    RawFec::Element fecX2;
                    fec.fromString(fecX2, x2.get_str());
                    fec.add(subtrahend, fecX1, fecX1 );
                }
                fec.mul(minuend, s, s);
                fec.sub(fecS, minuend, subtrahend);
                _x3.set_str(fec.toString(fecS), 10);

                // Calculate _y3 = s*(x1-x3) - y1
                RawFec::Element fecX3;
                fec.fromString(fecX3, x3.get_str());
                fec.sub(minuend, fecX1, fecX3);
                fec.mul(minuend, s, minuend);
                fec.fromString(subtrahend, y1.get_str());
                fec.sub(fecS, minuend, subtrahend);
                _y3.set_str(fec.toString(fecS), 10);

                // Compare
                bool x3eq = (x3 == _x3);
                bool y3eq = (y3 == _y3);

                if (!x3eq || !y3eq)
                {
                    cerr << "Error: Arithmetic curve " << (dbl?"dbl":"add") << "point does not match" << endl;
                    cerr << "x1=" << x1.get_str() << " y1=" << y1.get_str() << 
                            " x2=" << x2.get_str() << " y2=" << y2.get_str() << 
                            " x3=" << x3.get_str() << " y3=" << y3.get_str() << 
                            " _x3=" << _x3.get_str() << " _y3=" << _y3.get_str() << endl;
                    exit(-1);
                }

                pols.arith[i] = 1;
                pols.arithEq0[i] = rom.line[zkPC].arithEq0;
                pols.arithEq1[i] = rom.line[zkPC].arithEq1;
                pols.arithEq2[i] = rom.line[zkPC].arithEq2;
                pols.arithEq3[i] = rom.line[zkPC].arithEq3;
                
                // required.Arith.push({x1: x1, y1: y1, x2: dbl ? x1:x2, y2: dbl? y1:y2, x3: x3, y3: y3, selEq0: 0, selEq1: dbl ? 0 : 1, selEq2: dbl ? 1 : 0, selEq3: 1});
                // TODO: Store arithmetic actions
            }
        }

        // Copy ROM flags into the polynomials
        //if (rom.line[zkPC].shl == 1) pols.shl[i] = 1; TODO: Check if this is correct
        //if (rom.line[zkPC].shr == 1) pols.shr[i] = 1;

        if (rom.line[zkPC].bin == 1)
        {
            if (rom.line[zkPC].binOpcode == 0) // NOP
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 0});
            }
            else if (rom.line[zkPC].binOpcode == 1) // ADD
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a + b) & Mask256;
                if (c != expectedC)
                {
                    cerr << "Error: Binary ADD operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 1;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 1});
            }
            else if (rom.line[zkPC].binOpcode == 2) // SUB
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a + b + twoTo256) & Mask256;
                if (c != expectedC)
                {
                    cerr << "Error: Binary SUB operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 2;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 2});
            }
            else if (rom.line[zkPC].binOpcode == 3) // LT
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a < b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary LT operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 3;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 3});
            }
            else if (rom.line[zkPC].binOpcode == 4) // GT
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a > b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary GT operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 4;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 4});
            }
            else if (rom.line[zkPC].binOpcode == 5) // SLT
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);
                if (a >= twoTo255) a = a - twoTo256;
                if (b >= twoTo255) b = b - twoTo256;


                mpz_class expectedC;
                expectedC = (a < b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary SLT operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 5;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 5});
            }
            else if (rom.line[zkPC].binOpcode == 6) // SGT
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);
                if (a >= twoTo255) a = a - twoTo256;
                if (b >= twoTo255) b = b - twoTo256;


                mpz_class expectedC;
                expectedC = (a > b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary SGT operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 6;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 6});
            }
            else if (rom.line[zkPC].binOpcode == 7) // EQ
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a == b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary EQ operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 7;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 7});
            }
            else if (rom.line[zkPC].binOpcode == 8) // ISZERO
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a == 0);
                if (c != expectedC)
                {
                    cerr << "Error: Binary ISZERO operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 8;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 8});
            }
            else if (rom.line[zkPC].binOpcode == 9) // AND
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a & b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary AND operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 9;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 9});
            }
            else if (rom.line[zkPC].binOpcode == 10) // OR
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a | b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary OR operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 10;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 10});
            }
            else if (rom.line[zkPC].binOpcode == 11) // XOR
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a ^ b);
                if (c != expectedC)
                {
                    cerr << "Error: Binary XOR operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 11;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 11});
            }
            else if (rom.line[zkPC].binOpcode == 12) // NOT
            {
                mpz_class a, b, c;
                fea2scalar(fr, a, pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i], pols.A4[i], pols.A5[i], pols.A6[i], pols.A7[i]);
                fea2scalar(fr, b, pols.B0[i], pols.B1[i], pols.B2[i], pols.B3[i], pols.B4[i], pols.B5[i], pols.B6[i], pols.B7[i]);
                fea2scalar(fr, c, op0, op1, op2, op3, op4, op5, op6, op7);

                mpz_class expectedC;
                expectedC = (a ^ Mask256);
                if (c != expectedC)
                {
                    cerr << "Error: Binary NOT operation does not match" << endl;
                    exit(-1);
                }
                
                pols.binOpcode[i] = 12;
            
                // TODO: required.Binary.push({a: a, b: b, c: c, opcode: 12});
            }
            else
            {
                cerr << "Error: Invalid binary operation opcode" << rom.line[zkPC].binOpcode <<  endl;
                exit(-1);
            }
            pols.bin[i] = 1;
        }

        //if (rom.line[zkPC].comparator == 1) pols.comparator[i] = 1; TODO: Check if this is correct
        if (rom.line[zkPC].opcodeRomMap == 1) pols.opcodeRomMap[i] = 1;

        // If setHASHPOS, HASHPOS' = op0 + incHashPos
        if (rom.line[zkPC].setHASHPOS == 1) {
            pols.HASHPOS[nexti] = fe2n(fr, op0) + incHashPos;
            pols.setHASHPOS[i] = 1;
        } else {
            pols.HASHPOS[nexti] = pols.HASHPOS[i] + incHashPos;
        }

        // Evaluate the list cmdAfter commands, and any children command, recursively
        for (uint64_t j=0; j<rom.line[zkPC].cmdAfter.size(); j++)
        {
            CommandResult cr;
            evalCommand(ctx, *rom.line[zkPC].cmdAfter[j], cr);
        }

#ifdef LOG_STEPS
        cout << "<-- Completed step: " << ii << " zkPC: " << zkPC << " op0: " << fr.toString(op0,16) << " A0: " << fr.toString(pols.A0[i],16) << " FREE0: " << fr.toString(pols.FREE0[i],16) << endl;
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
            byte4Pols.freeIN[p] = num >> 16;
            byte4Pols.out[p] = last;
            p++;
            byte4Pols.freeIN[p] = num & 0xFFFF;
            byte4Pols.out[p] = num >> 16;
            p++;
            last = num;
        }
        byte4Pols.freeIN[p] = 0;
        byte4Pols.out[p] = last;
        p++;
        byte4Pols.freeIN[p] = 0;
        byte4Pols.out[p] = 0;
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
    ctx.pols.A0[0] = fr.zero();
    ctx.pols.A1[0] = fr.zero();
    ctx.pols.A2[0] = fr.zero();
    ctx.pols.A3[0] = fr.zero();
    ctx.pols.A4[0] = fr.zero();
    ctx.pols.A5[0] = fr.zero();
    ctx.pols.A6[0] = fr.zero();
    ctx.pols.B0[0] = fr.zero();
    ctx.pols.B1[0] = fr.zero();
    ctx.pols.B2[0] = fr.zero();
    ctx.pols.B3[0] = fr.zero();
    ctx.pols.B4[0] = fr.zero();
    ctx.pols.B5[0] = fr.zero();
    ctx.pols.B6[0] = fr.zero();
    ctx.pols.B7[0] = fr.zero();
    ctx.pols.C0[0] = fr.zero();
    ctx.pols.C1[0] = fr.zero();
    ctx.pols.C2[0] = fr.zero();
    ctx.pols.C3[0] = fr.zero();
    ctx.pols.C4[0] = fr.zero();
    ctx.pols.C5[0] = fr.zero();
    ctx.pols.C6[0] = fr.zero();
    ctx.pols.C7[0] = fr.zero();
    ctx.pols.D0[0] = fr.zero();
    ctx.pols.D1[0] = fr.zero();
    ctx.pols.D2[0] = fr.zero();
    ctx.pols.D3[0] = fr.zero();
    ctx.pols.D4[0] = fr.zero();
    ctx.pols.D5[0] = fr.zero();
    ctx.pols.D6[0] = fr.zero();
    ctx.pols.D7[0] = fr.zero();
    ctx.pols.E0[0] = fr.zero();
    ctx.pols.E1[0] = fr.zero();
    ctx.pols.E2[0] = fr.zero();
    ctx.pols.E3[0] = fr.zero();
    ctx.pols.E4[0] = fr.zero();
    ctx.pols.E5[0] = fr.zero();
    ctx.pols.E6[0] = fr.zero();
    ctx.pols.E7[0] = fr.zero();
    ctx.pols.SR0[0] = fr.zero();
    ctx.pols.SR1[0] = fr.zero();
    ctx.pols.SR2[0] = fr.zero();
    ctx.pols.SR3[0] = fr.zero();
    ctx.pols.SR4[0] = fr.zero();
    ctx.pols.SR5[0] = fr.zero();
    ctx.pols.SR6[0] = fr.zero();
    ctx.pols.SR7[0] = fr.zero();
    ctx.pols.CTX[0] = 0;
    ctx.pols.SP[0] = 0;
    ctx.pols.PC[0] = 0;
    ctx.pols.MAXMEM[0] = 0;
    ctx.pols.GAS[0] = 0;
    ctx.pols.zkPC[0] = 0;
}

// Check that last evaluation (which is in fact the first one) is zero
void Executor::checkFinalState(Context &ctx)
{
    if ( 
        (!fr.isZero(ctx.pols.A0[0])) ||
        (!fr.isZero(ctx.pols.A1[0])) ||
        (!fr.isZero(ctx.pols.A2[0])) ||
        (!fr.isZero(ctx.pols.A3[0])) ||
        (!fr.isZero(ctx.pols.A4[0])) ||
        (!fr.isZero(ctx.pols.A5[0])) ||
        (!fr.isZero(ctx.pols.A6[0])) ||
        (!fr.isZero(ctx.pols.A7[0])) ||
        (!fr.isZero(ctx.pols.B0[0])) ||
        (!fr.isZero(ctx.pols.B1[0])) ||
        (!fr.isZero(ctx.pols.B2[0])) ||
        (!fr.isZero(ctx.pols.B3[0])) ||
        (!fr.isZero(ctx.pols.B4[0])) ||
        (!fr.isZero(ctx.pols.B5[0])) ||
        (!fr.isZero(ctx.pols.B6[0])) ||
        (!fr.isZero(ctx.pols.B7[0])) ||
        (!fr.isZero(ctx.pols.C0[0])) ||
        (!fr.isZero(ctx.pols.C1[0])) ||
        (!fr.isZero(ctx.pols.C2[0])) ||
        (!fr.isZero(ctx.pols.C3[0])) ||
        (!fr.isZero(ctx.pols.C4[0])) ||
        (!fr.isZero(ctx.pols.C5[0])) ||
        (!fr.isZero(ctx.pols.C6[0])) ||
        (!fr.isZero(ctx.pols.C7[0])) ||
        (!fr.isZero(ctx.pols.D0[0])) ||
        (!fr.isZero(ctx.pols.D1[0])) ||
        (!fr.isZero(ctx.pols.D2[0])) ||
        (!fr.isZero(ctx.pols.D3[0])) ||
        (!fr.isZero(ctx.pols.D4[0])) ||
        (!fr.isZero(ctx.pols.D5[0])) ||
        (!fr.isZero(ctx.pols.D6[0])) ||
        (!fr.isZero(ctx.pols.D7[0])) ||
        (!fr.isZero(ctx.pols.E0[0])) ||
        (!fr.isZero(ctx.pols.E1[0])) ||
        (!fr.isZero(ctx.pols.E2[0])) ||
        (!fr.isZero(ctx.pols.E3[0])) ||
        (!fr.isZero(ctx.pols.E4[0])) ||
        (!fr.isZero(ctx.pols.E5[0])) ||
        (!fr.isZero(ctx.pols.E6[0])) ||
        (!fr.isZero(ctx.pols.E7[0])) ||
        (!fr.isZero(ctx.pols.SR0[0])) ||
        (!fr.isZero(ctx.pols.SR1[0])) ||
        (!fr.isZero(ctx.pols.SR2[0])) ||
        (!fr.isZero(ctx.pols.SR3[0])) ||
        (!fr.isZero(ctx.pols.SR4[0])) ||
        (!fr.isZero(ctx.pols.SR5[0])) ||
        (!fr.isZero(ctx.pols.SR6[0])) ||
        (!fr.isZero(ctx.pols.SR7[0])) ||
        (ctx.pols.CTX[0]!=0) ||
        (ctx.pols.SP[0]!=0) ||
        (ctx.pols.PC[0]!=0) ||
        (ctx.pols.MAXMEM[0]!=0) ||
        (ctx.pols.GAS[0]!=0) ||
        (ctx.pols.zkPC[0]!=0)
    ) {
        cerr << "Error: Program terminated with registers not set to zero" << endl;
        exit(-1);
    }
    else{
        //cout << "checkFinalState() succeeded" << endl;
    }
}
