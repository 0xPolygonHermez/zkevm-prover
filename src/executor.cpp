
#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "ffiasm/fr.hpp"
#include "executor.hpp"
#include "rom_line.hpp"
#include "rom_command.hpp"
#include "rom.hpp"
#include "context.hpp"

using namespace std;
using json = nlohmann::json;

/******************/
/* DATA STRUCTURE */
/******************/


/*
   Polynomials size:
   Today all pols have the size of a finite field element, but some pols will only be 0 or 1 (way smaller).
   It is not efficient to user 256 bits to store just 1 bit.
   The PIL JSON file will specify the type of every polynomial: bit, byte, 32b, 64b, field element.
   pols[n will store the polynomial type, size, ID, and a pointer to the memory area.
*/
/*
   Polynomials memory:
   Using memory mapping to HDD file.
   TODO: Allocate dynamically according to the PIL JSON file contents.
*/

#define INVALID_ID 0xFFFFFFFFFFFFFFFF
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

#define MEMORY_SIZE 1000 // TODO: decide maximum size
#define MEM_OFFSET 0x300000000
#define STACK_OFFSET 0x200000000
#define CODE_OFFSET 0x100000000
#define CTX_OFFSET 0x400000000

void createPols(Context &ctx, json &pil);
void mapPols(Context &ctx);
void unmapPols(Context &ctx);
void initState(RawFr &fr, Context &ctx);
void preprocessTxs(Context &ctx, json &input);
int64_t fe2n(RawFr &fr, RawFr::Element &fe);
void printRegs(Context &ctx);
void printVars(Context &ctx);
void printMem(Context &ctx);

void fea2bn(Context &ctx, mpz_t &result, RawFr::Element fe0, RawFr::Element fe1, RawFr::Element fe2, RawFr::Element fe3);

//#define pols (*ctx.pPols) // TODO: Decide the way to identify the pols in the code
//#define rom ctx.pRom

RawFr::Element evalCommand(Context &ctx, romCommand &cmd);

/***********/
/* Execute */
/***********/

void execute(RawFr &fr, json &input, json &romJson, json &pil, string &outputFile)
{
    cout << "execute()" << endl;

    Context ctx;
    ctx.pFr = &fr;
    ctx.outputFile = outputFile;

    // opN are local, uncommitted polynomials
    RawFr::Element op3, op2, op1, op0;

    /* Load ROM JSON file content to memory */
    loadRom(ctx, romJson);

    /* Create polynomials list in memory */
    createPols(ctx, pil);

    /* Create and map pols file to memory */
    mapPols(ctx);

    /* Sets first evaluation of all polynomials to zero */
    initState(fr, ctx);

    preprocessTxs(ctx, input);

    uint64_t i = 0; // execution line

    for (uint64_t step=0; step<NEVALUATIONS; step++)
    {
        //i = fe2n(fr, pols[zkPC][i]); // This is the read line of code, but using step for debugging purposes, to execute all possible instructions
        i=step;
        ctx.ln = i;
        ctx.step = step; // To be used inside evaluateCommand() to find the current value of the registers, e.g. (*ctx.pPols)[A0][ctx.step]

        if ( i>=ctx.romSize )
        {
            cout << "Reached end of rom" << endl;
            break;
        }

        ctx.fileName = rom[i].fileName; // TODO: Is this required?  It is only used in printRegs(), and it is an overhead in every loop.
        ctx.line = rom[i].line; // TODO: Is this required? It is only used in printRegs(), and it is an overhead in every loop.

        for (uint64_t j=0; j<rom[i].cmdBefore.size(); j++)
        {
            evalCommand(ctx, *rom[i].cmdBefore[j]);
        }

        op0 = fr.zero();
        op1 = fr.zero();
        op2 = fr.zero();
        op3 = fr.zero();

        // inX adds the corresponding register values to the op local register set
        // In case several inXs are set to 1, those values will be added
        if (rom[i].inA)
        {
            fr.add(op0, op0, pols[A0][i]);
            fr.add(op1, op1, pols[A1][i]);
            fr.add(op2, op2, pols[A2][i]);
            fr.add(op3, op3, pols[A3][i]);
            pols[inA][i] = fr.one();
        }
        else {
            pols[inA][i] = fr.zero();
        }
        
        if (rom[i].inB) {
            fr.add(op0, op0, pols[B0][i]);
            fr.add(op1, op1, pols[B1][i]);
            fr.add(op2, op2, pols[B2][i]);
            fr.add(op3, op3, pols[B3][i]);
            pols[inB][i] = fr.one();
        } else {
            pols[inB][i] = fr.zero();
        }

        if (rom[i].inC) {
            fr.add(op0, op0, pols[C0][i]);
            fr.add(op1, op1, pols[C1][i]);
            fr.add(op2, op2, pols[C2][i]);
            fr.add(op3, op3, pols[C3][i]);
            pols[inC][i] = fr.one();
        } else {
            pols[inC][i] = fr.zero();
        }

        if (rom[i].inD) {
            fr.add(op0, op0, pols[D0][i]);
            fr.add(op1, op1, pols[D1][i]);
            fr.add(op2, op2, pols[D2][i]);
            fr.add(op3, op3, pols[D3][i]);
            pols[inD][i] = fr.one();
        } else {
            pols[inD][i] = fr.zero();
        }

        if (rom[i].inE) {
            fr.add(op0, op0, pols[E0][i]);
            fr.add(op1, op1, pols[E1][i]);
            fr.add(op2, op2, pols[E2][i]);
            fr.add(op3, op3, pols[E3][i]);
            pols[inE][i] = fr.one();
        } else {
            pols[inE][i] = fr.zero();
        }

        if (rom[i].inSR) {
            fr.add(op0, op0, pols[SR][i]);
            pols[inSR][i] = fr.one();
        } else {
            pols[inSR][i] = fr.zero();
        }

        if (rom[i].inCTX) {
            fr.add(op0, op0, pols[CTX][i]);
            pols[inCTX][i] = fr.one();
        } else {
            pols[inCTX][i] = fr.zero();
        }

        if (rom[i].inSP) {
            fr.add(op0, op0, pols[SP][i]);
            pols[inSP][i] = fr.one();
        } else {
            pols[inSP][i] = fr.zero();
        }

        if (rom[i].inPC) {
            fr.add(op0, op0, pols[PC][i]);
            pols[inPC][i] = fr.one();
        } else {
            pols[inPC][i] = fr.zero();
        }
        
        if (rom[i].inGAS) {
            fr.add(op0, op0, pols[GAS][i]);
            pols[inGAS][i] = fr.one();
        } else {
            pols[inGAS][i] = fr.zero();
        }

        if (rom[i].inMAXMEM) {
            fr.add(op0, op0, pols[MAXMEM][i]);
            pols[inMAXMEM][i] = fr.one();
        } else {
            pols[inMAXMEM][i] = fr.zero();
        }

        if (rom[i].inSTEP) {
            RawFr::Element eI;
            fr.fromUI(eI, i);
            fr.add(op0, op0, eI);
            pols[inSTEP][i] = fr.one();
        } else {
            pols[inSTEP][i] = fr.zero();
        }

        if (rom[i].bConstPresent) {
            fr.fromUI(pols[CONST][i], rom[i].CONST);
            fr.add(op0, op0, pols[CONST][i]);
        } else {
            pols[CONST][i] = fr.zero();
        }

        uint64_t addrRel = 0; // TODO: Check with Jordi if this is the right type for an address
        uint64_t addr = 0;

        // If address involved, load offset into addr
        if (rom[i].mRD || rom[i].mWR || rom[i].hashRD || rom[i].hashWR || rom[i].hashE || rom[i].JMP || rom[i].JMPC) {
            if (rom[i].ind)
            {
                addrRel = fe2n(fr, pols[E0][i]);
            }
            if (rom[i].bOffsetPresent)
            {
                int64_t offset = rom[i].offset;
                // If offset is possitive, and the sum is too big, fail
                if (offset>0 && (addrRel+offset)>=0x100000000)
                {
                    cerr << "Error: addrRel >= 0x100000000 ln: " << ctx.ln << endl;
                    exit(-1); // TODO: Should we kill the process?                    
                }
                // If offset is negative, and its modulo is bigger than addrRel, fail
                if (offset<0 && (-offset)>addrRel)
                {
                    cerr << "Error: addrRel < 0 ln: " << ctx.ln << endl;
                    exit(-1); // TODO: Should we kill the process?
                }
                addrRel += offset;
            }
            addr = addrRel;
        }

        if (rom[i].useCTX) {
            addr += CTX_OFFSET;
            pols[useCTX][i] = fr.one();
        } else {
            pols[useCTX][i] = fr.zero();
        }

        if (rom[i].isCode) {
            addr += CODE_OFFSET;
            pols[isCode][i] = fr.one();
        } else {
            pols[isCode][i] = fr.zero();
        }

        if (rom[i].isStack) {
            addr += STACK_OFFSET;
            pols[isStack][i] = fr.one();
        } else {
            pols[isStack][i] = fr.zero();
        }

        if (rom[i].isMem) {
            addr += MEM_OFFSET;
            pols[isMem][i] = fr.one();
        } else {
            pols[isMem][i] = fr.zero();
        }

        if (rom[i].inc) {
            //fr.fromUI(pols[inc][i],rom[i].inc);
            // TODO: Migrate
            pols[inc][i] = fr.one();
        } else {
            pols[inc][i] = fr.zero();
        }

        if (rom[i].dec) {
            //fr.fromUI(pols[inc][i],rom[i].inc);
            // TODO: Migrate
            pols[dec2][i] = fr.one();
        } else {
            pols[dec2][i] = fr.zero();
        }

        if (rom[i].bIndPresent) {
            fr.fromUI(pols[ind][i],rom[i].ind);
        } else {
            pols[ind][i] = fr.zero();
        }

        if (rom[i].bOffsetPresent) {
            fr.fromUI(pols[offset][i],rom[i].offset);
        } else {
            pols[offset][i] = fr.zero();
        }

        /*
        if (l.inFREE) {

            if (!l.freeInTag) {
                throw new Error(`Instruction with freeIn without freeInTag: ${ctx.ln}`);
            }
            
            let fi;
            if (l.freeInTag.op=="") {
                let nHits = 0;
                if (l.mRD == 1) {
                    if (typeof ctx.mem[addr] != "undefined") {
                        fi = ctx.mem[addr];
                    } else {
                        fi = [Fr.zero, Fr.zero, Fr.zero, Fr.zero];
                    }
                    nHits++;
                }
                if (l.sRD == 1) {
                    const saddr = fe2bns([ctx.A0, ctx.A1, ctx.A2, ctx.A3]); // 256 bis number, scalar > fe size
                    if (typeof ctx.sto[saddr] === "undefined" ) throw new Error(`Storage not initialized: ${ctx.ln}`);
                    fi = ctx.sto[ saddr ];
                    nHits++;
                } // Library gmp -> C big numbers library, function to convert fe to bn
                // raw->toMpz()void toMpz(mpz_t r, Element &a); DNS google 8.8.8.8
                if (l.hashRD == 1) {
                    if (!ctx.hash[addr]) throw new Error("Hash address not initialized");
                    if (typeof ctx.hash[addr].result == "undefined") throw new Error("Hash not finalized");
                    fi = bn2bna(Fr, ctx.hash[addr].result);
                    nHits++;
                }
                if (l.ecRecover == 1) {
                    const d = ethers.utils.hexlify(fea2bn(Fr, ctx.A));
                    const r = ethers.utils.hexlify(fea2bn(Fr, ctx.B));
                    const s = ethers.utils.hexlify(fea2bn(Fr, ctx.C));
                    const v = ethers.utils.hexlify(fe2n(Fr, ctx.D[0]));
                    const raddr = .recoverAddress(d, {
                        r: r,
                        s: s,
                        v: v
                    });
                    fi = bn2bna(Fr, raddr);
                    nHits++;
                }
                if (l.shl == 1) {
                    const a = Scalar.e(fea2bn(Fr, ctx.A));
                    const s = fe2n(Fr, ctx.D[0]);
                    if ((s>32) || (s<0)) throw new Error(`SHL too big: ${ctx.ln}`);
                    fi = bn2bna(Fr, Scalar.band(Scalar.shl(a, s*8), Scalar.e("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")));
                    nHits++;
                } 
                if (l.shr == 1) {
                    const a = Scalar.e(fea2bn(Fr, ctx.A));
                    const s = fe2n(Fr, ctx.D[0]);
                    if ((s>32) || (s<0)) throw new Error(`SHR too big: ${ctx.ln}`);
                    fi = bn2bna(Fr, Scalar.shr(a, s*8));
                    nHits++;
                } 
                if (nHits==0) {
                    throw new Error(`Empty freeIn without a valid instruction: ${ctx.ln}`);
                }
                if (nHits>1) {
                    throw new Error(`Only one instructuin that requires freeIn is alllowed: ${ctx.ln}`);
                }
            } else {
                fi = evalCommand(ctx, l.freeInTag);
            }
            [pols.main.FREE0[i], pols.main.FREE1[i], pols.main.FREE2[i], pols.main.FREE3[i]] = fi;
            [op0, op1, op2, op3] = [Fr.add(op0, fi[0]), Fr.add(op1, fi[1]), Fr.add(op2, fi[2]), Fr.add(op3, fi[3])];
            pols.main.inFREE[i] = Fr.one;

        } else {
            [pols.main.FREE0[i], pols.main.FREE1[i], pols.main.FREE2[i], pols.main.FREE3[i]] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero];
            pols.main.inFREE[i] = Fr.zero;
        }*/

        if (rom[i].neg) {
            fr.neg(op0,op0);
            pols[neg][i] = fr.one();
        } else {
            pols[neg][i] = fr.zero();
        }

        if (rom[i].assert) {
            if ( (!fr.eq(pols[A0][i],op0)) ||
                 (!fr.eq(pols[A1][i],op1)) ||
                 (!fr.eq(pols[A2][i],op2)) ||
                 (!fr.eq(pols[A3][i],op3)) )
            {
                cerr << "Error: ROM assert failed: AN!=opN ln: " << ctx.ln << endl;
                //exit(-1); // TODO: Should we kill the process?  Temporarly disabling because assert is failing, since executor is not completed
            }
            pols[assert][i] = fr.one();
        } else {
            pols[assert][i] = fr.zero();
        }

        if (rom[i].setA) {
            fr.copy(pols[A0][i+1],op0);
            fr.copy(pols[A1][i+1],op1);
            fr.copy(pols[A2][i+1],op2);
            fr.copy(pols[A3][i+1],op3);
            pols[setA][i] = fr.one();
        } else {
            fr.copy(pols[A0][i+1],pols[A0][i]);
            fr.copy(pols[A1][i+1],pols[A1][i]);
            fr.copy(pols[A2][i+1],pols[A2][i]);
            fr.copy(pols[A3][i+1],pols[A3][i]);
            pols[setA][i] = fr.zero();
        }

        if (rom[i].setB) {
            fr.copy(pols[B0][i+1],op0);
            fr.copy(pols[B1][i+1],op1);
            fr.copy(pols[B2][i+1],op2);
            fr.copy(pols[B3][i+1],op3);
            pols[setB][i] = fr.one();
        } else {
            fr.copy(pols[B0][i+1],pols[B0][i]);
            fr.copy(pols[B1][i+1],pols[B1][i]);
            fr.copy(pols[B2][i+1],pols[B2][i]);
            fr.copy(pols[B3][i+1],pols[B3][i]);
            pols[setB][i] = fr.zero();
        }

        if (rom[i].setC) {
            fr.copy(pols[C0][i+1],op0);
            fr.copy(pols[C1][i+1],op1);
            fr.copy(pols[C2][i+1],op2);
            fr.copy(pols[C3][i+1],op3);
            pols[setC][i] = fr.one();
        } else {
            fr.copy(pols[C0][i+1],pols[C0][i]);
            fr.copy(pols[C1][i+1],pols[C1][i]);
            fr.copy(pols[C2][i+1],pols[C2][i]);
            fr.copy(pols[C3][i+1],pols[C3][i]);
            pols[setC][i] = fr.zero();
        }

        if (rom[i].setD) {
            fr.copy(pols[D0][i+1],op0);
            fr.copy(pols[D1][i+1],op1);
            fr.copy(pols[D2][i+1],op2);
            fr.copy(pols[D3][i+1],op3);
            pols[setD][i] = fr.one();
        } else {
            fr.copy(pols[D0][i+1],pols[D0][i]);
            fr.copy(pols[D1][i+1],pols[D1][i]);
            fr.copy(pols[D2][i+1],pols[D2][i]);
            fr.copy(pols[D3][i+1],pols[D3][i]);
            pols[setD][i] = fr.zero();
        }

        if (rom[i].setE) {
            fr.copy(pols[E0][i+1],op0);
            fr.copy(pols[E1][i+1],op1);
            fr.copy(pols[E2][i+1],op2);
            fr.copy(pols[E3][i+1],op3);
            pols[setE][i] = fr.one();
        } else {
            fr.copy(pols[E0][i+1],pols[E0][i]);
            fr.copy(pols[E1][i+1],pols[E1][i]);
            fr.copy(pols[E2][i+1],pols[E2][i]);
            fr.copy(pols[E3][i+1],pols[E3][i]);
            pols[setE][i] = fr.zero();
        }

        if (rom[i].setSR) {
            fr.copy(pols[SR][i+1],op0);
            pols[setSR][i] = fr.one();
        } else {
            fr.copy(pols[SR][i+1],pols[SR][i]);
            pols[setSR][i] = fr.zero();
        }

        if (rom[i].setCTX) {
            fr.copy(pols[CTX][i+1],op0);
            pols[setCTX][i] = fr.one();
        } else {
            fr.copy(pols[CTX][i+1],pols[CTX][i]);
            pols[setCTX][i] = fr.zero();
        }

        if (rom[i].setSP) {
            fr.copy(pols[SP][i+1],op0);
            pols[setSP][i] = fr.one();
        } else {
            fr.copy(pols[SP][i+1],pols[SP][i]);
            if ((rom[i].inc)&&(rom[i].isStack)){
                fr.add(pols[SP][i+1], pols[SP][i+1], fr.one());
            }
            if ((rom[i].dec)&&(rom[i].isStack)){
                fr.sub(pols[SP][i+1], pols[SP][i+1], fr.one());
            }
            pols[setSP][i] = fr.zero();
        }

        if (rom[i].setPC) {
            fr.copy(pols[PC][i+1],op0);
            pols[setPC][i] = fr.one();
        } else {
            fr.copy(pols[PC][i+1],pols[PC][i]);
            if ( (rom[i].inc) && (rom[i].isCode) ) {
                fr.add(pols[PC][i+1], pols[PC][i+1], fr.one()); // PC is part of Ethereum's program
            }
            if ( (rom[i].dec) && (rom[i].isCode) ) {
                fr.sub(pols[PC][i+1], pols[PC][i+1], fr.one()); // PC is part of Ethereum's program
            }
            pols[setPC][i] = fr.zero();
        }

        if (rom[i].JMPC) {
            int64_t o = 0; // TODO: migrate const o = fe2n(Fr, op0);
            if (o<0) {
                pols[isNeg][i] = fr.one();
                fr.fromUI(pols[zkPC][i+1], addr);
            } else {
                pols[isNeg][i] = fr.zero();
                fr.add(pols[zkPC][i+1], pols[zkPC][i], fr.one());
            }
            pols[JMP][i] = fr.zero();
            pols[JMPC][i] = fr.one();
        } else if (rom[i].JMP) {
            pols[isNeg][i] = fr.zero();
            fr.fromUI(pols[zkPC][i+1], addr);
            pols[JMP][i] = fr.one();
            pols[JMPC][i] = fr.zero();
        } else {
            pols[isNeg][i] = fr.zero();
            fr.add(pols[zkPC][i+1], pols[zkPC][i], fr.one());
            pols[JMP][i] = fr.zero();
            pols[JMPC][i] = fr.zero();
        }

        uint64_t maxMemCalculated = 0;
        uint64_t mm = fe2n(fr, pols[MAXMEM][i]);
        if (rom[i].isMem && addrRel>mm) {
            pols[isMaxMem][i] = fr.one();
            maxMemCalculated = addrRel;
        } else {
            pols[isMaxMem][i] = fr.zero();
            maxMemCalculated = mm;
        }

        if (rom[i].setMAXMEM) {
            pols[MAXMEM][i+1] = op0;
            pols[setMAXMEM][i] = fr.one();
        } else {
            fr.fromUI(pols[MAXMEM][i+1],maxMemCalculated);
            pols[setMAXMEM][i] = fr.zero();
        }

        if (rom[i].setGAS) {
            pols[GAS][i+1] = op0;
            pols[setGAS][i] = fr.one();
        } else {
            pols[GAS][i+1] = pols[GAS][i];
            pols[setGAS][i] = fr.zero();
        }

        if (rom[i].mRD) { // TODO: Shouldn't we read from memory?
            pols[mRD][i] = fr.one();
        } else {
            pols[mRD][i] = fr.zero();
        }

        if (rom[i].mWR) {
            ctx.mem[addr][0] = op0;
            ctx.mem[addr][1] = op1;
            ctx.mem[addr][2] = op2;
            ctx.mem[addr][3] = op3;
            pols[mWR][i] = fr.one();
        } else {
            pols[mWR][i] = fr.zero();
        }

        if (rom[i].sRD) {
            pols[sRD][i] = fr.one();
        } else {
            pols[sRD][i] = fr.zero();
        }

/*
        if (l.sWR) {
            pols.main.sWR[i] = Fr.one;
            if (typeof ctx.sto[addr] === "undefined" ) throw new Error(`Storage not initialized: ${ctx.ln}`);
            ctx.sto[ fe2bns([pols.A0[i], pols.A1[i], pols.A2[i], pols.A3[i]]) ] = op;
        } else {
            pols.main.sWR[i] = Fr.zero;
        }
*/
        if (rom[i].hashRD) {
            pols[hashRD][i] = fr.one();
        } else {
            pols[hashRD][i] = fr.zero();
        }
/*
        if (l.hashWR) {
            pols.main.hashWR[i] = Fr.one;

            size = fe2n(Fr, ctx.D[0]);
            if ((size<0) || (size>32)) throw new Error(`Invalid size for hash: ${ctx.ln}`);
            const a = fea2bn(Fr, [op0, op1, op2, op3]);
            if (!ctx.hash[addr]) ctx.hash[addr] = { data: [] } ; // Array: string hexa
            for (let i=0; i<size; i++) {
                ctx.hash[addr].data.push(Scalar.toNumber(Scalar.band( Scalar.shr( a, (size-i -1)*8 ) , Scalar.e("0xFF"))));
            } // storing bytes 1 by 1, hash is a bytes vector
        } else {
            pols.main.hashWR[i] = Fr.zero;
        }

        if (l.hashE) {
            pols.main.hashE[i] = Fr.one;
            
            ctx.hash[addr].result = ethers.utils.keccak256(ethers.utils.hexlify(ctx.hash[addr].data)); // array of bytes, hexa string
        } else {
            pols.main.hashE[i] = Fr.zero;
        }
*/
        if (rom[i].ecRecover) {
            pols[ecRecover][i] = fr.one();
        } else {
            pols[ecRecover][i] = fr.zero();
        }

        if (rom[i].arith) {
            pols[arith][i] = fr.one();
        } else {
            pols[arith][i] = fr.zero();
        }

        if (rom[i].shl) {
            pols[shl][i] = fr.one();
        } else {
            pols[shl][i] = fr.zero();
        }

        if (rom[i].shr) {
            pols[shr][i] = fr.one();
        } else {
            pols[shr][i] = fr.zero();
        }

        if (rom[i].bin) {
            pols[bin][i] = fr.one();
        } else {
            pols[bin][i] = fr.zero();
        }

        if (rom[i].comparator) {
            pols[comparator][i] = fr.one();
        } else {
            pols[comparator][i] = fr.zero();
        }

        if (rom[i].opcodeRomMap) {
            pols[opcodeRomMap][i] = fr.one();
        } else {
            pols[opcodeRomMap][i] = fr.zero();
        }

        for (uint64_t j=0; j<rom[i].cmdAfter.size(); j++)
        {
            evalCommand(ctx, *rom[i].cmdAfter[j]);
        }

    }

    printRegs(ctx);
    printVars(ctx);
    printMem(ctx);

    /* Unmap output file from memory */
    unmapPols(ctx);

    /*for (uint64_t i=0; i<ctx.romSize; i++)
    {
        freeRomCommandArray(rom[i].cmdBefore);
        freeRomCommand(rom[i].freeInTag);
        freeRomCommandArray(rom[i].cmdAfter);
    }
    delete[] rom;*/
    unloadRom(ctx);
    
}

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

/* Sets first evaluation of all polynomials to zero */
void initState(RawFr &fr, Context &ctx)
{
    // Register value initial parameters
    pols[A0][0] = fr.zero();
    pols[A1][0] = fr.zero();
    pols[A2][0] = fr.zero();
    pols[A3][0] = fr.zero();
    pols[B0][0] = fr.zero();
    pols[B1][0] = fr.zero();
    pols[B2][0] = fr.zero();
    pols[B3][0] = fr.zero();
    pols[C0][0] = fr.zero();
    pols[C1][0] = fr.zero();
    pols[C2][0] = fr.zero();
    pols[C3][0] = fr.zero();
    pols[D0][0] = fr.zero();
    pols[D1][0] = fr.zero();
    pols[D2][0] = fr.zero();
    pols[D3][0] = fr.zero();
    pols[E0][0] = fr.zero();
    pols[E1][0] = fr.zero();
    pols[E2][0] = fr.zero();
    pols[E3][0] = fr.zero();
    pols[SR][0] = fr.zero();
    pols[CTX][0] = fr.zero();
    pols[SP][0] = fr.zero();
    pols[PC][0] = fr.zero();
    pols[MAXMEM][0] = fr.zero();
    pols[GAS][0] = fr.zero();
    pols[zkPC][0] = fr.zero();
}

RawFr::Element eval_number(Context &ctx, romCommand &cmd);
RawFr::Element eval_getReg(Context &ctx, romCommand &cmd);
RawFr::Element eval_declareVar(Context &ctx, romCommand &cmd);
RawFr::Element eval_setVar(Context &ctx, romCommand &cmd);
RawFr::Element eval_getVar(Context &ctx, romCommand &cmd);
RawFr::Element eval_add(Context &ctx, romCommand &cmd);
RawFr::Element eval_sub(Context &ctx, romCommand &cmd);
RawFr::Element eval_neg(Context &ctx, romCommand &cmd);
RawFr::Element eval_mul(Context &ctx, romCommand &cmd);
RawFr::Element eval_div(Context &ctx, romCommand &cmd);
RawFr::Element eval_mod(Context &ctx, romCommand &cmd);

RawFr::Element evalCommand(Context &ctx, romCommand &cmd) {
    if (cmd.op=="number") {
        return eval_number(ctx, cmd); // TODO: return a big number, an mpz, >253bits, here and in all evalXxx() to unify
    } else if (cmd.op=="declareVar") {
        return eval_declareVar(ctx, cmd);
    } else if (cmd.op=="setVar") {
        return eval_setVar(ctx, cmd);
    } else if (cmd.op=="getVar") {
        return eval_getVar(ctx, cmd);
    } else if (cmd.op=="getReg") {
        return eval_getReg(ctx, cmd);
    } else if (cmd.op=="functionCall") {
        //return eval_functionCall(ctx, cmd);
    } else if (cmd.op=="add") {
        return eval_add(ctx, cmd);
    } else if (cmd.op=="sub") {
        return eval_sub(ctx, cmd);
    } else if (cmd.op=="neg") {
        return eval_neg(ctx, cmd);
    } else if (cmd.op=="mul") {
        //return eval_mul(ctx, cmd);
    } else if (cmd.op=="div") {
        return eval_div(ctx, cmd);
    } else if (cmd.op=="mod") {
        return eval_mod(ctx, cmd);
    }
    cerr << "Error: evalCommand() found invalid operation: " << cmd.op << endl;
    exit(-1);
}

RawFr::Element eval_number(Context &ctx, romCommand &cmd) {
    RawFr::Element num;
    ctx.pFr->fromUI(num,cmd.num); // TODO: Check existence and type of num element
    return num;
}

/*************/
/* Variables */
/*************/

/* If defined, logs variable declaration, get and set actions */
#define LOG_VARIABLES

/* Declares a new variable, and fails if it already exists */
RawFr::Element eval_declareVar(Context &ctx, romCommand &cmd)
{
    // Check the variable name
    if (cmd.varName == "") {
        cerr << "Error: eval_declareVar() Variable name not found" << endl;
        exit(-1);  
    }

    // Check that this variable does not exists
    if ( ctx.vars.find(cmd.varName) != ctx.vars.end() ) {
        cerr << "Error: eval_declareVar() Variable already declared: " << cmd.varName << endl;
        exit(-1);
    }

    // Create the new variable with a zero value
    ctx.vars[cmd.varName] = ctx.pFr->zero(); // TODO: Should it be Scalar.e(0)?
#ifdef LOG_VARIABLES
    cout << "Declare variable: " << cmd.varName << endl;
#endif
    return ctx.vars[cmd.varName];
}

/* Gets the value of the variable, and fails if it does not exist */
RawFr::Element eval_getVar(Context &ctx, romCommand &cmd)
{
    // Check the variable name
    if (cmd.varName == "") {
        cerr << "Error: eval_getVar() Variable name not found" << endl;
        exit(-1);  
    }

    // Check that this variable exists
    if ( ctx.vars.find(cmd.varName) == ctx.vars.end() ) {
        cerr << "Error: eval_getVar() Undefined variable: " << cmd. varName << endl;
        exit(-1);
    }

#ifdef LOG_VARIABLES
    cout << "Get variable: " << cmd.varName << endl;
#endif
    return ctx.vars[cmd.varName];
}

string eval_left(Context &ctx, romCommand &cmd);

/* Sets variable to value, and fails if it does not exist */
RawFr::Element eval_setVar(Context &ctx, romCommand &cmd)
{
    // Check that tag contains a values array
    if (cmd.values.size()==0) {
        cerr << "Error: eval_setVar() could not find array values in setVar command" << endl;
        exit(-1);
    }
    

    // Get varName from the first element in values
    string varName = eval_left(ctx,*cmd.values[0]);

    // Check that this variable exists
    if ( ctx.vars.find(varName) == ctx.vars.end() ) {
        cerr << "Error: eval_setVar() Undefined variable: " << varName << endl;
        exit(-1);
    }

    ctx.vars[varName] = evalCommand(ctx, *cmd.values[1]);
#ifdef LOG_VARIABLES
    cout << "Set variable: " << varName << endl;
#endif
    return ctx.vars[varName];
}

string eval_left(Context &ctx, romCommand &cmd)
{
    if (cmd.op == "declareVar") {
        eval_declareVar(ctx, cmd);
        return cmd.varName;
    } else if (cmd.op == "getVar") {
        return cmd.varName;
    }
    cerr << "Error: invalid left expression, op: " << cmd.op << "ln: " << ctx.ln << endl;
    exit(-1);
}





RawFr::Element eval_getReg(Context &ctx, romCommand &cmd) {
    if (cmd.regName=="A") { // TODO: Consider using a string local variable to avoid searching every time
        //return fea2bn(ctx.pFr,ctx.pols[]);
        mpz_t result;
        mpz_init(result);
        fea2bn(ctx, result, pols[A0][ctx.step], pols[A1][ctx.step], pols[A2][ctx.step], pols[A3][ctx.step]);
        RawFr::Element feResult;
        ctx.pFr->fromMpz(feResult, result);
        mpz_clear(result);
        return feResult;
        //return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="B") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="C") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="D") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="E") {
        return ctx.pFr->zero(); // TODO: migrate
    } else if (cmd.regName=="SR") {
        return pols[SR][ctx.step];
    } else if (cmd.regName=="CTX") {
        return pols[CTX][ctx.step];
    } else if (cmd.regName=="SP") {
        return pols[SP][ctx.step];
    } else if (cmd.regName=="PC") {
        return pols[PC][ctx.step];
    } else if (cmd.regName=="MAXMEM") {
        return pols[MAXMEM][ctx.step];
    } else if (cmd.regName=="GAS") {
        return pols[GAS][ctx.step];
    } else if (cmd.regName=="zkPC") {
        return pols[zkPC][ctx.step];
    }
    cerr << "Error: eval_getReg() Invalid register: " << cmd.regName << ": " << ctx.ln << endl;
    exit(-1);
}
/*
function eval_getReg(ctx, tag) {
    if (tag.regName == "A") {
        return fea2bn(ctx.Fr, ctx.A);
    } else if (tag.regName == "B") {
        return fea2bn(ctx.Fr, ctx.B);
    } else if (tag.regName == "C") {
        return fea2bn(ctx.Fr, ctx.C);
    } else if (tag.regName == "D") {
        return fea2bn(ctx.Fr, ctx.D);
    } else if (tag.regName == "E") {
        return fea2bn(ctx.Fr, ctx.E);
    } else if (tag.regName == "SR") {
        return ctx.SR;
    } else if (tag.regName == "CTX") {
        return ctx.CTX;
    } else if (tag.regName == "SP") {
        return ctx.SP;
    } else if (tag.regName == "PC") {
        return ctx.PC;
    } else if (tag.regName == "MAXMEM") {
        return ctx.MAXMEM;
    } else if (tag.regName == "GAS") {
        return ctx.GAS;
    } else if (tag.regName == "zkPC") {
        return ctx.zkPC;
    } else {
        throw new Error(`Invalid register ${tag.regName}:  ${ctx.ln}`);
    }
}
*/
RawFr::Element eval_add(Context &ctx, romCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    ctx.pFr->add(r,a,b);
    return r;
}
/*
function eval_add(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Scalar.add(a,b);
}*/
RawFr::Element eval_sub(Context &ctx, romCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    ctx.pFr->sub(r,a,b);
    return r;
}
/*
function eval_sub(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Scalar.sub(a,b);
}*/
RawFr::Element eval_neg(Context &ctx, romCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element r;
    ctx.pFr->neg(r,a);
    return r;
}
/*
function eval_neg(ctx, tag) {
    const a = evalCommand(ctx, values[0]);
    return Scalar.neg(a);
}*/

/*
function eval_mul(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Sacalar.and(Scalar.mul(a,b), Mask256);
}*/
RawFr::Element eval_div(Context &ctx, romCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    ctx.pFr->div(r,a,b);
    return r;
}
/*
function eval_div(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Sacalar.div(a,b);
}*/
RawFr::Element eval_mod(Context &ctx, romCommand &cmd)
{
    RawFr::Element a = evalCommand(ctx,*cmd.values[0]);
    RawFr::Element b = evalCommand(ctx,*cmd.values[1]);
    RawFr::Element r;
    //ctx.pFr->mod(r,a,b); // TODO: Migrate.  This method does not exist in C.
    return r;
}
/*
function eval_mod(ctx, tag) {
    const a = evalCommand(ctx, tag.values[0]);
    const b = evalCommand(ctx, tag.values[1]);
    return Sacalar.mod(a,b);
}
*/
RawFr::Element eval_functionCall(Context &ctx, romCommand &cmd) {
    if (cmd.funcName == "getGlobalHash") {
        //return eval_getGlobalHash(ctx, tag);
    } else if (cmd.funcName == "getOldStateRoot") {
        //return eval_getOldStateRoot(ctx, tag);
    } else if (cmd.funcName == "getNewStateRoot") {
        //return eval_getNewStateRoot(ctx, tag);
    } else if (cmd.funcName == "getNTxs") {
        //return eval_getNTxs(ctx, tag);
    } else if (cmd.funcName == "getRawTx") {
        //return eval_getRawTx(ctx, tag);
    } else if (cmd.funcName == "getTxSigR") {
        //return eval_getTxSigR(ctx, tag);
    } else if (cmd.funcName == "getTxSigS") {
        //return eval_getTxSigS(ctx, tag);
    } else if (cmd.funcName == "getTxSigV") {
        //return eval_getTxSigV(ctx, tag);
    } else if (cmd.funcName == "getSequencerAddr") {
        //return eval_getSequencerAddr(ctx, tag);
    } else if (cmd.funcName == "getChainId") {
        //return eval_getChainId(ctx, tag);
    }
    cerr << "Error: eval_functionCall() function not defined: " << cmd.funcName << " line: " << ctx.ln << endl;
    exit(-1); 
}

/*
function eval_functionCall(ctx, tag) {
    if (tag.funcName == "getGlobalHash") {
        return eval_getGlobalHash(ctx, tag);
    } else if (tag.funcName == "getOldStateRoot") {
        return eval_getOldStateRoot(ctx, tag);
    } else if (tag.funcName == "getNewStateRoot") {
        return eval_getNewStateRoot(ctx, tag);
    } else if (tag.funcName == "getNTxs") {
        return eval_getNTxs(ctx, tag);
    } else if (tag.funcName == "getRawTx") {
        return eval_getRawTx(ctx, tag);
    } else if (tag.funcName == "getTxSigR") {
        return eval_getTxSigR(ctx, tag);
    } else if (tag.funcName == "getTxSigS") {
        return eval_getTxSigS(ctx, tag);
    } else if (tag.funcName == "getTxSigV") {
        return eval_getTxSigV(ctx, tag);
    } else if (tag.funcName == "getSequencerAddr") {
        return eval_getSequencerAddr(ctx, tag);
    } else if (tag.funcName == "getChainId") {
        return eval_getChainId(ctx, tag);
    } else {
        throw new Error(`function not defined ${tag.funcName}:  ${ctx.ln}`);
    }
}
*/
/*

function eval_getGlobalHash(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return bn2bna(ctx.Fr, Scalar.e(ctx.globalHash));
}

function eval_getSequencerAddr(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return bn2bna(ctx.Fr, Scalar.e(ctx.input.sequencerAddr));
}

function eval_getChainId(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.input.chainId), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getOldStateRoot(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.input.oldStateRoot), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getNewStateRoot(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.input.newStateRoot), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getNTxs(ctx, tag) {
    if (tag.params.length != 0) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    return [ctx.Fr.e(ctx.pTxs.length), ctx.Fr.zero, ctx.Fr.zero, ctx.Fr.zero];
}

function eval_getRawTx(ctx, tag) {
    if (tag.params.length != 3) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    const offset = Number(evalCommand(ctx,tag.params[1]));
    const len = Number(evalCommand(ctx,tag.params[2]));
    let d = "0x" +ctx.pTxs[txId].signData.slice(2+offset*2, 2+offset*2 + len*2);
    if (d.length == 2) d = d+'0';
    return bn2bna(ctx.Fr, Scalar.e(d));
}

function eval_getTxSigR(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    return bn2bna(ctx.Fr, Scalar.e(ctx.pTxs[txId].signature.r));
}

function eval_getTxSigS(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    return bn2bna(ctx.Fr, Scalar.e(ctx.pTxs[txId].signature.s));
}

function eval_getTxSigV(ctx, tag) {
    if (tag.params.length != 1) throw new Error(`Invalid number of parameters function ${tag.funcName}: ${ctx.ln}`)
    const txId = Number(evalCommand(ctx,tag.params[0]));
    return bn2bna(ctx.Fr, Scalar.e(ctx.pTxs[txId].signature.v));
}
*/

void preprocessTxs(Context &ctx, json &input)
{
    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("oldStateRoot") ||
         !input["oldStateRoot"].is_string() )
    {
        cerr << "Error: oldStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.oldStateRoot = input["oldStateRoot"];
    cout << "preprocessTxs(): oldStateRoot=" << ctx.oldStateRoot << endl;

    // Input JSON file must contain a oldStateRoot key at the root level
    if ( !input.contains("newStateRoot") ||
         !input["newStateRoot"].is_string() )
    {
        cerr << "Error: newStateRoot key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.newStateRoot = input["newStateRoot"];
    cout << "preprocessTxs(): newStateRoot=" << ctx.newStateRoot << endl;

    // Input JSON file must contain a sequencerAddr key at the root level
    if ( !input.contains("sequencerAddr") ||
         !input["sequencerAddr"].is_string() )
    {
        cerr << "Error: sequencerAddr key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.sequencerAddr = input["sequencerAddr"];
    cout << "preprocessTxs(): sequencerAddr=" << ctx.sequencerAddr << endl;

    // Input JSON file must contain a chainId key at the root level
    if ( !input.contains("chainId") ||
         !input["chainId"].is_number_unsigned() )
    {
        cerr << "Error: chainId key not found in input JSON file" << endl;
        exit(-1);
    }
    ctx.chainId = input["chainId"];
    cout << "preprocessTxs(): chainId=" << ctx.chainId << endl;

    // Input JSON file must contain a txs string array at the root level
    if ( !input.contains("txs") ||
         !input["txs"].is_array() )
    {
        cerr << "Error: txs key not found in input JSON file" << endl;
        exit(-1);
    }
    for (int i=0; i<input["txs"].size(); i++)
    {
        string tx = input["txs"][i];
        ctx.txs.push_back(tx);
        cout << "preprocessTxs(): tx=" << tx << endl;

    }

    // Input JSON file must contain a keys structure at the root level
    if ( !input.contains("keys") ||
         !input["keys"].is_structured() )
    {
        cerr << "Error: keys key not found in input JSON file" << endl;
        exit(-1);
    }
    cout << "keys content:" << endl;
    for (json::iterator it = input["keys"].begin(); it != input["keys"].end(); ++it)
    {
        ctx.keys[it.key()] = it.value();
        cout << "key: " << it.key() << " value: " << it.value() << endl;
    }

    // Input JSON file must contain a db structure at the root level
    if ( !input.contains("db") ||
         !input["db"].is_structured() )
    {
        cerr << "Error: db key not found in input JSON file" << endl;
        exit(-1);
    }
    cout << "db content:" << endl;
    for (json::iterator it = input["db"].begin(); it != input["db"].end(); ++it)
    {
        if (!it.value().is_array() ||
            !it.value().size()==16)
        {
            cerr << "Error: keys value not a 16-elements array in input JSON file: " << it.value() << endl;
            exit(-1);
        }
        tDbValue dbValue;
        for (int i=0; i<16; i++)
        {
            dbValue.value[i] = it.value()[i];
        }
        ctx.db[it.key()] = dbValue;
        cout << "key: " << it.key() << " value: " << dbValue.value[0] << " etc." << endl;
    }

}
/*
function preprocessTxs(ctx) {
    ctx.pTxs = [];
    const d = [];
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.oldStateRoot)), 32));
    d.push(ethers.utils.hexZeroPad(ethers.utils.hexlify(ethers.BigNumber.from(ctx.input.newStateRoot)), 32));
    for (let i=0; i<ctx.input.txs.length; i++) {
        const rtx = ethers.utils.RLP.decode(ctx.input.txs[i]);
        const chainId = (Number(rtx[6]) - 35) >> 1;
        const sign = Number(rtx[6])  & 1;
        const e =[rtx[0], rtx[1], rtx[2], rtx[3], rtx[4], rtx[5], ethers.utils.hexlify(chainId), "0x", "0x"];
        const signData = ethers.utils.RLP.encode( e );
        ctx.pTxs.push({
            signData: signData,
            signature: {
                r: rtx[7],k
                s: rtx[8],
                v: sign + 26
            }
        });
        d.push(signData);
    }
    ctx.globalHash = ethers.utils.keccak256(ctx.globalHash = ethers.utils.concat(d));
}*/

void fea2bn(Context &ctx, mpz_t &result, RawFr::Element fe0, RawFr::Element fe1, RawFr::Element fe2, RawFr::Element fe3)
{
    // Convert field elements to mpz
    mpz_t r0, r1, r2, r3;
    mpz_init_set_ui(r0,0);
    mpz_init_set_ui(r1,0);
    mpz_init_set_ui(r2,0);
    mpz_init_set_ui(r3,0);
    ctx.pFr->toMpz(r0, fe0);
    ctx.pFr->toMpz(r1, fe1);
    ctx.pFr->toMpz(r2, fe2);
    ctx.pFr->toMpz(r3, fe3);

    // Multiply by the proper power of 2, i.e. shift left
    mpz_t r1_64, r2_128, r3_192;
    mpz_init_set_ui(r1_64,0);
    mpz_init_set_ui(r2_128,0);
    mpz_init_set_ui(r3_192,0);
    mpz_mul_2exp(r1_64, r1, 64U);
    mpz_mul_2exp(r2_128, r2, 128U);
    mpz_mul_2exp(r3_192, r3, 192U);

    // Aggregate in result
    mpz_t result01, result23;
    mpz_init(result01);
    mpz_init(result23);
    mpz_add(result01, r0, r1_64);
    mpz_add(result23, r2_128, r3_192);
    mpz_add(result, result01, result23);

    // Free memory
    mpz_clear(r0);
    mpz_clear(r1);
    mpz_clear(r2);
    mpz_clear(r3); 
    mpz_clear(r1_64); 
    mpz_clear(r2_128); 
    mpz_clear(r3_192); 
    mpz_clear(result01); 
    mpz_clear(result23); 
}

/*
// Field element array to Big Number
function fea2bn(Fr, arr) {
    let res = Fr.toObject(arr[0]);
    res = Scalar.add(res, Scalar.shl(Fr.toObject(arr[1]), 64));
    res = Scalar.add(res, Scalar.shl(Fr.toObject(arr[2]), 128));
    res = Scalar.add(res, Scalar.shl(Fr.toObject(arr[3]), 192));
    return res;
}
*/
//void bn2bna(RawFr &fr, mpz_t bn, RawFr::Element &result[4])
//{
  //  ;//mfz_
//}
/*
// Big Number to field element array 
function bn2bna(Fr, bn) {
    bn = Scalar.e(bn);
    const r0 = Scalar.band(bn, Scalar.e("0xFFFFFFFFFFFFFFFF"));
    const r1 = Scalar.band(Scalar.shr(bn, 64), Scalar.e("0xFFFFFFFFFFFFFFFF"));
    const r2 = Scalar.band(Scalar.shr(bn, 128), Scalar.e("0xFFFFFFFFFFFFFFFF"));
    const r3 = Scalar.band(Scalar.shr(bn, 192), Scalar.e("0xFFFFFFFFFFFFFFFF"));
    return [Fr.e(r0), Fr.e(r1), Fr.e(r2),Fr.e(r3)];
}
*/

// Field Element to Number
int64_t fe2n(RawFr &fr, RawFr::Element &fe) {
    int64_t result;
    mpz_t maxInt;
    mpz_init_set_str(maxInt, "0x7FFFFFFF", 16);
    mpz_t minInt;
    mpz_init_set_str(minInt, "0x80000000", 16);
    mpz_t n;
    mpz_init_set_str(n, fr.toString(fe,10).c_str(), 10); // TODO: Refactor not to use strings
    if ( mpz_cmp(n,maxInt) > 0 )
    {
        mpz_t on;
        mpz_init_set_si(on,0);
        mpz_t q;
        mpz_init_set_str(q, Fr_element2str(&Fr_q), 16); // TODO: Refactor not to use strings
        //RawFr::Element prime;
        //fr.fromUI(prime, Fr_q.longVal[0]);
        //fr.toMpz(q, prime);
        mpz_sub(on, q, n);
        if ( mpz_cmp(on, minInt) > 0 )
        {
            result = -mpz_get_ui(on);
        } else {
            cerr << "Error: fe2n() Accessing a no 32bit value" << endl;
            exit(-1);
        }
        mpz_clear(q);
        mpz_clear(on);
    } else {
        result = mpz_get_ui(n);
    }
    mpz_clear(maxInt);
    mpz_clear(minInt);
    mpz_clear(n);
    return result;
}
/*
// Field Element to Number
function fe2n(Fr, fe) {
    const maxInt = Scalar.e("0x7FFFFFFF");
    const minInt = Scalar.sub(Fr.p, Scalar.e("0x80000000"));
    const o = Fr.toObject(fe);
    if (Scalar.gt(o, maxInt)) {
        const on = Scalar.sub(Fr.p, o);
        if (Scalar.gt(o, minInt)) {
            return -Scalar.toNumber(on);
        }
        throw new Error(`Accessing a no 32bit value: ${ctx.ln}`);
    } else {
        return Scalar.toNumber(o);
    }
}
*/

/*********/
/* Print */
/*********/

void printReg(Context &ctx, string name, RawFr::Element &V, bool h = false, bool bShort = false);
void printRegs(Context &ctx)
{
    cout << "Registers:" << endl;
    printReg( ctx, "A3", (*ctx.pPols)[A3][ctx.step] );
    printReg( ctx, "A2", (*ctx.pPols)[A2][ctx.step] );
    printReg( ctx, "A1", (*ctx.pPols)[A1][ctx.step] );
    printReg( ctx, "A0", (*ctx.pPols)[A0][ctx.step] );
    printReg( ctx, "B3", (*ctx.pPols)[B3][ctx.step] );
    printReg( ctx, "B2", (*ctx.pPols)[B2][ctx.step] );
    printReg( ctx, "B1", (*ctx.pPols)[B1][ctx.step] );
    printReg( ctx, "B0", (*ctx.pPols)[B0][ctx.step] );
    printReg( ctx, "C3", (*ctx.pPols)[C3][ctx.step] );
    printReg( ctx, "C2", (*ctx.pPols)[C2][ctx.step] );
    printReg( ctx, "C1", (*ctx.pPols)[C1][ctx.step] );
    printReg( ctx, "C0", (*ctx.pPols)[C0][ctx.step] );
    printReg( ctx, "D3", (*ctx.pPols)[D3][ctx.step] );
    printReg( ctx, "D2", (*ctx.pPols)[D2][ctx.step] );
    printReg( ctx, "D1", (*ctx.pPols)[D1][ctx.step] );
    printReg( ctx, "D0", (*ctx.pPols)[D0][ctx.step] );
    printReg( ctx, "E3", (*ctx.pPols)[E3][ctx.step] );
    printReg( ctx, "E2", (*ctx.pPols)[E2][ctx.step] );
    printReg( ctx, "E1", (*ctx.pPols)[E1][ctx.step] );
    printReg( ctx, "E0", (*ctx.pPols)[E0][ctx.step] );
    printReg( ctx, "SR", (*ctx.pPols)[SR][ctx.step] );
    printReg( ctx, "CTX", (*ctx.pPols)[CTX][ctx.step] );
    printReg( ctx, "SP", (*ctx.pPols)[SP][ctx.step] );
    printReg( ctx, "PC", (*ctx.pPols)[PC][ctx.step] );
    printReg( ctx, "MAXMEM", (*ctx.pPols)[MAXMEM][ctx.step] );
    printReg( ctx, "GAS", (*ctx.pPols)[GAS][ctx.step] );
    printReg( ctx, "zkPC", (*ctx.pPols)[zkPC][ctx.step] );
    RawFr::Element step;
    ctx.pFr->fromUI(step, ctx.step);
    printReg( ctx, "STEP", step, false, true );
    cout << "File: " << ctx.fileName << " Line: " << ctx.line << endl;
}

void printReg(Context &ctx, string name, RawFr::Element &V, bool h, bool bShort)
{
    cout << "    Register: " << name << " Value: " << ctx.pFr->toString(V) << endl;
}
/*

function printReg(Fr, name, V, h, short) {
    const maxInt = Scalar.e("0x7FFFFFFF");
    const minInt = Scalar.sub(Fr.p, Scalar.e("0x80000000"));

    let S;
    S = name.padEnd(6) +": ";

    let S2;
    if (!h) {
        const o = Fr.toObject(V);
        if (Scalar.gt(o, maxInt)) {
            const on = Scalar.sub(Fr.p, o);
            if (Scalar.gt(o, minInt)) {
                S2 = "-" + Scalar.toString(on);
            } else {
                S2 = "LONG";
            }
        } else {
            S2 = Scalar.toString(o);
        }
    } else {
        S2 = "";
    }

    S += S2.padStart(16, " ");
    
    if (!short) {
        const o = Fr.toObject(V);
        S+= "   " + o.toString(16).padStart(64, "0");
    }

    console.log(S);


}*/

void printVars(Context &ctx)
{
    cout << "Variables:" << endl;
    uint64_t i = 0;
    for (map<string,RawFr::Element>::iterator it=ctx.vars.begin(); it!=ctx.vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << fe2n((*ctx.pFr), it->second) << endl;
        i++;
    }
}

void printMem(Context &ctx)
{
    cout << "Memory:" << endl;
    uint64_t i = 0;
    for (map<uint64_t,RawFr::Element[4]>::iterator it=ctx.mem.begin(); it!=ctx.mem.end(); it++)
    {
        cout << "i: " << i << " address: " << it->first;
        cout << " fe[0]: " << fe2n((*ctx.pFr), it->second[0]);
        cout << " fe[1]: " << fe2n((*ctx.pFr), it->second[1]);
        cout << " fe[2]: " << fe2n((*ctx.pFr), it->second[2]);
        cout << " fe[3]: " << fe2n((*ctx.pFr), it->second[3]);
        cout << endl;
        i++;
    }
}