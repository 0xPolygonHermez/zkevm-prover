#include <iostream>
#include "utils.hpp"
#include "scalar.hpp"
#include "pols.hpp"

void printRegs (Context &ctx)
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

void printReg (Context &ctx, string name, RawFr::Element &V, bool h, bool bShort)
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

void printVars (Context &ctx)
{
    cout << "Variables:" << endl;
    uint64_t i = 0;
    for (map<string,RawFr::Element>::iterator it=ctx.vars.begin(); it!=ctx.vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << fe2n((*ctx.pFr), it->second) << endl;
        i++;
    }
}

void printMem (Context &ctx)
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