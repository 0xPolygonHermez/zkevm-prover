#include <iostream>
#include "utils.hpp"
#include "scalar.hpp"
#include "pols.hpp"

void printRegs (Context &ctx)
{
    cout << "Registers:" << endl;
    printU64( ctx, "A3", pols(A3)[ctx.step] );
    printU64( ctx, "A2", pols(A2)[ctx.step] );
    printU64( ctx, "A1", pols(A1)[ctx.step] );
    printReg( ctx, "A0", pols(A0)[ctx.step] );
    printU64( ctx, "B3", pols(B3)[ctx.step] );
    printU64( ctx, "B2", pols(B2)[ctx.step] );
    printU64( ctx, "B1", pols(B1)[ctx.step] );
    printReg( ctx, "B0", pols(B0)[ctx.step] );
    printU64( ctx, "C3", pols(C3)[ctx.step] );
    printU64( ctx, "C2", pols(C2)[ctx.step] );
    printU64( ctx, "C1", pols(C1)[ctx.step] );
    printReg( ctx, "C0", pols(C0)[ctx.step] );
    printU64( ctx, "D3", pols(D3)[ctx.step] );
    printU64( ctx, "D2", pols(D2)[ctx.step] );
    printU64( ctx, "D1", pols(D1)[ctx.step] );
    printReg( ctx, "D0", pols(D0)[ctx.step] );
    printU64( ctx, "E3", pols(E3)[ctx.step] );
    printU64( ctx, "E2", pols(E2)[ctx.step] );
    printU64( ctx, "E1", pols(E1)[ctx.step] );
    printReg( ctx, "E0", pols(E0)[ctx.step] );
    printReg( ctx, "SR", pols(SR)[ctx.step] );
    printU32( ctx, "CTX", pols(CTX)[ctx.step] );
    printU16( ctx, "SP", pols(SP)[ctx.step] );
    printU32( ctx, "PC", pols(PC)[ctx.step] );
    printU32( ctx, "MAXMEM", pols(MAXMEM)[ctx.step] );
    printU64( ctx, "GAS", pols(GAS)[ctx.step] );
    printU32( ctx, "zkPC", pols(zkPC)[ctx.step] );
    RawFr::Element step;
    ctx.fr.fromUI(step, ctx.step);
    printReg( ctx, "STEP", step, false, true );
    cout << "File: " << ctx.fileName << " Line: " << ctx.line << endl;
}

void printVars (Context &ctx)
{
    cout << "Variables:" << endl;
    uint64_t i = 0;
    for (map<string,RawFr::Element>::iterator it=ctx.vars.begin(); it!=ctx.vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << fe2n(ctx, it->second) << endl;
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
        cout << " fe[0]: " << fe2n(ctx, it->second[0]);
        cout << " fe[1]: " << fe2n(ctx, it->second[1]);
        cout << " fe[2]: " << fe2n(ctx, it->second[2]);
        cout << " fe[3]: " << fe2n(ctx, it->second[3]);
        cout << endl;
        i++;
    }
}

void printStorage (Context &ctx)
{
    uint64_t i = 0;
    for (map< RawFr::Element, mpz_class, CompareFe>::iterator it=ctx.sto.begin(); it!=ctx.sto.end(); it++)
    {
        RawFr::Element fe = it->first;
        mpz_class scalar = it->second;
        cout << "Storage: " << i << " fe: " << ctx.fr.toString(fe) << " scalar: " << scalar.get_str() << endl;
    } 
}

void printReg (Context &ctx, string name, RawFr::Element &V, bool h, bool bShort)
{
    cout << "    Register: " << name << " Value: " << ctx.fr.toString(V) << endl;
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

void printU64 (Context &ctx, string name, uint64_t v)
{
    cout << "    U64: " << name << ":" << v << endl;
}

void printU32 (Context &ctx, string name, uint32_t v)
{
    cout << "    U32: " << name << ":" << v << endl;
}

void printU16 (Context &ctx, string name, uint16_t v)
{
    cout <<  "    U16: " << name << ":" << v << endl;
}