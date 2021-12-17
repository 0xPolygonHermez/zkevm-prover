#include <iostream>
#include "utils.hpp"
#include "scalar.hpp"
#include "pols.hpp"

void printRegs (Context &ctx)
{
    cout << "Registers:" << endl;
    printU64( ctx, "A3", pol(A3)[ctx.step] );
    printU64( ctx, "A2", pol(A2)[ctx.step] );
    printU64( ctx, "A1", pol(A1)[ctx.step] );
    printReg( ctx, "A0", pol(A0)[ctx.step] );
    printU64( ctx, "B3", pol(B3)[ctx.step] );
    printU64( ctx, "B2", pol(B2)[ctx.step] );
    printU64( ctx, "B1", pol(B1)[ctx.step] );
    printReg( ctx, "B0", pol(B0)[ctx.step] );
    printU64( ctx, "C3", pol(C3)[ctx.step] );
    printU64( ctx, "C2", pol(C2)[ctx.step] );
    printU64( ctx, "C1", pol(C1)[ctx.step] );
    printReg( ctx, "C0", pol(C0)[ctx.step] );
    printU64( ctx, "D3", pol(D3)[ctx.step] );
    printU64( ctx, "D2", pol(D2)[ctx.step] );
    printU64( ctx, "D1", pol(D1)[ctx.step] );
    printReg( ctx, "D0", pol(D0)[ctx.step] );
    printU64( ctx, "E3", pol(E3)[ctx.step] );
    printU64( ctx, "E2", pol(E2)[ctx.step] );
    printU64( ctx, "E1", pol(E1)[ctx.step] );
    printReg( ctx, "E0", pol(E0)[ctx.step] );
    printReg( ctx, "SR", pol(SR)[ctx.step] );
    printU32( ctx, "CTX", pol(CTX)[ctx.step] );
    printU16( ctx, "SP", pol(SP)[ctx.step] );
    printU32( ctx, "PC", pol(PC)[ctx.step] );
    printU32( ctx, "MAXMEM", pol(MAXMEM)[ctx.step] );
    printU64( ctx, "GAS", pol(GAS)[ctx.step] );
    printU32( ctx, "zkPC", pol(zkPC)[ctx.step] );
    RawFr::Element step;
    ctx.fr.fromUI(step, ctx.step);
    printReg( ctx, "STEP", step, false, true );
#ifdef LOG_FILENAME
    cout << "File: " << ctx.fileName << " Line: " << ctx.line << endl;
#endif
}

void printVars (Context &ctx)
{
    cout << "Variables:" << endl;
    uint64_t i = 0;
    for (map<string,RawFr::Element>::iterator it=ctx.vars.begin(); it!=ctx.vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << fe2n(ctx.fr, ctx.prime, it->second) << endl;
        i++;
    }
}

string printFea (Context &ctx, Fea &fea)
{
    return "fe0:" + ctx.fr.toString(fea.fe0, 16) + 
           " fe1:" + ctx.fr.toString(fea.fe1, 16) +
           " fe2:" + ctx.fr.toString(fea.fe2, 16) +
           " fe3:" + ctx.fr.toString(fea.fe3, 16);
}

void printMem (Context &ctx)
{
    cout << "Memory:" << endl;
    uint64_t i = 0;
    for (map<uint64_t,Fea>::iterator it=ctx.mem.begin(); it!=ctx.mem.end(); it++)
    {
        mpz_class addr(it->first);
        cout << "i: " << i << " address:" << addr.get_str(16) << " ";
        cout << printFea(ctx, it->second);
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
        cout << "Storage: " << i << " fe: " << ctx.fr.toString(fe, 16) << " scalar: " << scalar.get_str(16) << endl;
    } 
}

void printReg (Context &ctx, string name, RawFr::Element &fe, bool h, bool bShort)
{
    cout << "    Register: " << name << " Value: " << ctx.fr.toString(fe, 16) << endl;
}

void printDb (Context &ctx)
{
    printDb(ctx.fr, ctx.db);
}

void printDb (RawFr &fr, map< RawFr::Element, vector<RawFr::Element>, CompareFe > &db)
{
    cout << "Database of " << db.size() << " elements:" << endl;
    for ( map< RawFr::Element, vector<RawFr::Element>, CompareFe >::iterator it = db.begin(); it!=db.end(); it++)
    {
        RawFr::Element fe = it->first;
        vector<RawFr::Element> vect = it->second;
        cout << "key:" << fr.toString(fe, 16);
        for (int i=0; i<vect.size(); i++)
            cout << " " << i << ":" << fr.toString(vect[i], 16);
        cout << endl;
    }
}

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
    cout << "    U16: " << name << ":" << v << endl;
}

uint64_t TimeDiff (const struct timeval &startTime, const struct timeval &endTime )
{
    struct timeval diff;

    // Calculate the time difference
    diff.tv_sec = endTime.tv_sec - startTime.tv_sec;
    if (endTime.tv_usec >= startTime.tv_usec)
        diff.tv_usec = endTime.tv_usec - startTime.tv_usec;
    else if (diff.tv_sec>0)
    {
        diff.tv_usec = 1000000 + endTime.tv_usec - startTime.tv_usec;
        diff.tv_sec--;
    }
    else{
        cerr << "Error: TimeDiff() got startTime > endTime" << endl;
        exit(-1);
    }

    // Return the total number of us
    return diff.tv_usec + 1000000*diff.tv_sec;
}

uint64_t TimeDiff (const struct timeval &startTime )
{
    struct timeval endTime;
    gettimeofday(&endTime, NULL);
    return TimeDiff(startTime, endTime);
}