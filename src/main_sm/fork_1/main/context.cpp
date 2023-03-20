#include "main_sm/fork_1/main/context.hpp"

namespace fork_1
{

void Context::printRegs()
{
    cout << "Registers:" << endl;
    printReg("A7", pols.A7[*pStep]);
    printReg("A6", pols.A6[*pStep]);
    printReg("A5", pols.A5[*pStep]);
    printReg("A4", pols.A4[*pStep]);
    printReg("A3", pols.A3[*pStep]);
    printReg("A2", pols.A2[*pStep]);
    printReg("A1", pols.A1[*pStep]);
    printReg("A0", pols.A0[*pStep]);
    printReg("B7", pols.B7[*pStep]);
    printReg("B6", pols.B6[*pStep]);
    printReg("B5", pols.B5[*pStep]);
    printReg("B4", pols.B4[*pStep]);
    printReg("B3", pols.B3[*pStep]);
    printReg("B2", pols.B2[*pStep]);
    printReg("B1", pols.B1[*pStep]);
    printReg("B0", pols.B0[*pStep]);
    printReg("C7", pols.C7[*pStep]);
    printReg("C6", pols.C6[*pStep]);
    printReg("C5", pols.C5[*pStep]);
    printReg("C4", pols.C4[*pStep]);
    printReg("C3", pols.C3[*pStep]);
    printReg("C2", pols.C2[*pStep]);
    printReg("C1", pols.C1[*pStep]);
    printReg("C0", pols.C0[*pStep]);
    printReg("D7", pols.D7[*pStep]);
    printReg("D6", pols.D6[*pStep]);
    printReg("D5", pols.D5[*pStep]);
    printReg("D4", pols.D4[*pStep]);
    printReg("D3", pols.D3[*pStep]);
    printReg("D2", pols.D2[*pStep]);
    printReg("D1", pols.D1[*pStep]);
    printReg("D0", pols.D0[*pStep]);
    printReg("E7", pols.E7[*pStep]);
    printReg("E6", pols.E6[*pStep]);
    printReg("E5", pols.E5[*pStep]);
    printReg("E4", pols.E4[*pStep]);
    printReg("E3", pols.E3[*pStep]);
    printReg("E2", pols.E2[*pStep]);
    printReg("E1", pols.E1[*pStep]);
    printReg("E0", pols.E0[*pStep]);
    printReg("SR7", pols.SR7[*pStep]);
    printReg("SR6", pols.SR6[*pStep]);
    printReg("SR5", pols.SR5[*pStep]);
    printReg("SR4", pols.SR4[*pStep]);
    printReg("SR3", pols.SR3[*pStep]);
    printReg("SR2", pols.SR2[*pStep]);
    printReg("SR1", pols.SR1[*pStep]);
    printReg("SR0", pols.SR0[*pStep]);
    printReg("CTX", pols.CTX[*pStep]);
    printReg("SP", pols.SP[*pStep]);
    printReg("PC", pols.PC[*pStep]);
    printReg("GAS", pols.GAS[*pStep]);
    printReg("zkPC", pols.zkPC[*pStep]);
    Goldilocks::Element step;
    step = fr.fromU64(*pStep);
    printReg("STEP", step, false, true);
#ifdef LOG_FILENAME
    cout << "File: " << fileName << " Line: " << line << endl;
#endif
}

void Context::printVars()
{
    cout << "Variables:" << endl;
    uint64_t i = 0;
    for (unordered_map<string, mpz_class>::iterator it = vars.begin(); it != vars.end(); it++)
    {
        cout << "i: " << i << " varName: " << it->first << " fe: " << it->second.get_str(16) << endl;
        i++;
    }
}

string Context::printFea(Fea &fea)
{
    return fr.toString(fea.fe7, 16) +
           ":" + fr.toString(fea.fe6, 16) +
           ":" + fr.toString(fea.fe5, 16) +
           ":" + fr.toString(fea.fe4, 16) +
           ":" + fr.toString(fea.fe3, 16) +
           ":" + fr.toString(fea.fe2, 16) +
           ":" + fr.toString(fea.fe1, 16) +
           ":" + fr.toString(fea.fe0, 16);
}

void Context::printMem()
{
    cout << "Memory:" << endl;
    uint64_t i = 0;
    for (unordered_map<uint64_t, Fea>::iterator it = mem.begin(); it != mem.end(); it++)
    {
        mpz_class addr(it->first);
        cout << "i: " << i << " address:" << addr.get_str(16) << " ";
        cout << printFea(it->second);
        cout << endl;
        i++;
    }
}

void Context::printReg(string name, Goldilocks::Element &fe, bool h, bool bShort)
{
    cout << "    Register: " << name << " Value: " << fr.toString(fe, 16) << endl;
}

void Context::printU64(string name, uint64_t v)
{
    cout << "    U64: " << name << ":" << v << endl;
}

void Context::printU32(string name, uint32_t v)
{
    cout << "    U32: " << name << ":" << v << endl;
}

void Context::printU16(string name, uint16_t v)
{
    cout << "    U16: " << name << ":" << v << endl;
}

} // namespace