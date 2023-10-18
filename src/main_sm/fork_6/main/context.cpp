#include "main_sm/fork_6/main/context.hpp"
#include "zklog.hpp"

namespace fork_6
{

void Context::printRegs()
{
    zklog.info("Registers:");
    printReg("A", pols.A0[*pStep], pols.A1[*pStep], pols.A2[*pStep], pols.A3[*pStep], pols.A4[*pStep], pols.A5[*pStep], pols.A6[*pStep], pols.A7[*pStep]);
    printReg("B", pols.B0[*pStep], pols.B1[*pStep], pols.B2[*pStep], pols.B3[*pStep], pols.B4[*pStep], pols.B5[*pStep], pols.B6[*pStep], pols.B7[*pStep]);
    printReg("C", pols.C0[*pStep], pols.C1[*pStep], pols.C2[*pStep], pols.C3[*pStep], pols.C4[*pStep], pols.C5[*pStep], pols.C6[*pStep], pols.C7[*pStep]);
    printReg("D", pols.D0[*pStep], pols.D1[*pStep], pols.D2[*pStep], pols.D3[*pStep], pols.D4[*pStep], pols.D5[*pStep], pols.D6[*pStep], pols.D7[*pStep]);
    printReg("E", pols.E0[*pStep], pols.E1[*pStep], pols.E2[*pStep], pols.E3[*pStep], pols.E4[*pStep], pols.E5[*pStep], pols.E6[*pStep], pols.E7[*pStep]);
    printReg("SR", pols.SR0[*pStep], pols.SR1[*pStep], pols.SR2[*pStep], pols.SR3[*pStep], pols.SR4[*pStep], pols.SR5[*pStep], pols.SR6[*pStep], pols.SR7[*pStep]);
    printReg("CTX", pols.CTX[*pStep]);
    printReg("SP", pols.SP[*pStep]);
    printReg("PC", pols.PC[*pStep]);
    printReg("GAS", pols.GAS[*pStep]);
    printReg("zkPC", pols.zkPC[*pStep]);
    Goldilocks::Element step;
    step = fr.fromU64(*pStep);
    printReg("STEP", step);
#ifdef LOG_FILENAME
    zklog.info("File: " + fileName + " Line: " + to_string(line));
#endif
}

void Context::printVars()
{
    zklog.info("Variables:");
    uint64_t i = 0;
    for (unordered_map<string, mpz_class>::iterator it = vars.begin(); it != vars.end(); it++)
    {
        zklog.info("i: " + to_string(i) + " varName: " + it->first + " fe: " + it->second.get_str(16));
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
    zklog.info("Memory:");
    uint64_t i = 0;
    for (unordered_map<uint64_t, Fea>::iterator it = mem.begin(); it != mem.end(); it++)
    {
        mpz_class addr(it->first);
        zklog.info("i: " + to_string(i) + " address:" + addr.get_str(16) + " " + printFea(it->second));
        i++;
    }
}

void Context::printReg(string name, Goldilocks::Element &fe)
{
    zklog.info("    Register " + name + " = " + fr.toString(fe, 16));
}

void Context::printReg(string name, Goldilocks::Element &fe0, Goldilocks::Element &fe1, Goldilocks::Element &fe2, Goldilocks::Element &fe3, Goldilocks::Element &fe4, Goldilocks::Element &fe5, Goldilocks::Element &fe6, Goldilocks::Element &fe7)
{
    zklog.info("    Register " + name + " = " + fr.toString(fe7, 16) + ":" + fr.toString(fe6, 16) + ":" + fr.toString(fe5, 16) + ":" + fr.toString(fe4, 16) + ":" + fr.toString(fe3, 16) + ":" + fr.toString(fe2, 16) + ":" + fr.toString(fe1, 16) + ":" + fr.toString(fe0, 16));
}

void Context::printU64(string name, uint64_t v)
{
    zklog.info("    U64: " + name + ":" + to_string(v));
}

void Context::printU32(string name, uint32_t v)
{
    zklog.info("    U32: " + name + ":" + to_string(v));
}

void Context::printU16(string name, uint16_t v)
{
    zklog.info("    U16: " + name + ":" + to_string(v));
}

} // namespace