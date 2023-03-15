#include "main_sm/fork_3/main/eth_opcodes.hpp"

namespace fork_3
{

string ethOpcode[256];

void ethOpcodeInit (void)
{
    ethOpcode[0x00] = "opSTOP";
    ethOpcode[0x01] = "opADD";
    ethOpcode[0x02] = "opMUL";
    ethOpcode[0x03] = "opSUB";
    ethOpcode[0x04] = "opDIV";
    ethOpcode[0x05] = "opSDIV";
    ethOpcode[0x06] = "opMOD";
    ethOpcode[0x07] = "opSMOD";
    ethOpcode[0x08] = "opADDMOD";
    ethOpcode[0x09] = "opMULMOD";
    ethOpcode[0x0a] = "opEXP";
    ethOpcode[0x0b] = "opSIGNEXTEND";
    ethOpcode[0x10] = "opLT";
    ethOpcode[0x11] = "opGT";
    ethOpcode[0x12] = "opSLT";
    ethOpcode[0x13] = "opSGT";
    ethOpcode[0x14] = "opEQ";
    ethOpcode[0x15] = "opISZERO";
    
    ethOpcode[0x16] = "opAND";
    ethOpcode[0x17] = "opOR";
    ethOpcode[0x18] = "opXOR";
    ethOpcode[0x19] = "opNOT";
    ethOpcode[0x1a] = "opBYTE";
    ethOpcode[0x1b] = "opSHL";
    ethOpcode[0x1c] = "opSHR";
    ethOpcode[0x1d] = "opSAR";

    ethOpcode[0x20] = "opKECCAK256";

    ethOpcode[0x30] = "opADDRESS";
    ethOpcode[0x31] = "opBALANCE";
    ethOpcode[0x32] = "opORIGIN";
    ethOpcode[0x33] = "opCALLER";
    ethOpcode[0x34] = "opCALLVALUE";
    ethOpcode[0x35] = "opCALLDATALOAD";
    ethOpcode[0x36] = "opCALLDATASIZE";
    ethOpcode[0x37] = "opCALLDATACOPY";
    ethOpcode[0x38] = "opCODESIZE";
    ethOpcode[0x39] = "opCODECOPY";
    ethOpcode[0x3a] = "opGASPRICE";
    ethOpcode[0x3b] = "opEXTCODESIZE";
    ethOpcode[0x3c] = "opEXTCODECOPY";
    ethOpcode[0x3d] = "opRETURNDATASIZE";
    ethOpcode[0x3e] = "opRETURNDATACOPY";
    ethOpcode[0x3f] = "opEXTCODEHASH";

    ethOpcode[0x40] = "opBLOCKHASH";
    ethOpcode[0x41] = "opCOINBASE";
    ethOpcode[0x42] = "opTIMESTAMP";
    ethOpcode[0x43] = "opNUMBER";
    ethOpcode[0x44] = "opDIFFICULT";
    ethOpcode[0x45] = "opGASLIMIT";
    ethOpcode[0x46] = "opCHAINID";
    ethOpcode[0x47] = "opSELFBALANCE";

    ethOpcode[0x50] = "opPOP";
    ethOpcode[0x51] = "opMLOAD";
    ethOpcode[0x52] = "opMSTORE";
    ethOpcode[0x53] = "opMSTORE8";
    ethOpcode[0x54] = "opSLOAD";
    ethOpcode[0x55] = "opSSTORE";
    ethOpcode[0x56] = "opJUMP";
    ethOpcode[0x57] = "opJUMPI";
    ethOpcode[0x58] = "opPC";
    ethOpcode[0x59] = "opMSIZE";
    ethOpcode[0x5a] = "opGAS";
    ethOpcode[0x5b] = "opJUMPDEST";

    ethOpcode[0x60] = "opPUSH1";
    ethOpcode[0x61] = "opPUSH2";
    ethOpcode[0x62] = "opPUSH3";
    ethOpcode[0x63] = "opPUSH4";
    ethOpcode[0x64] = "opPUSH5";
    ethOpcode[0x65] = "opPUSH6";
    ethOpcode[0x66] = "opPUSH7";
    ethOpcode[0x67] = "opPUSH8";
    ethOpcode[0x68] = "opPUSH9";
    ethOpcode[0x69] = "opPUSH10";
    ethOpcode[0x6a] = "opPUSH11";
    ethOpcode[0x6b] = "opPUSH12";
    ethOpcode[0x6c] = "opPUSH13";
    ethOpcode[0x6d] = "opPUSH14";
    ethOpcode[0x6e] = "opPUSH15";
    ethOpcode[0x6f] = "opPUSH16";
    ethOpcode[0x70] = "opPUSH17";
    ethOpcode[0x71] = "opPUSH18";
    ethOpcode[0x72] = "opPUSH19";
    ethOpcode[0x73] = "opPUSH20";
    ethOpcode[0x74] = "opPUSH21";
    ethOpcode[0x75] = "opPUSH22";
    ethOpcode[0x76] = "opPUSH23";
    ethOpcode[0x77] = "opPUSH24";
    ethOpcode[0x78] = "opPUSH25";
    ethOpcode[0x79] = "opPUSH26";
    ethOpcode[0x7a] = "opPUSH27";
    ethOpcode[0x7b] = "opPUSH28";
    ethOpcode[0x7c] = "opPUSH29";
    ethOpcode[0x7d] = "opPUSH30";
    ethOpcode[0x7e] = "opPUSH31";
    ethOpcode[0x7f] = "opPUSH32";

    ethOpcode[0x80] = "opDUP1";
    ethOpcode[0x81] = "opDUP2";
    ethOpcode[0x82] = "opDUP3";
    ethOpcode[0x83] = "opDUP4";
    ethOpcode[0x84] = "opDUP5";
    ethOpcode[0x85] = "opDUP6";
    ethOpcode[0x86] = "opDUP7";
    ethOpcode[0x87] = "opDUP8";
    ethOpcode[0x88] = "opDUP9";
    ethOpcode[0x89] = "opDUP10";
    ethOpcode[0x8a] = "opDUP11";
    ethOpcode[0x8b] = "opDUP12";
    ethOpcode[0x8c] = "opDUP13";
    ethOpcode[0x8d] = "opDUP14";
    ethOpcode[0x8e] = "opDUP15";
    ethOpcode[0x8f] = "opDUP16";

    ethOpcode[0x90] = "opSWAP1";
    ethOpcode[0x91] = "opSWAP2";
    ethOpcode[0x92] = "opSWAP3";
    ethOpcode[0x93] = "opSWAP4";
    ethOpcode[0x94] = "opSWAP5";
    ethOpcode[0x95] = "opSWAP6";
    ethOpcode[0x96] = "opSWAP7";
    ethOpcode[0x97] = "opSWAP8";
    ethOpcode[0x98] = "opSWAP9";
    ethOpcode[0x99] = "opSWAP10";
    ethOpcode[0x9a] = "opSWAP11";
    ethOpcode[0x9b] = "opSWAP12";
    ethOpcode[0x9c] = "opSWAP13";
    ethOpcode[0x9d] = "opSWAP14";
    ethOpcode[0x9e] = "opSWAP15";
    ethOpcode[0x9f] = "opSWAP16";

    ethOpcode[0xa0] = "opLOG0";
    ethOpcode[0xa1] = "opLOG1";
    ethOpcode[0xa2] = "opLOG2";
    ethOpcode[0xa3] = "opLOG3";
    ethOpcode[0xa4] = "opLOG4";

    ethOpcode[0xf0] = "opCREATE";
    ethOpcode[0xf1] = "opCALL";
    ethOpcode[0xf2] = "opCALLCODE";
    ethOpcode[0xf3] = "opRETURN";
    ethOpcode[0xf4] = "opDELEGATECALL";
    ethOpcode[0xf5] = "opCREATE2";
    ethOpcode[0xfa] = "opSTATICCALL";
    ethOpcode[0xfd] = "opREVERT";
    ethOpcode[0xfe] = "opINVALID";
    ethOpcode[0xff] = "opSELFDESTRUCT";
};

} // namespace