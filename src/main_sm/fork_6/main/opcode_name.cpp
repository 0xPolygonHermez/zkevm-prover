#include <unordered_map>
#include <cstdint>
#include "main_sm/fork_6/main/opcode_name.hpp"

using namespace std;

namespace fork_6
{

OpcodeInfo opcodeInfo[256] = {
    { 0x00, "STOP", 0 },
    { 0x01, "ADD", 3 },
    { 0x02, "MUL", 5 },
    { 0x03, "SUB", 3 },
    { 0x04, "DIV", 5 },
    { 0x05, "SDIV", 5 },
    { 0x06, "MOD", 5 },
    { 0x07, "SMOD", 5 },
    { 0x08, "ADDMOD", 8 },
    { 0x09, "MULMOD", 8 },
    { 0x0a, "EXP", 10 },
    { 0x0b, "SIGNEXTEND", 5 },
    { 0xfe, "INVALID", 0 }, // 0x0c
    { 0xfe, "INVALID", 0 }, // 0x0d
    { 0xfe, "INVALID", 0 }, // 0x0e
    { 0xfe, "INVALID", 0 }, // 0x0f

    { 0x10, "LT", 3 },
    { 0x11, "GT", 3 },
    { 0x12, "SLT", 3 },
    { 0x13, "SGT", 3 },
    { 0x14, "EQ", 3 },
    { 0x15, "ISZERO", 3 },
    { 0x16, "AND", 3 },
    { 0x17, "OR", 3 },
    { 0x18, "XOR", 3 },
    { 0x19, "NOT", 3 },
    { 0x1a, "BYTE", 3 },
    { 0x1b, "SHL", 3 },
    { 0x1c, "SHR", 3 },
    { 0x1d, "SAR", 3 },
    { 0xfe, "INVALID", 0 }, // 0x1e
    { 0xfe, "INVALID", 0 }, // 0x1f

    { 0x20, "SHA3", 30 },
    { 0xfe, "INVALID", 0 }, // 0x21
    { 0xfe, "INVALID", 0 }, // 0x22
    { 0xfe, "INVALID", 0 }, // 0x23
    { 0xfe, "INVALID", 0 }, // 0x24
    { 0xfe, "INVALID", 0 }, // 0x25
    { 0xfe, "INVALID", 0 }, // 0x26
    { 0xfe, "INVALID", 0 }, // 0x27
    { 0xfe, "INVALID", 0 }, // 0x28
    { 0xfe, "INVALID", 0 }, // 0x29
    { 0xfe, "INVALID", 0 }, // 0x2a
    { 0xfe, "INVALID", 0 }, // 0x2b
    { 0xfe, "INVALID", 0 }, // 0x2c
    { 0xfe, "INVALID", 0 }, // 0x2d
    { 0xfe, "INVALID", 0 }, // 0x2e
    { 0xfe, "INVALID", 0 }, // 0x2f

    { 0x30, "ADDRESS", 2 },
    { 0x31, "BALANCE", 100 },
    { 0x32, "ORIGIN", 2 },
    { 0x33, "CALLER", 2 },
    { 0x34, "CALLVALUE", 2 },
    { 0x35, "CALLDATALOAD", 3 },
    { 0x36, "CALLDATASIZE", 2 },
    { 0x37, "CALLDATACOPY", 3 },
    { 0x38, "CODESIZE", 2 },
    { 0x39, "CODECOPY", 3 },
    { 0x3a, "GASPRICE", 2 },
    { 0x3b, "EXTCODESIZE", 100 },
    { 0x3c, "EXTCODECOPY", 100 },
    { 0x3d, "RETURNDATASIZE", 2 },
    { 0x3e, "RETURNDATACOPY", 3 },
    { 0x3f, "EXTCODEHASH", 100 },

    { 0x40, "BLOCKHASH", 20 },
    { 0x41, "COINBASE", 2 },
    { 0x42, "TIMESTAMP", 2 },
    { 0x43, "NUMBER", 2 },
    { 0x44, "DIFFICULTY", 2 },
    { 0x45, "GASLIMIT", 2 },
    { 0x46, "CHAINID", 2 },
    { 0x47, "SELFBALANCE", 5 },
    { 0x48, "BASEFEE", 2 },
    { 0xfe, "INVALID", 0 }, // 0x49
    { 0xfe, "INVALID", 0 }, // 0x4a
    { 0xfe, "INVALID", 0 }, // 0x4b
    { 0xfe, "INVALID", 0 }, // 0x4c
    { 0xfe, "INVALID", 0 }, // 0x4d
    { 0xfe, "INVALID", 0 }, // 0x4e
    { 0xfe, "INVALID", 0 }, // 0x4f

    { 0x50, "POP", 2 },
    { 0x51, "MLOAD", 3 },
    { 0x52, "MSTORE", 3 },
    { 0x53, "MSTORE8", 3 },
    { 0x54, "SLOAD", 100 },
    { 0x55, "SSTORE", 100 },
    { 0x56, "JUMP", 8 },
    { 0x57, "JUMPI", 10 },
    { 0x58, "PC", 2 },
    { 0x59, "MSIZE", 2 },
    { 0x5a, "GAS", 2 },
    { 0x5b, "JUMPDEST", 1 },
    { 0xfe, "INVALID", 0 }, // 0x5c
    { 0xfe, "INVALID", 0 }, // 0x5d
    { 0xfe, "INVALID", 0 }, // 0x5e
    { 0x5f, "PUSH0", 3 },

    { 0x60, "PUSH1", 3 },
    { 0x61, "PUSH2", 3 },
    { 0x62, "PUSH3", 3 },
    { 0x63, "PUSH4", 3 },
    { 0x64, "PUSH5", 3 },
    { 0x65, "PUSH6", 3 },
    { 0x66, "PUSH7", 3 },
    { 0x67, "PUSH8", 3 },
    { 0x68, "PUSH9", 3 },
    { 0x69, "PUSH10", 3 },
    { 0x6a, "PUSH11", 3 },
    { 0x6b, "PUSH12", 3 },
    { 0x6c, "PUSH13", 3 },
    { 0x6d, "PUSH14", 3 },
    { 0x6e, "PUSH15", 3 },
    { 0x6f, "PUSH16", 3 },

    { 0x70, "PUSH17", 3 },
    { 0x71, "PUSH18", 3 },
    { 0x72, "PUSH19", 3 },
    { 0x73, "PUSH20", 3 },
    { 0x74, "PUSH21", 3 },
    { 0x75, "PUSH22", 3 },
    { 0x76, "PUSH23", 3 },
    { 0x77, "PUSH24", 3 },
    { 0x78, "PUSH25", 3 },
    { 0x79, "PUSH26", 3 },
    { 0x7a, "PUSH27", 3 },
    { 0x7b, "PUSH28", 3 },
    { 0x7c, "PUSH29", 3 },
    { 0x7d, "PUSH30", 3 },
    { 0x7e, "PUSH31", 3 },
    { 0x7f, "PUSH32", 3 },

    { 0x80, "DUP1", 3 },
    { 0x81, "DUP2", 3 },
    { 0x82, "DUP3", 3 },
    { 0x83, "DUP4", 3 },
    { 0x84, "DUP5", 3 },
    { 0x85, "DUP6", 3 },
    { 0x86, "DUP7", 3 },
    { 0x87, "DUP8", 3 },
    { 0x88, "DUP9", 3 },
    { 0x89, "DUP10", 3 },
    { 0x8a, "DUP11", 3 },
    { 0x8b, "DUP12", 3 },
    { 0x8c, "DUP13", 3 },
    { 0x8d, "DUP14", 3 },
    { 0x8e, "DUP15", 3 },
    { 0x8f, "DUP16", 3 },

    { 0x90, "SWAP1", 3 },
    { 0x91, "SWAP2", 3 },
    { 0x92, "SWAP3", 3 },
    { 0x93, "SWAP4", 3 },
    { 0x94, "SWAP5", 3 },
    { 0x95, "SWAP6", 3 },
    { 0x96, "SWAP7", 3 },
    { 0x97, "SWAP8", 3 },
    { 0x98, "SWAP9", 3 },
    { 0x99, "SWAP10", 3 },
    { 0x9a, "SWAP11", 3 },
    { 0x9b, "SWAP12", 3 },
    { 0x9c, "SWAP13", 3 },
    { 0x9d, "SWAP14", 3 },
    { 0x9e, "SWAP15", 3 },
    { 0x9f, "SWAP16", 3 },

    { 0xa0, "LOG0", 375 },
    { 0xa1, "LOG1", 750 },
    { 0xa2, "LOG2", 1125 },
    { 0xa3, "LOG3", 1500 },
    { 0xa4, "LOG4", 1875 },
    { 0xfe, "INVALID", 0 }, // 0xa5
    { 0xfe, "INVALID", 0 }, // 0xa6
    { 0xfe, "INVALID", 0 }, // 0xa7
    { 0xfe, "INVALID", 0 }, // 0xa8
    { 0xfe, "INVALID", 0 }, // 0xa9
    { 0xfe, "INVALID", 0 }, // 0xaa
    { 0xfe, "INVALID", 0 }, // 0xab
    { 0xfe, "INVALID", 0 }, // 0xac
    { 0xfe, "INVALID", 0 }, // 0xad
    { 0xfe, "INVALID", 0 }, // 0xae
    { 0xfe, "INVALID", 0 }, // 0xaf

    { 0xfe, "INVALID", 0 }, // 0xb0
    { 0xfe, "INVALID", 0 }, // 0xb1
    { 0xfe, "INVALID", 0 }, // 0xb2
    { 0xfe, "INVALID", 0 }, // 0xb3
    { 0xfe, "INVALID", 0 }, // 0xb4
    { 0xfe, "INVALID", 0 }, // 0xb5
    { 0xfe, "INVALID", 0 }, // 0xb6
    { 0xfe, "INVALID", 0 }, // 0xb7
    { 0xfe, "INVALID", 0 }, // 0xb8
    { 0xfe, "INVALID", 0 }, // 0xb9
    { 0xfe, "INVALID", 0 }, // 0xba
    { 0xfe, "INVALID", 0 }, // 0xbb
    { 0xfe, "INVALID", 0 }, // 0xbc
    { 0xfe, "INVALID", 0 }, // 0xbd
    { 0xfe, "INVALID", 0 }, // 0xbe
    { 0xfe, "INVALID", 0 }, // 0xbf

    { 0xfe, "INVALID", 0 }, // 0xc0
    { 0xfe, "INVALID", 0 }, // 0xc1
    { 0xfe, "INVALID", 0 }, // 0xc2
    { 0xfe, "INVALID", 0 }, // 0xc3
    { 0xfe, "INVALID", 0 }, // 0xc4
    { 0xfe, "INVALID", 0 }, // 0xc5
    { 0xfe, "INVALID", 0 }, // 0xc6
    { 0xfe, "INVALID", 0 }, // 0xc7
    { 0xfe, "INVALID", 0 }, // 0xc8
    { 0xfe, "INVALID", 0 }, // 0xc9
    { 0xfe, "INVALID", 0 }, // 0xca
    { 0xfe, "INVALID", 0 }, // 0xcb
    { 0xfe, "INVALID", 0 }, // 0xcc
    { 0xfe, "INVALID", 0 }, // 0xcd
    { 0xfe, "INVALID", 0 }, // 0xce
    { 0xfe, "INVALID", 0 }, // 0xcf

    { 0xfe, "INVALID", 0 }, // 0xd0
    { 0xfe, "INVALID", 0 }, // 0xd1
    { 0xfe, "INVALID", 0 }, // 0xd2
    { 0xfe, "INVALID", 0 }, // 0xd3
    { 0xfe, "INVALID", 0 }, // 0xd4
    { 0xfe, "INVALID", 0 }, // 0xd5
    { 0xfe, "INVALID", 0 }, // 0xd6
    { 0xfe, "INVALID", 0 }, // 0xd7
    { 0xfe, "INVALID", 0 }, // 0xd8
    { 0xfe, "INVALID", 0 }, // 0xd9
    { 0xfe, "INVALID", 0 }, // 0xda
    { 0xfe, "INVALID", 0 }, // 0xdb
    { 0xfe, "INVALID", 0 }, // 0xdc
    { 0xfe, "INVALID", 0 }, // 0xdd
    { 0xfe, "INVALID", 0 }, // 0xde
    { 0xfe, "INVALID", 0 }, // 0xdf

    { 0xfe, "INVALID", 0 }, // 0xe0
    { 0xfe, "INVALID", 0 }, // 0xe1
    { 0xfe, "INVALID", 0 }, // 0xe2
    { 0xfe, "INVALID", 0 }, // 0xe3
    { 0xfe, "INVALID", 0 }, // 0xe4
    { 0xfe, "INVALID", 0 }, // 0xe5
    { 0xfe, "INVALID", 0 }, // 0xe6
    { 0xfe, "INVALID", 0 }, // 0xe7
    { 0xfe, "INVALID", 0 }, // 0xe8
    { 0xfe, "INVALID", 0 }, // 0xe9
    { 0xfe, "INVALID", 0 }, // 0xea
    { 0xfe, "INVALID", 0 }, // 0xeb
    { 0xfe, "INVALID", 0 }, // 0xec
    { 0xfe, "INVALID", 0 }, // 0xed
    { 0xfe, "INVALID", 0 }, // 0xee
    { 0xfe, "INVALID", 0 }, // 0xef

    { 0xf0, "CREATE", 32000 },
    { 0xf1, "CALL", 100 },
    { 0xf2, "CALLCODE", 100 },
    { 0xf3, "RETURN", 0 },
    { 0xf4, "DELEGATECALL", 100 },
    { 0xf5, "CREATE2", 32000 },
    { 0xfe, "INVALID", 0 }, // 0xf6
    { 0xfe, "INVALID", 0 }, // 0xf7
    { 0xfe, "INVALID", 0 }, // 0xf8
    { 0xfe, "INVALID", 0 }, // 0xf9
    { 0xfa, "STATICCALL", 100 },
    { 0xfe, "INVALID", 0 }, // 0xfb
    { 0xfe, "INVALID", 0 }, // 0xfc
    { 0xfd, "REVERT", 0 },
    { 0xfe, "INVALID", 0 },
    { 0xff, "SENDALL", 5000 }
};

} // namespace