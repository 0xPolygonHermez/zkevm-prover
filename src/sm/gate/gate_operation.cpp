#include <iostream>
#include "gate_operation.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

using namespace std;

// Map an operation code into a string
string gateop2string (GateOperation op)
{
    switch (op)
    {
        case gop_xor:
            return "xor";
        case gop_or:
            return "xor";
        case gop_andp:
            return "andp";
        case gop_and:
            return "and";
        default:
            zklog.error("gateop2string() found invalid op value:" + to_string(op));
            exitProcess();
            return "UNKNOWN";
    }
}