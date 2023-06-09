#ifndef CONTEXT_C_HPP_fork_5
#define CONTEXT_C_HPP_fork_5

#include "registers_c.hpp"
#include "variables_c.hpp"

namespace fork_5
{

class ContextC
{
public:
    // Registers
    RegistersC regs;

    // Global variables
    GlobalVariablesC globalVars;

    // Context variables
    unordered_map<uint64_t, ContextVariablesC> contextVars;

};

}

#endif