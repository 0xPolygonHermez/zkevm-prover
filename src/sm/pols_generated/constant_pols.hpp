#ifndef CONSTANT_POLS_WRAPPER_HPP
#define CONSTANT_POLS_WRAPPER_HPP

#if (PROVER_FORK_ID == 10 || PROVER_FORK_ID == 11)
    #include "main_sm/fork_10/pols_generated/constant_pols.hpp"
#else 
    #include "main_sm/fork_9/pols_generated/constant_pols.hpp"
#endif

#endif