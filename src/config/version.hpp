#ifndef ZKEVM_PROVER_VERSION_HPP
#define ZKEVM_PROVER_VERSION_HPP

#include "definitions.hpp"

#if (PROVER_FORK_ID==10)
#define ZKEVM_PROVER_VERSION "v7.0.0-RC10-fork.10"
#elif (PROVER_FORK_ID==11)
#define ZKEVM_PROVER_VERSION "v7.0.0-RC10-fork.11"
#else
#error "Invalid PROVER_FORK_ID"
#endif

#endif
