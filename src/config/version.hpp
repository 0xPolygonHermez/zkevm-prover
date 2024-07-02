#ifndef ZKEVM_PROVER_VERSION_HPP
#define ZKEVM_PROVER_VERSION_HPP

#include "definitions.hpp"

#if (PROVER_FORK_ID==10)

#ifdef __AVX512__
#define ZKEVM_PROVER_VERSION "v7.0.0-RC13-fork.10.avx512"
#else
#define ZKEVM_PROVER_VERSION "v7.0.0-RC13-fork.10"
#endif
#elif (PROVER_FORK_ID==11)
#ifdef __AVX512__
#define ZKEVM_PROVER_VERSION "v7.0.0-RC13-fork.11.avx512"
#else
#define ZKEVM_PROVER_VERSION "v7.0.0-RC13-fork.11"
#endif

#else
#error "Invalid PROVER_FORK_ID"
#endif

#endif
