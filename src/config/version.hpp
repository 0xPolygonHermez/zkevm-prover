#ifndef ZKEVM_PROVER_VERSION_HPP
#define ZKEVM_PROVER_VERSION_HPP

#include "definitions.hpp"

#ifdef __AVX512__
#define ZKEVM_PROVER_VERSION "v8.0.0-RC10.avx512"
#else
#define ZKEVM_PROVER_VERSION "v8.0.0-RC10"
#endif

#endif
