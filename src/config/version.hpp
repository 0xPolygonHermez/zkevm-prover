#ifndef ZKEVM_PROVER_VERSION_HPP
#define ZKEVM_PROVER_VERSION_HPP

#include "definitions.hpp"

#ifdef __AVX512__
#define ZKEVM_PROVER_VERSION "v9.0.0-RC1.avx512"
#else
#define ZKEVM_PROVER_VERSION "v9.0.0-RC1"
#endif

#endif
