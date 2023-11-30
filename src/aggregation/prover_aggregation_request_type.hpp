#ifndef PROVER_AGGREGATION_REQUEST_TYPE_HPP
#define PROVER_AGGREGATION_REQUEST_TYPE_HPP

#include <string>

using namespace std;

typedef enum
{
    prt_genPrepareMultichainProof = 1,
    prt_genAggregatedMultichainProof = 2,
    prt_genFinalMultichainProof = 3,
    prt_calculateHash = 4,
} tProverAggregationRequestType;

string proverAggregationRequestType2string (tProverAggregationRequestType type);

#endif