#ifndef PROVER_REQUEST_TYPE_HPP
#define PROVER_REQUEST_TYPE_HPP

#include <string>

using namespace std;

typedef enum
{
    prt_none = 0,

    // Executor service
    prt_processBatch = 1,
    prt_executeBatch = 2,
    prt_processBlobInner = 3,
    prt_executeBlobInner = 4,

    // Aggregator client --> batch
    prt_genBatchProof = 5,
    prt_genAggregatedBatchProof = 6,

    // Aggregator client --> blob inner
    prt_genBlobInnerProof = 7,

    // Aggregator client --> blob outer and final
    prt_genBlobOuterProof = 8,
    prt_genAggregatedBlobOuterProof = 9,
    prt_genFinalProof = 10

} tProverRequestType;

string proverRequestType2string (tProverRequestType type);

#endif