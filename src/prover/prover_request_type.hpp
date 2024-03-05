#ifndef PROVER_REQUEST_TYPE_HPP
#define PROVER_REQUEST_TYPE_HPP

#include <string>

using namespace std;

typedef enum
{
    prt_none = 0,

    prt_processBatch = 1,
    prt_executeBatch = 2,
    prt_genBatchProof = 3,
    prt_genAggregatedBatchProof = 4,

    prt_processBlobInner = 5,
    prt_executeBlobInner = 6,
    prt_genBlobInnerProof = 7,

    prt_genBlobOuterProof = 8,
    prt_genAggregatedBlobOuterProof = 9,
    prt_genFinalProof = 10

} tProverRequestType;

string proverRequestType2string (tProverRequestType type);

#endif