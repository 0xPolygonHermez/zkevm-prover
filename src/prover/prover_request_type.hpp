#ifndef PROVER_REQUEST_TYPE_HPP
#define PROVER_REQUEST_TYPE_HPP

typedef enum
{
    prt_none = 0,
    prt_genProof = 1,
    prt_genBatchProof = 2,
    prt_genAggregatedProof = 3,
    prt_genFinalProof = 4,
    prt_processBatch = 5
} tProverRequestType;

#endif