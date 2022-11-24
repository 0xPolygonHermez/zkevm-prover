#ifndef PROVER_REQUEST_TYPE_HPP
#define PROVER_REQUEST_TYPE_HPP

typedef enum
{
    prt_none = 0,
    prt_genBatchProof = 1,
    prt_genAggregatedProof = 2,
    prt_genFinalProof = 3,
    prt_processBatch = 4,
    prt_execute = 5
} tProverRequestType;

#endif