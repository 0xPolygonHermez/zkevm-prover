#include <iostream>
#include "prover_request_type.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

string proverRequestType2string (tProverRequestType type)
{
    switch (type)
    {
        case prt_none:                        return "none";
        case prt_processBatch:                return "process_batch";
        case prt_executeBatch:                return "execute_batch";
        case prt_processBlobInner:            return "process_blob_inner";
        case prt_executeBlobInner:            return "execute_blob_inner";
        case prt_genBatchProof:               return "gen_batch_proof";
        case prt_genAggregatedBatchProof:     return "gen_aggregated_batch_proof";
        case prt_genBlobInnerProof:           return "gen_blob_inner_proof";
        case prt_genBlobOuterProof:           return "gen_blob_outer_proof";
        case prt_genAggregatedBlobOuterProof: return "gen_aggregated_blob_outer_proof";
        case prt_genFinalProof:               return "gen_final_proof";
        default:
            zklog.error("proverRequestType2string() got invalid type=" + to_string(type));
            exitProcess();
            return "";
    }
}
