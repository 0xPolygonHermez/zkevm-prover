#include <iostream>
#include "prover_aggregation_request_type.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

string proverAggregationRequestType2string (tProverAggregationRequestType type)
{
    switch (type)
    {
        case prt_genPrepareMultichainProof:        return "gen_prepare_multichain_proof";
        case prt_genAggregatedMultichainProof:     return "gen_aggregated_multichain_proof";
        case prt_genFinalMultichainProof:          return "gen_final_multichain_proof";
        case prt_calculateHash:                    return "calculate_hash";
        default:
            zklog.error("proverAggregationRequestType2string() got invalid type=" + to_string(type));
            exitProcess();
            return "";
    }
}