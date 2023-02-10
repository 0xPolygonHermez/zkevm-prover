#include <iostream>
#include "prover_request_type.hpp"
#include "utils.hpp"
#include "exit_process.hpp"

string proverRequestType2string (tProverRequestType type)
{
    switch (type)
    {
        case prt_none:               return "none";
        case prt_genBatchProof:      return "gen_batch_proof";
        case prt_genAggregatedProof: return "gen_aggregated_proof";
        case prt_genFinalProof:      return "gen_final_proof";
        case prt_processBatch:       return "process_batch";
        case prt_execute:            return "execute";
        default:
            cerr << "Error: proverRequestType2string() got invalid type=" << type << endl;
            exitProcess();
            return "";
    }
}