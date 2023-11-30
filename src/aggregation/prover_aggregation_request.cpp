#include "prover_aggregation_request.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "zklog.hpp"

ProverAggregationRequest::ProverAggregationRequest (Goldilocks &fr, const Config &config, tProverAggregationRequestType type) :
    fr(fr),
    config(config),
    startTime(0),
    endTime(0),
    type(type),
    bCompleted(false),
    bCancelling(false),
    result(ZKR_UNSPECIFIED)
{
    sem_init(&completedSem, 0, 0);
    
    uuid = getUUID();

    if (config.saveFilesInSubfolders)
    {
        string folder, file;
        getTimestampWithSlashes(timestamp, folder, file);
        string directory = config.outputPath + "/" + folder;
        ensureDirectoryExists(directory);
        filePrefix = directory + "/" + timestamp + "_" + uuid + ".";
    }
    else
    {

        timestamp = getTimestamp();
        filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    }
}

string ProverAggregationRequest::proofFile (void)
{
    return filePrefix + config.proofFile;
}


string ProverAggregationRequest::publicsOutputFile (void)
{
    return filePrefix + "." + proverAggregationRequestType2string(type) + "_" + config.publicsOutput;
}


ProverAggregationRequest::~ProverAggregationRequest() {}