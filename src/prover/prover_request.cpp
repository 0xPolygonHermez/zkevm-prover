#include "prover_request.hpp"
#include "utils.hpp"

void ProverRequest::init (const Config &config)
{
    uuid = getUUID();
    timestamp = getTimestamp();

    string filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    inputFile = filePrefix + config.inputFile;
    inputFileEx = filePrefix + config.inputFile;
    publicFile = filePrefix + config.publicFile;
    proofFile = filePrefix + config.proofFile;
}