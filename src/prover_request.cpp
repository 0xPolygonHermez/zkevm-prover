#include "prover_request.hpp"
#include "utils.hpp"

void ProverRequest::init (const Config &config)
{
    uuid = getUUID();
    timestamp = getTimestamp();

    string filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    inputFile = filePrefix + "input.json";
    inputFileEx = filePrefix + "input.json";
    publicFile = filePrefix + "public.json";
    proofFile = filePrefix + "proof.json";

    db.init(config);
}