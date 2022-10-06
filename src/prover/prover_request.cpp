#include "prover_request.hpp"
#include "utils.hpp"

void ProverRequest::init (const Config &config, bool bExecutor)
{

    uuid = getUUID();
    timestamp = getTimestamp();

    string sfile;
    if (bExecutor)
    {
        sfile = "input_executor.json";
    }
    else
    {
        sfile = "input_prover.json";
    } 
   
    filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    inputFile = filePrefix + sfile;
    inputFileEx = filePrefix + "db." + sfile;
    publicFile = filePrefix + config.publicFile;
    proofFile = filePrefix + config.proofFile;
}