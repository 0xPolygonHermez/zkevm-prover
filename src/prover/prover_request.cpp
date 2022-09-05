#include "prover_request.hpp"
#include "utils.hpp"

void ProverRequest::init (const Config &config)
{
    init (config, "");
}

void ProverRequest::init (const Config &config, string infile)
{
    uuid = getUUID();
    timestamp = getTimestamp();

    string sfile;
    
    if (infile != "") sfile=infile;
    else sfile=config.inputFile;
    
    string filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    inputFile = filePrefix + sfile;
    inputFileEx = filePrefix + sfile;
    publicFile = filePrefix + config.publicFile;
    proofFile = filePrefix + config.proofFile;
}