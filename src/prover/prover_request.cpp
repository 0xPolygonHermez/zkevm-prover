#include "prover_request.hpp"
#include "utils.hpp"

void ProverRequest::init (const Config &config)
{

    uuid = getUUID();
    timestamp = getTimestamp();

    string sfile;
    size_t folder;

    folder = config.inputFile.find_last_of("/");
    if (folder != string::npos) {
        sfile = config.inputFile.substr(folder+1);
    } else {
        sfile = config.inputFile;
    }    
   
    string filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    inputFile = filePrefix + sfile;
    inputFileEx = filePrefix + sfile;
    publicFile = filePrefix + config.publicFile;
    proofFile = filePrefix + config.proofFile;
}