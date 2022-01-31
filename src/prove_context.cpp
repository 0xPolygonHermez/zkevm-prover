#include "prove_context.hpp"
#include "utils.hpp"
#include "config.hpp"

void ProveContext::init (const Config &config)
{
    string filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    inputFile = filePrefix + "input.json";
    inputFileEx = filePrefix + "input.json";
    publicFile = filePrefix + "public.json";
    proofFile = filePrefix + "proof.json";

    db.init(config);
}