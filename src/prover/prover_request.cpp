#include "prover_request.hpp"
#include "utils.hpp"
#include "exit_process.hpp"

ProverRequest::ProverRequest (Goldilocks &fr, const Config &config, tProverRequestType type) :
    fr(fr),
    config(config),
    startTime(0),
    endTime(0),
    type(type),
    input(fr),
    dbReadLog(NULL),
    fullTracer(fr),
    bCompleted(false),
    bCancelling(false),
    result(ZKR_UNSPECIFIED)
{
    sem_init(&completedSem, 0, 0);
    
    uuid = getUUID();
    timestamp = getTimestamp();

    filePrefix = config.outputPath + "/" + timestamp + "_" + uuid + ".";
    proofFile = filePrefix + config.proofFile;

    switch (type)
    {
        case prt_genBatchProof:
        {
            inputFile = filePrefix + "gen_batch_proof_input.json";
            inputFileEx = filePrefix + "gen_batch_proof_input_db.json";
            publicsOutput = filePrefix + "batch_proof." + config.publicsOutput;
            break;
        }
        case prt_genAggregatedProof:
        {
            inputFile = filePrefix + "gen_aggregated_proof_input.json";
            inputFileEx = filePrefix + "gen_aggregated_proof_input_db.json";
            publicsOutput = filePrefix + "aggregated_proof." + config.publicsOutput;
            break;
        }
        case prt_genFinalProof:
        {
            inputFile = filePrefix + "gen_final_proof_input.json";
            inputFileEx = filePrefix + "gen_final_proof_input_db.json";
            publicsOutput = filePrefix + "final_proof." + config.publicsOutput;
            break;
        }
        case prt_processBatch:
        {
            inputFile = filePrefix + "process_batch_input.json";
            inputFileEx = filePrefix + "process_batch_input_db.json";
            publicsOutput = filePrefix + "process_batch_" + config.publicsOutput;
            break;
        }
        default:
        {
            cerr << "ProverRequest::ProverRequest() got invalid type=" << type << endl;
            exitProcess();
        }
    }

    if (config.saveDbReadsToFile) {
        dbReadLog = new DatabaseMap();
        if (config.saveDbReadsToFileOnChange)
            dbReadLog->setOnChangeCallback(this, ProverRequest::onDBReadLogChangeCallback);
    }
}

void ProverRequest::onDBReadLogChange(DatabaseMap *dbMap)
{
    json inputJsonEx;
    input.save(inputJsonEx, *dbMap);
    json2file(inputJsonEx, inputFileEx);
}

ProverRequest::~ProverRequest()
{
    if (dbReadLog != NULL)
        delete dbReadLog;
}