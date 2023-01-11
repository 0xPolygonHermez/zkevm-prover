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

    if (config.saveDbReadsToFile)
    {
        dbReadLog = new DatabaseMap();
        if (config.saveDbReadsToFileOnChange)
        {
            dbReadLog->setOnChangeCallback(this, ProverRequest::onDBReadLogChangeCallback);
        }
    }
}

string ProverRequest::proofFile (void)
{
    return filePrefix + config.proofFile;
}

string ProverRequest::inputFile (void)
{
    return filePrefix + to_string(input.publicInputsExtended.publicInputs.oldBatchNum) + "." + proverRequestType2string(type) + "_input.json";
}

string ProverRequest::inputDbFile (void)
{
    return filePrefix + to_string(input.publicInputsExtended.publicInputs.oldBatchNum) + "." + proverRequestType2string(type) + "_input_db.json";
}

string ProverRequest::publicsOutputFile (void)
{
    return filePrefix + to_string(input.publicInputsExtended.publicInputs.oldBatchNum) + "." + proverRequestType2string(type) + "_" + config.publicsOutput;
}

void ProverRequest::onDBReadLogChange(DatabaseMap *dbMap)
{
    json inputJsonEx;
    input.save(inputJsonEx, *dbMap);
    json2file(inputJsonEx, inputDbFile());
}

ProverRequest::~ProverRequest()
{
    if (dbReadLog != NULL)
        delete dbReadLog;
}