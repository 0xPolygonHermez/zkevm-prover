#include "prover_request.hpp"
#include "utils.hpp"
#include "exit_process.hpp"
#include "main_sm/fork_1/main/full_tracer.hpp"
#include "main_sm/fork_2/main/full_tracer.hpp"
#include "main_sm/fork_3/main/full_tracer.hpp"
#include "main_sm/fork_4/main/full_tracer.hpp"
#include "main_sm/fork_5/main/full_tracer.hpp"
#include "main_sm/fork_6/main/full_tracer.hpp"
#include "zklog.hpp"

ProverRequest::ProverRequest (Goldilocks &fr, const Config &config, tProverRequestType type) :
    fr(fr),
    config(config),
    startTime(0),
    endTime(0),
    type(type),
    input(fr),
    flushId(0),
    lastSentFlushId(0),
    dbReadLog(NULL),
    pFullTracer(NULL),
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

    if (config.saveDbReadsToFile || config.dbMetrics)
    {
        dbReadLog = new DatabaseMap();
        dbReadLog->setSaveKeys(false);

        if (config.saveDbReadsToFile){              
            dbReadLog->setSaveKeys(true);
        }
        
        if (config.saveDbReadsToFileOnChange)
        {
            dbReadLog->setSaveKeys(true);
            dbReadLog->setOnChangeCallback(this, ProverRequest::onDBReadLogChangeCallback);
        }
    }
}

void ProverRequest::CreateFullTracer(void)
{
    if (pFullTracer != NULL)
    {
        zklog.error("ProverRequest::CreateFullTracer() called with pFullTracer != NULL");
        exitProcess();
    }
    switch (input.publicInputsExtended.publicInputs.forkID)
    {
        case 1: // fork_1
        {
            pFullTracer = new fork_1::FullTracer(fr);
            if (pFullTracer == NULL)
            {
                zklog.error("ProverRequest::CreateFullTracer() failed calling new fork_1::FullTracer()");
                exitProcess();
            }
            result = ZKR_SUCCESS;
            return;
        }
        case 2: // fork_2
        {
            pFullTracer = new fork_2::FullTracer(fr);
            if (pFullTracer == NULL)
            {
                zklog.error("ProverRequest::CreateFullTracer() failed calling new fork_2::FullTracer()");
                exitProcess();
            }
            result = ZKR_SUCCESS;
            return;
        }
        case 3: // fork_3
        {
            pFullTracer = new fork_3::FullTracer(fr);
            if (pFullTracer == NULL)
            {
                zklog.error("ProverRequest::CreateFullTracer() failed calling new fork_3::FullTracer()");
                exitProcess();
            }
            result = ZKR_SUCCESS;
            return;
        }
        case 4: // fork_4
        {
            pFullTracer = new fork_4::FullTracer(fr);
            if (pFullTracer == NULL)
            {
                zklog.error("ProverRequest::CreateFullTracer() failed calling new fork_4::FullTracer()");
                exitProcess();
            }
            result = ZKR_SUCCESS;
            return;
        }
        case 5: // fork_5
        {
            pFullTracer = new fork_5::FullTracer(fr);
            if (pFullTracer == NULL)
            {
                zklog.error("ProverRequest::CreateFullTracer() failed calling new fork_5::FullTracer()");
                exitProcess();
            }
            result = ZKR_SUCCESS;
            return;
        }
        case 6: // fork_6
        {
            pFullTracer = new fork_6::FullTracer(fr);
            if (pFullTracer == NULL)
            {
                zklog.error("ProverRequest::CreateFullTracer() failed calling new fork_6::FullTracer()");
                exitProcess();
            }
            result = ZKR_SUCCESS;
            return;
        }
        default:
        {
            zklog.error("ProverRequest::CreateFullTracer() failed calling invalid fork ID=" + to_string(input.publicInputsExtended.publicInputs.forkID));
            result = ZKR_SM_MAIN_INVALID_FORK_ID;
            return;
        }
    }
}

void ProverRequest::DestroyFullTracer(void)
{
    if (pFullTracer == NULL)
    {
        zklog.error("ProverRequest::DestroyFullTracer() called with pFullTracer == NULL");
        exitProcess();
    }
    switch (input.publicInputsExtended.publicInputs.forkID)
    {
        case 1: // fork_1
        {
            delete pFullTracer;
            pFullTracer = NULL; 
            break;
        }
        case 2: // fork_2
        {
            delete pFullTracer;
            pFullTracer = NULL; 
            break;
        }
        case 3: // fork_3
        {
            delete pFullTracer;
            pFullTracer = NULL; 
            break;
        }
        case 4: // fork_4
        {
            delete pFullTracer;
            pFullTracer = NULL; 
            break;
        }
        case 5: // fork_5
        {
            delete pFullTracer;
            pFullTracer = NULL; 
            break;
        }
        case 6: // fork_6
        {
            delete pFullTracer;
            pFullTracer = NULL; 
            break;
        }
        default:
        {
            zklog.error("ProverRequest::DestroyFullTracer() failed calling invalid fork ID=" + to_string(input.publicInputsExtended.publicInputs.forkID));
            result = ZKR_SM_MAIN_INVALID_FORK_ID;
            return;
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
    {
        delete dbReadLog;
    }

    if (pFullTracer != NULL)
    {
        DestroyFullTracer();
    }
}