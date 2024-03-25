#include <unistd.h>
#include "hashdb_program_test.hpp"
#include "zklog.hpp"
#include "zkresult.hpp"
#include "hashdb_factory.hpp"
#include "utils.hpp"
#include "hashdb_singleton.hpp"
#include "timer.hpp"

uint64_t HashDBProgramTest (const Config& config)
{
    TimerStart(HASHDB64_PROGRAM_TEST);

    zklog.info("HashDBProgramTest() started");
    Goldilocks fr;
    PoseidonGoldilocks poseidon;
    zkresult zkr;
    Persistence persistence = PERSISTENCE_DATABASE;
    HashDBInterface* pHashDB = HashDBClientFactory::createHashDBClient(fr, config);
    zkassertpermanent(pHashDB != NULL);
    
    const uint64_t numberOfPrograms = 1000;
    uint64_t numberOfErrors = 0;

    zklog.info("HashDBWorkflowTest() numberOfPrograms=" + to_string(numberOfPrograms));


    Goldilocks::Element key[4]={0,0,0,0};
    Goldilocks::Element keyfea[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    mpz_class value = 0;
    mpz_class keyScalar = 0;
    vector<uint8_t> program;

    for (uint64_t i=0; i<numberOfPrograms; i++)
    {
        keyScalar++;
        keyfea[1] = fr.fromU64(keyScalar.get_ui());
        poseidon.hash(key, keyfea);
        program.emplace_back((uint8_t)i);
                
        zkr = pHashDB->setProgram(key, program, persistence);
        zkassertpermanent(zkr==ZKR_SUCCESS);
    }
    
    keyScalar = 0;
    program.clear();
    for (uint64_t i=0; i<numberOfPrograms; i++)
    {
        keyScalar++;
        keyfea[1] = fr.fromU64(keyScalar.get_ui());
        poseidon.hash(key, keyfea);
        program.emplace_back(i);
                
        zkr = pHashDB->getProgram(key, program, nullptr);
        zkassertpermanent(zkr==ZKR_SUCCESS);
        zkassertpermanent(program.size() == i + 1);
        for (uint64_t j=0; j<program.size(); j++)
        {
            zkassertpermanent(program[j] == (uint8_t)j);
        }
    }

    TimerStopAndLog(HASHDB64_PROGRAM_TEST);

    return numberOfErrors;
}