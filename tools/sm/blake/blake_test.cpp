#include <iostream>
#include "blake_test.hpp"
#include "blake.hpp"
#include "blake_gate.hpp"
#include "zkassert.hpp"
#include "timer.hpp"

using namespace std;

/***************/
/* PERFORMANCE */
/***************/

#define PERFORMANCE_TEST_LENGTH (1024*1024*10)

void PerformanceTest (void)
{
    cout << "PerformanceTest" << endl;

    uint64_t *randomValues = new uint64_t[PERFORMANCE_TEST_LENGTH];
    zkassert(randomValues);
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        randomValues[i] = rand();
    }

    uint64_t aux;

    TimerStart(ADD_OPERATION);
    aux = 0;
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        aux += randomValues[i];
    }
    TimerStopAndLog(ADD_OPERATION);

    TimerStart(XOR_OPERATION);
    aux = 0;
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        aux ^= randomValues[i];
    }
    TimerStopAndLog(XOR_OPERATION);

    TimerStart(AND_OPERATION);
    aux = 0;
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        aux &= randomValues[i];
    }
    TimerStopAndLog(AND_OPERATION);

    delete(randomValues);
}

void PerformanceTestFE (void)
{
    cout << "PerformanceTestFE" << endl;

    Goldilocks fr;
    Goldilocks::Element *randomValuesFE = new Goldilocks::Element[PERFORMANCE_TEST_LENGTH];
    zkassert(randomValuesFE);
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        randomValuesFE[i] = fr.fromU64(rand());
    }

    uint64_t *randomValues = new uint64_t[PERFORMANCE_TEST_LENGTH];
    zkassert(randomValues);
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        randomValues[i] = rand();
    }

    Goldilocks::Element aux;

    TimerStart(ADD_OPERATION);
    aux = fr.zero();
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        aux = fr.add(aux, randomValuesFE[i]);
    }
    TimerStopAndLog(ADD_OPERATION);

    TimerStart(MUL_OPERATION);
    aux = fr.zero();
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        aux = fr.mul(aux, randomValuesFE[i]);
    }
    TimerStopAndLog(MUL_OPERATION);

    TimerStart(XOR_OPERATION);
    aux = fr.zero();
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        aux = fr.fromU64( fr.toU64(aux) ^ fr.toU64(randomValuesFE[i]) );
    }
    TimerStopAndLog(XOR_OPERATION);

    TimerStart(XOR_OPERATION_2);
    aux = fr.zero();
    for (uint64_t i=0; i<PERFORMANCE_TEST_LENGTH; i++)
    {
        aux = fr.fromU64( fr.toU64(aux) ^ randomValues[i] );
    }
    TimerStopAndLog(XOR_OPERATION_2);

    delete(randomValuesFE);
    delete(randomValues);
}

/********/
/* TEST */
/********/

vector<vector<string>> blakeTestVectors = {
    {"", "0xe5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"},
    {"0x30", "0xfd923ca5e7218c4ba3c3801c26a617ecdbfdaebb9c76ce2eca166e7855efbb8"},
    {"0x3030", "0xcbc63dc2acb86bd8967453ef98fd4f2be2f26d7337a0937958211c128a18b442"},
    {"0x303030", "0x4f3b771750d60ed12c38f5f80683fb53b37e3da02dd7381454add8f1dbd2ee60"},
    {"0x30313233343536373839", "0x7b6cb8d374484e221785288b035dc53fc9ddf000607f473fc2a3258d89a70398"},
    {"0x30313233343536373839303132333435363738393031323334353637383930313233343536373839303132333435363738393031323334353637383930313233343536373839303132333435363738393031323334353637383930313233343536373839", "0xcd7d6f7e704dc181e217c07f97c61b4240483e30fc5509024f71eb236058e575"},
    };

uint64_t Blake2b256_Test (Goldilocks &fr, const Config &config)
{
    //PerformanceTest();
    //PerformanceTestFE();

    uint64_t numberOfErrors = 0;

    TimerStart(BLAKE_2B_256_TEST);
    for (uint64_t i=0; i<blakeTestVectors.size(); i++)
    {
        string input = blakeTestVectors[i][0];
        string expectedHash = blakeTestVectors[i][1];
        string hash;

        Blake2b256_String(input, hash);
        
        if (hash != expectedHash)
        {
            zklog.error("BlakeTest() 1 failed, hash of " + input + " is " + hash + " instead of " + expectedHash);
            numberOfErrors++;
        }
        else
        {
            zklog.info("Hash of \"" + input + "\" is " + hash);
        }
    }
    TimerStopAndLog(BLAKE_2B_256_TEST);

    TimerStart(BLAKE_2B_256_GATE_TEST);
    for (uint64_t i=0; i<blakeTestVectors.size(); i++)
    {
        string input = blakeTestVectors[i][0];
        string expectedHash = blakeTestVectors[i][1];
        string hash;

        Blake2b256Gate_String(input, hash);
        
        if (hash != expectedHash)
        {
            zklog.error("BlakeTest() 2 failed, hash of " + input + " is " + hash + " instead of " + expectedHash);
            numberOfErrors++;
        }
        else
        {
            zklog.info("Hash of \"" + input + "\" is " + hash);
        }
    }
    TimerStopAndLog(BLAKE_2B_256_GATE_TEST);

    zklog.info("BlakeTest() done, errors=" + to_string(numberOfErrors));
    return numberOfErrors;
}