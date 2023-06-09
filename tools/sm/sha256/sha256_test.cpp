#include "sha256_test.hpp"
#include "sha256.hpp"
#include "sha256_gate.hpp"
#include "timer.hpp"
#include "exit_process.hpp"

vector<vector<string>> sha256TestVectors = {
    {"", "0xe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    {"0x", "0xe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    {"0x00", "0x6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"},
    {"0x0000", "0x96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"},
    {"0x01", "0x4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"},
    {"0x0102030405060708090a0b0c0d0e0f", "0x36030fcc7e566294905b49a720eb45bf962209d2ee1c9b73e2b7bc7ae8830376"},
    {"0x000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f", "0xb1d1a68efaee9083e1e43459e6ef9620320ba3eff096b2a96ef77956472c0e33"},
    {"0x000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0fffffff", "0xe603b922cc427b9b171e8c6fd23fbfbcd775913b4ec9242411e1d0cb77d1ef06"},
    };

uint64_t SHA256Test (Goldilocks &fr, const Config &config)
{
    uint64_t numberOfErrors = 0;

    TimerStart(SHA256_TEST);

    for (uint64_t i=0; i<sha256TestVectors.size(); i++)
    {
        string input = sha256TestVectors[i][0];
        string expectedHash = sha256TestVectors[i][1];
        string hash;
        SHA256String(input, hash);
        if (hash != expectedHash)
        {
            zklog.error("SHA256Test() 1 failed, hash of " + input + " is " + hash + " instead of " + expectedHash);
            numberOfErrors++;
        }
    }

    TimerStopAndLog(SHA256_TEST);

    TimerStart(SHA256_GATES_TEST);

    for (uint64_t i=0; i<sha256TestVectors.size(); i++)
    {
        string input = sha256TestVectors[i][0];
        string expectedHash = sha256TestVectors[i][1];
        string hash;
        SHA256GateString(input, hash);
        if (hash != expectedHash)
        {
            zklog.error("SHA256Test() 2 failed, hash of " + input + " is " + hash + " instead of " + expectedHash);
            numberOfErrors++;
        }
    }

    TimerStopAndLog(SHA256_GATES_TEST);

    zklog.info("SHA256Test() done with errors=" + to_string(numberOfErrors));
    return numberOfErrors;
}