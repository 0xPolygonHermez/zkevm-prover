

#include <string>
#include <iostream>
#include "calculateSha256Publics.hpp"
#include "scalar.hpp"

using namespace std;

ordered_json calculateSha256(ordered_json &publics, ordered_json &prevHashInfo)
{
  
    std::string strOldStateRoot = PrependZeros(Remove0xIfPresent(publics["oldStateRoot"]), 64);
    std::string strOldAccInputHash = PrependZeros(Remove0xIfPresent(publics["oldAccInputHash"]), 64);

    mpz_t oldNumBatch;
    mpz_init(oldNumBatch);
    mpz_set_ui(oldNumBatch, publics["oldNumBatch"]);
    std::string strOldNumBatch = PrependZeros(mpz_get_str(0, 16, oldNumBatch), 16);
    
    mpz_t chainID;
    mpz_init(chainID);
    mpz_set_ui(chainID, publics["chainID"]);
    std::string strChainID = PrependZeros(mpz_get_str(0, 16, chainID), 16);

    mpz_t forkID;
    mpz_init(forkID);
    mpz_set_ui(forkID, publics["forkID"]);
    std::string strForkID = PrependZeros(mpz_get_str(0, 16, forkID), 16);

    std::string strNewStateRoot = PrependZeros(Remove0xIfPresent(publics["newStateRoot"]), 64);
    std::string strNewAccInputHash = PrependZeros(Remove0xIfPresent(publics["newAccInputHash"]), 64);
    std::string strNewLocalExitRoot = PrependZeros(Remove0xIfPresent(publics["newLocalExitRoot"]), 64);

    mpz_t newNumBatch;
    mpz_init(newNumBatch);
    mpz_set_ui(newNumBatch, publics["newNumBatch"]);
    std::string strNewNumBatch = PrependZeros(mpz_get_str(0, 16, newNumBatch), 16);

    std::string hexSha256Input = "0x" + strOldStateRoot + strOldAccInputHash + strOldNumBatch + strChainID + strForkID + strNewStateRoot + strNewAccInputHash + strNewLocalExitRoot + strNewNumBatch;

    uint64_t nPrevBlocksSha256;
    std::uint32_t prevHash[8]; 

    if(prevHashInfo != nullptr) {
        nPrevBlocksSha256 = std::stoi(to_string(prevHashInfo["nPrevBlocks"]));
        for (int i = 0; i < 8; i++)
        {
            prevHash[i] = prevHashInfo["prevHash"][i];
        }
    } else {
        nPrevBlocksSha256 = 0;
        prevHash[0] = 1779033703;
        prevHash[1] = 3144134277;
        prevHash[2] = 1013904242;
        prevHash[3] = 2773480762;
        prevHash[4] = 1359893119;
        prevHash[5] = 2600822924;
        prevHash[6] = 528734635;
        prevHash[7] = 1541459225;
    }

    uint32_t outHash[8];
    
    SHA256_PARTIAL(hexSha256Input, prevHash, outHash);

    ordered_json hashInfo = ordered_json::object();
    hashInfo["nPrevBlocks"] = nPrevBlocksSha256 + 3;
    hashInfo["prevHash"] = outHash;

    mpz_clear(oldNumBatch);
    mpz_clear(forkID);
    mpz_clear(chainID);
    mpz_clear(newNumBatch);

    return hashInfo;
}
