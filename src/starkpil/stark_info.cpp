#include "stark_info.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

StarkInfo::StarkInfo(string file)
{

    // Load contents from json file
    TimerStart(STARK_INFO_LOAD);
    json starkInfoJson;
    file2json(file, starkInfoJson);
    load(starkInfoJson);
    TimerStopAndLog(STARK_INFO_LOAD);
}

void StarkInfo::load(json j)
{   
    starkStruct.nBits = j["starkStruct"]["nBits"];
    starkStruct.nBitsExt = j["starkStruct"]["nBitsExt"];
    starkStruct.nQueries = j["starkStruct"]["nQueries"];
    starkStruct.verificationHashType = j["starkStruct"]["verificationHashType"];
    if(starkStruct.verificationHashType == "BN128") {
        if(j["starkStruct"].contains("merkleTreeArity")) {
            starkStruct.merkleTreeArity = j["starkStruct"]["merkleTreeArity"];
        } else {
            starkStruct.merkleTreeArity = 16;
        }
        if(j["starkStruct"].contains("merkleTreeCustom")) {
            starkStruct.merkleTreeCustom = j["starkStruct"]["merkleTreeCustom"];
        } else {
            starkStruct.merkleTreeCustom = false;
        }
    }
    if(j["starkStruct"].contains("hashCommits")) {
        starkStruct.hashCommits = j["starkStruct"]["hashCommits"];
    } else {
        starkStruct.hashCommits = false;
    }

    for (uint64_t i = 0; i < j["starkStruct"]["steps"].size(); i++)
    {
        StepStruct step;
        step.nBits = j["starkStruct"]["steps"][i]["nBits"];
        starkStruct.steps.push_back(step);
    }

    nPublics = j["nPublics"];
    nConstants = j["nConstants"];

    if(j.contains("nSubAirValues")) {
        nSubProofValues = j["nSubAirValues"];
    } else {
        nSubProofValues = 0;
    }

    nStages = j["nStages"];

    qDeg = j["qDeg"];
    qDim = j["qDim"];

    for(uint64_t i = 0; i < j["openingPoints"].size(); i++) {
        openingPoints.push_back(j["openingPoints"][i]);
    }

    for(uint64_t i = 0; i < j["boundaries"].size(); i++) {
        Boundary b;
        b.name = j["boundaries"][i]["name"];
        if(b.name == string("everyFrame")) {
            b.offsetMin = j["boundaries"][i]["offsetMin"];
            b.offsetMax = j["boundaries"][i]["offsetMax"];
        }
        boundaries.push_back(b);
    }
    
    for (uint64_t i = 0; i < j["challengesMap"].size(); i++) 
    {
        PolMap map;
        map.stageNum = j["challengesMap"][i]["stageNum"];
        map.name = j["challengesMap"][i]["name"];
        map.dim = j["challengesMap"][i]["dim"];
        map.stageId = j["challengesMap"][i]["stageId"];
        challengesMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["cmPolsMap"].size(); i++) 
    {
        PolMap map;
        map.stage = j["cmPolsMap"][i]["stage"];
        map.stageNum = j["cmPolsMap"][i]["stageNum"];
        map.name = j["cmPolsMap"][i]["name"];
        map.dim = j["cmPolsMap"][i]["dim"];
        map.imPol = j["cmPolsMap"][i]["imPol"];
        map.stagePos = j["cmPolsMap"][i]["stagePos"];
        map.stageId = j["cmPolsMap"][i]["stageId"];
        cmPolsMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["evMap"].size(); i++)
    {
        EvMap map;
        map.setType(j["evMap"][i]["type"]);
        map.id = j["evMap"][i]["id"];
        map.prime = j["evMap"][i]["prime"];
        evMap.push_back(map);
    }

    for (auto it = j["mapSectionsN"].begin(); it != j["mapSectionsN"].end(); it++)  
    {
        mapSectionsN[it.key()] = it.value();
    }

    // TODO: Call this function from the prover
    setMapOffsets(true);
}

void StarkInfo::setMapOffsets(bool optimizeCommitStage1Pols) {
    uint64_t N = (1 << starkStruct.nBits);
    uint64_t NExtended = (1 << starkStruct.nBitsExt);

    mapOffsets[std::make_pair("cm1", false)] = 0;
    for(uint64_t stage = 2; stage <= nStages; stage++) {
        string prevStage = "cm" + to_string(stage - 1);
        string currStage = "cm" + to_string(stage);
        mapOffsets[std::make_pair(currStage, false)] = mapOffsets[std::make_pair(prevStage, false)] + N * mapSectionsN[prevStage];
    }

    if(optimizeCommitStage1Pols) {
        mapOffsets[std::make_pair("cm1", true)] = mapOffsets[std::make_pair("cm" + to_string(nStages), false)] + N * mapSectionsN["cm" + to_string(nStages)];
        mapOffsets[std::make_pair("tmpExp", false)] = mapOffsets[std::make_pair("cm1", true)] + NExtended * mapSectionsN["cm1"];
        mapOffsets[std::make_pair("cm2", true)] = mapOffsets[std::make_pair("tmpExp", false)] + N * mapSectionsN["tmpExp"];
    } else {
        mapOffsets[std::make_pair("tmpExp", false)] = mapOffsets[std::make_pair("cm" + to_string(nStages), false)] + N * mapSectionsN["cm" + to_string(nStages)];
        mapOffsets[std::make_pair("cm1", true)] = mapOffsets[std::make_pair("tmpExp", false)] + N * mapSectionsN["tmpExp"];
        mapOffsets[std::make_pair("cm2", true)] = mapOffsets[std::make_pair("cm1", true)] + NExtended * mapSectionsN["cm1"];
    }

    for(uint64_t stage = 3; stage <= nStages + 1; stage++) {
        string prevStage = "cm" + to_string(stage - 1);
        string currStage = "cm" + to_string(stage);
        mapOffsets[std::make_pair(currStage, true)] = mapOffsets[std::make_pair(prevStage, true)] + NExtended * mapSectionsN[prevStage];
    }

    mapOffsets[std::make_pair("q", true)] = mapOffsets[std::make_pair("cm" + to_string(nStages + 1), true)] + NExtended * mapSectionsN["cm" + to_string(nStages + 1)];
    mapOffsets[std::make_pair("f", true)] = mapOffsets[std::make_pair("q", true)] + NExtended * qDim;

    // TODO: Use hints to check if enough space is reserved, otherwise add more extra space
    mapTotalN = mapOffsets[std::make_pair("f", true)] + NExtended * 3;
}

Polinomial StarkInfo::getPolinomial(Goldilocks::Element *pAddress, uint64_t idPol, uint64_t deg)
{
    PolMap polInfo = cmPolsMap[idPol];
    uint64_t dim = polInfo.dim;
    uint64_t domainExtended = deg == uint64_t(1 << starkStruct.nBitsExt);
    uint64_t nCols = mapSectionsN[polInfo.stage];
    uint64_t offset = mapOffsets[std::make_pair(polInfo.stage, domainExtended)];
    offset += polInfo.stagePos;
    return Polinomial(&pAddress[offset], deg, dim, nCols, std::to_string(idPol));
}

opType string2opType(const string s) 
{
    if(s == "const") 
        return const_;
    if(s == "cm")
        return cm;
    if(s == "tmp")
        return tmp;
    if(s == "public")
        return public_;
    if(s == "subproofValue")
        return subproofvalue;
    if(s == "challenge")
        return challenge;
    if(s == "number")
        return number;
    zklog.error("string2opType() found invalid string=" + s);
    exitProcess();
    exit(-1);
}