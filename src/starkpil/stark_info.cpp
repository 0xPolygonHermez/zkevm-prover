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

    if(j.contains("nSubproofValues")) {
        nSubProofValues = j["nSubproofValues"];
    } else {
        nSubProofValues = 0;
    }

    nStages = j["nStages"];

    qDeg = j["qDeg"];
    qDim = j["qDim"];

    friExpId = j["friExpId"];
    cExpId = j["cExpId"];

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
        map.stage = j["challengesMap"][i]["stage"];
        map.name = j["challengesMap"][i]["name"];
        map.dim = j["challengesMap"][i]["dim"];
        map.stageId = j["challengesMap"][i]["stageId"];
        challengesMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["cmPolsMap"].size(); i++) 
    {
        PolMap map;
        map.stage = j["cmPolsMap"][i]["stage"];
        map.name = j["cmPolsMap"][i]["name"];
        map.dim = j["cmPolsMap"][i]["dim"];
        map.imPol = j["cmPolsMap"][i].contains("imPol") ? true : false;
        map.stagePos = j["cmPolsMap"][i]["stagePos"];
        map.stageId = j["cmPolsMap"][i]["stageId"];
        if(j["cmPolsMap"][i].contains("expId")) {
            map.expId = j["cmPolsMap"][i]["expId"];
        }
        cmPolsMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["constPolsMap"].size(); i++) 
    {
        PolMap map;
        map.stage = j["constPolsMap"][i]["stage"];
        map.name = j["constPolsMap"][i]["name"];
        map.dim = j["constPolsMap"][i]["dim"];
        map.imPol = false;
        map.stagePos = j["constPolsMap"][i]["stageId"];
        map.stageId = j["constPolsMap"][i]["stageId"];
        constPolsMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["evMap"].size(); i++)
    {
        EvMap map;
        map.setType(j["evMap"][i]["type"]);
        map.id = j["evMap"][i]["id"];
        map.prime = j["evMap"][i]["prime"];
        if(j["evMap"][i].contains("openingPos")) {
            map.openingPos = j["evMap"][i]["openingPos"];
        } else {
            int64_t prime = map.prime;
            auto openingPoint = std::find_if(openingPoints.begin(), openingPoints.end(), [prime](int p) { return p == prime; });
            if(openingPoint == openingPoints.end()) {
                zklog.error("Opening point not found");
                exitProcess();
                exit(-1);
            }
            map.openingPos = std::distance(openingPoints.begin(), openingPoint);
        }
        evMap.push_back(map);
    }

    for (auto it = j["mapSectionsN"].begin(); it != j["mapSectionsN"].end(); it++)  
    {
        mapSectionsN[it.key()] = it.value();
    }

    setMapOffsets();
}

void StarkInfo::setMapOffsets() {
    uint64_t N = (1 << starkStruct.nBits);
    uint64_t NExtended = (1 << starkStruct.nBitsExt);

    // Set offsets for constants
    mapOffsets[std::make_pair("const", false)] = 0;
    mapOffsets[std::make_pair("const", true)] = 0;

    mapTotalN = 0;

    // Set offsets for all stages in the extended field (cm1, cm2, ..., cmN)
    for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
        mapOffsets[std::make_pair("cm" + to_string(stage), true)] = mapTotalN;
        mapTotalN += NExtended * mapSectionsN["cm" + to_string(stage)];
    }

    mapOffsets[std::make_pair("q", true)] = mapTotalN;
    mapTotalN += NExtended * qDim;

    uint64_t offsetPolsBasefield = mapOffsets[std::make_pair("cm" + to_string(nStages), true)];

    // Set offsets for all stages in the basefield field (cm1, cm2, ... )
    for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
        string section;
        if(stage == 1) {
            section = "cm" + to_string(nStages);
        } else {
            section = "cm" + to_string(stage - 1);
        }
        mapOffsets[std::make_pair(section, false)] = offsetPolsBasefield;
        offsetPolsBasefield += N * mapSectionsN[section];
    }

    if(offsetPolsBasefield > mapTotalN) mapTotalN = offsetPolsBasefield;

    // Stage FRIPolynomial
    uint64_t offsetPolsFRI = mapOffsets[std::make_pair("q", true)];
    mapOffsets[std::make_pair("xDivXSubXi", true)] = offsetPolsFRI;
    offsetPolsFRI += openingPoints.size() * NExtended * FIELD_EXTENSION;
    
    mapOffsets[std::make_pair("f", true)] = offsetPolsFRI;
    offsetPolsFRI += NExtended * FIELD_EXTENSION;

    if(offsetPolsFRI > mapTotalN) mapTotalN = offsetPolsFRI;

    uint64_t offsetPolsEvals = mapOffsets[std::make_pair("q", true)];
    mapOffsets[std::make_pair("LEv", true)] = offsetPolsEvals;
    offsetPolsEvals += N * openingPoints.size() * FIELD_EXTENSION;
    
    mapOffsets[std::make_pair("evals", true)] = offsetPolsEvals;
    offsetPolsEvals += evMap.size() * omp_get_max_threads() * FIELD_EXTENSION;

    mapNTTOffsetsHelpers["LEv"] = std::make_pair(offsetPolsEvals, N * FIELD_EXTENSION * openingPoints.size());
    offsetPolsEvals += N * FIELD_EXTENSION * openingPoints.size();
    if(offsetPolsEvals > mapTotalN) mapTotalN = offsetPolsEvals;

    for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
        uint64_t startBuffer;
        uint64_t memoryAvailable;
        if(stage == nStages + 1) {
            startBuffer = mapOffsets[std::make_pair("q", true)] + NExtended * qDim;
            memoryAvailable = mapTotalN - startBuffer;
        } else if(stage == nStages) {
            startBuffer = mapOffsets[std::make_pair("cm" + to_string(nStages + 1), true)];
            memoryAvailable = mapTotalN - startBuffer;
        } else {
            uint64_t startBufferEnd = offsetPolsBasefield;
            uint64_t startBufferExtended = mapOffsets[std::make_pair("cm" + to_string(stage + 1), true)];
            uint64_t memoryAvailableEnd = mapTotalN - startBufferEnd;
            uint64_t memoryAvailableExtended =  mapOffsets[std::make_pair("cm" + to_string(nStages), false)] - startBufferExtended;
            uint64_t nttMemoryHelper = NExtended * mapSectionsN["cm" + to_string(stage)];
            if(memoryAvailableExtended > memoryAvailableEnd && memoryAvailableExtended * 8 > nttMemoryHelper) {
                memoryAvailable = memoryAvailableExtended;
                startBuffer = startBufferExtended;
            } else {
                memoryAvailable = memoryAvailableEnd;
                startBuffer = startBufferEnd;
            }
        }
        
        uint64_t minBlocks = 4;

        uint64_t memoryNTTHelper = NExtended * mapSectionsN["cm" + to_string(stage)];
        if(startBuffer >= offsetPolsBasefield && memoryAvailable * minBlocks < memoryNTTHelper) {
            memoryAvailable = memoryNTTHelper / minBlocks;
            if(startBuffer + memoryAvailable > mapTotalN) {
                mapTotalN = startBuffer + memoryAvailable;
            }   
        }
        
        mapNTTOffsetsHelpers["cm" + to_string(stage)] = std::make_pair(startBuffer, memoryAvailable);
    }
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