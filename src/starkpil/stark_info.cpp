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
}

void StarkInfo::setMapOffsets(std::vector<Hint> &hints) {
    uint64_t N = (1 << starkStruct.nBits);
    uint64_t NExtended = (1 << starkStruct.nBitsExt);

    mapTotalN = 0;

    // Set offsets for all stages in the extended field (cm1, cm2, ..., cmN)
    for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
        mapOffsets[std::make_pair("cm" + to_string(stage), true)] = mapTotalN;
        mapTotalN += NExtended * mapSectionsN["cm" + to_string(stage)];
    }

    mapOffsets[std::make_pair("q", true)] = mapTotalN;
    mapTotalN += NExtended * qDim;

    uint64_t offsetPolsBasefield = mapOffsets[std::make_pair("cm" + to_string(nStages), true)];

    // Set offsets for all stages in the basefield field (cm1, cm2, ..., tmpExp)
    for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
        string section;
        if(stage == 1) {
            section = "cm" + to_string(nStages);
        } else if(stage == nStages + 1) {
            section = "tmpExp";
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

    mapOffsetsPolsHints.resize(nStages);
    offsetsExtraMemoryHints.resize(hints.size());

    uint64_t additionalMemoryOffsetAvailable = offsetPolsBasefield;
    uint64_t limitMemoryOffset = mapOffsets[std::make_pair("cm" + to_string(nStages), true)];
    for(uint64_t stage = 1; stage <= nStages; stage++) {
        uint64_t memoryOffset = stage == nStages 
            ? additionalMemoryOffsetAvailable
            : mapOffsets[std::make_pair("cm" + to_string(stage), true)];

        // Get hints stage 
        vector<uint64_t> hintsStage;

        for(uint64_t j = 0; j < hints.size(); ++j) {
            Hint hint = hints[j];
            auto hintHandler = Hints::HintHandlerBuilder::create(hint.name)->build();
            std::vector<string> srcFields = hintHandler->getSources();
            std::vector<string> dstFields = hintHandler->getDestinations();

            if(isHintStage(stage, hint, dstFields)) {
                hintsStage.push_back(j);
            }
        }

        // Set memory transposed pols
        for(uint64_t j = 0; j < hintsStage.size(); ++j) {
            Hint hint = hints[hintsStage[j]];
            auto hintHandler = Hints::HintHandlerBuilder::create(hint.name)->build();
            std::vector<string> srcFields = hintHandler->getSources();
            std::vector<string> dstFields = hintHandler->getDestinations();
            
            setMemoryPolsHint(stage, hint, srcFields, memoryOffset, limitMemoryOffset, additionalMemoryOffsetAvailable);
            setMemoryPolsHint(stage, hint, dstFields, memoryOffset, limitMemoryOffset, additionalMemoryOffsetAvailable);
        }

        if(memoryOffset > mapTotalN) {
            mapTotalN = memoryOffset;
        }

        // Set extra memory
        for(uint64_t j = 0; j < hintsStage.size(); ++j) {
            Hint hint = hints[hintsStage[j]];
            auto hintHandler = Hints::HintHandlerBuilder::create(hint.name)->build();

            uint64_t extraMemoryNeeded = hintHandler->getMemoryNeeded(N);
            if(memoryOffset < limitMemoryOffset && memoryOffset + extraMemoryNeeded > limitMemoryOffset) {
                memoryOffset = additionalMemoryOffsetAvailable;
            }
            offsetsExtraMemoryHints[hintsStage[j]] = memoryOffset;
            memoryOffset += extraMemoryNeeded;
        }

        if(memoryOffset > mapTotalN) {
            mapTotalN = memoryOffset;
        }
    }


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
            if(memoryAvailableEnd > memoryAvailableExtended) {
                memoryAvailable = memoryAvailableEnd;
                startBuffer = startBufferEnd;
            } else {
                memoryAvailable = memoryAvailableExtended;
                startBuffer = startBufferExtended;
            }
        }
        
        uint64_t minBlocks = 4;

        uint64_t memoryNTTHelper = NExtended * mapSectionsN["cm" + to_string(stage)];
        if(memoryAvailable * minBlocks < memoryNTTHelper) {
            memoryAvailable = memoryNTTHelper / minBlocks;
            if(startBuffer + memoryAvailable > mapTotalN) {
                mapTotalN = startBuffer + memoryAvailable;
            }   
        }
        
        mapNTTOffsetsHelpers["cm" + to_string(stage)] = std::make_pair(startBuffer, memoryAvailable);
    }
}

void StarkInfo::setMemoryPolsHint(uint64_t stage, Hint &hint, std::vector<string> &fields, uint64_t &memoryOffset, uint64_t limitMemoryOffset, uint64_t additionalMemoryOffset) {
     for(uint64_t k = 0; k < fields.size(); ++k) {
        std::string polName = fields[k];
        auto it = hint.fields.find(polName);
        if (it == hint.fields.end())
        {
            zklog.error("Unknown field name=" + polName);
            exitProcess();
            exit(-1);
        }
        HintField hintField = hint.fields[polName];
        if(mapOffsetsPolsHints[stage - 1].find(hintField.id) != mapOffsetsPolsHints[stage - 1].end()) continue;
        if (hintField.operand == opType::cm || hintField.operand == opType::tmp) {
            PolMap polInfo = cmPolsMap[hintField.id];
            uint64_t memoryUsed = (1 << starkStruct.nBits) * polInfo.dim + 8;
            if(memoryOffset < limitMemoryOffset && memoryOffset + memoryUsed > limitMemoryOffset) {
                memoryOffset = additionalMemoryOffset;
            }
            mapOffsetsPolsHints[stage - 1][hintField.id] = memoryOffset;
            memoryOffset += memoryUsed;
        }
    }
}

bool StarkInfo::isHintStage(uint64_t stage, Hint &hint, std::vector<string> &dstFields) {
    bool isHintStage_ = true;
    for(uint64_t i = 0; i < dstFields.size(); ++i) {
        auto it = hint.fields.find(dstFields[i]);
        if (it == hint.fields.end())
        {
            zklog.error("Unknown field name=" + dstFields[i]);
            exitProcess();
            exit(-1);
        }
        HintField hintField = hint.fields[dstFields[i]];
        if(hintField.operand == opType::subproofvalue) {
            if (nStages != stage) {
                isHintStage_ = false;
                break;
            }
        } else if(hintField.operand == opType::cm || hintField.operand == opType::tmp) {
            if(cmPolsMap[hintField.id].stageNum != stage) {
                isHintStage_ = false;
                break;
            }
        } else {
            zklog.error("Destination field=" + dstFields[i] + " has to be either a cm or tmp or subproofvalue");
            exitProcess();
            exit(-1);
        }
    }

    return isHintStage_;
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