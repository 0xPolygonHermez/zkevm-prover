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
    for (uint64_t i = 0; i < j["starkStruct"]["steps"].size(); i++)
    {
        StepStruct step;
        step.nBits = j["starkStruct"]["steps"][i]["nBits"];
        starkStruct.steps.push_back(step);
    }

    mapTotalN = j["mapTotalN"];
    nConstants = j["nConstants"];
    nPublics = j["nPublics"];
    nCm1 = j["nCm1"];
    nCm2 = j["nCm2"];
    nCm3 = j["nCm3"];
    nCm4 = j["nCm4"];
    friExpId = j["friExpId"];
    nExps = j["nExps"];
    qDim = j["qDim"];
    qDeg = j["qDeg"];

    if(starkStruct.verificationHashType == "BN128") {
        if(j.contains("merkleTreeArity")) {
            merkleTreeArity = j["merkleTreeArity"]; 
        } else {
            merkleTreeArity = 16;
        }
    }
    
    mapDeg.section[cm1_n] = j["mapDeg"]["cm1_n"];
    mapDeg.section[cm2_n] = j["mapDeg"]["cm2_n"];
    mapDeg.section[cm3_n] = j["mapDeg"]["cm3_n"];
    mapDeg.section[cm4_n] = j["mapDeg"]["cm4_n"];
    mapDeg.section[tmpExp_n] = j["mapDeg"]["tmpExp_n"];
    mapDeg.section[f_2ns] = j["mapDeg"]["f_2ns"];
    mapDeg.section[cm1_2ns] = j["mapDeg"]["cm1_2ns"];
    mapDeg.section[cm2_2ns] = j["mapDeg"]["cm2_2ns"];
    mapDeg.section[cm3_2ns] = j["mapDeg"]["cm3_2ns"];
    mapDeg.section[cm4_2ns] = j["mapDeg"]["cm4_2ns"];
    mapDeg.section[q_2ns] = j["mapDeg"]["q_2ns"];

    mapSectionsN.section[cm1_n] = j["mapSectionsN"]["cm1_n"];
    mapSectionsN.section[cm2_n] = j["mapSectionsN"]["cm2_n"];
    mapSectionsN.section[cm3_n] = j["mapSectionsN"]["cm3_n"];
    mapSectionsN.section[cm4_n] = j["mapSectionsN"]["cm4_n"];
    mapSectionsN.section[tmpExp_n] = j["mapSectionsN"]["tmpExp_n"];
    mapSectionsN.section[f_2ns] = j["mapSectionsN"]["f_2ns"];
    mapSectionsN.section[cm1_2ns] = j["mapSectionsN"]["cm1_2ns"];
    mapSectionsN.section[cm2_2ns] = j["mapSectionsN"]["cm2_2ns"];
    mapSectionsN.section[cm3_2ns] = j["mapSectionsN"]["cm3_2ns"];
    mapSectionsN.section[cm4_2ns] = j["mapSectionsN"]["cm4_2ns"];
    mapSectionsN.section[q_2ns] = j["mapSectionsN"]["q_2ns"];


    for (uint64_t i = 0; i < j["varPolMap"].size(); i++)
    {
        VarPolMap map;
        map.section = string2section(j["varPolMap"][i]["section"]);
        map.sectionPos = j["varPolMap"][i]["sectionPos"];
        map.dim = j["varPolMap"][i]["dim"];
        varPolMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["qs"].size(); i++)
        qs.push_back(j["qs"][i]);

    for (uint64_t i = 0; i < j["cm_n"].size(); i++)
        cm_n.push_back(j["cm_n"][i]);

    for (uint64_t i = 0; i < j["cm_2ns"].size(); i++)
        cm_2ns.push_back(j["cm_2ns"][i]);

    for (uint64_t i = 0; i < j["peCtx"].size(); i++)
    {
        PeCtx pe;
        pe.tExpId = j["peCtx"][i]["tExpId"];
        pe.fExpId = j["peCtx"][i]["fExpId"];
        pe.zId = j["peCtx"][i]["zId"];
        pe.c1Id = j["peCtx"][i]["c1Id"];
        pe.numId = j["peCtx"][i]["numId"];
        pe.denId = j["peCtx"][i]["denId"];
        pe.c2Id = j["peCtx"][i]["c2Id"];
        peCtx.push_back(pe);
    }

    for (uint64_t i = 0; i < j["puCtx"].size(); i++)
    {
        PuCtx pu;
        pu.tExpId = j["puCtx"][i]["tExpId"];
        pu.fExpId = j["puCtx"][i]["fExpId"];
        pu.h1Id = j["puCtx"][i]["h1Id"];
        pu.h2Id = j["puCtx"][i]["h2Id"];
        pu.zId = j["puCtx"][i]["zId"];
        pu.c1Id = j["puCtx"][i]["c1Id"];
        pu.numId = j["puCtx"][i]["numId"];
        pu.denId = j["puCtx"][i]["denId"];
        pu.c2Id = j["puCtx"][i]["c2Id"];
        puCtx.push_back(pu);
    }

    for (uint64_t i = 0; i < j["ciCtx"].size(); i++)
    {
        CiCtx ci;
        ci.zId = j["ciCtx"][i]["zId"];
        ci.numId = j["ciCtx"][i]["numId"];
        ci.denId = j["ciCtx"][i]["denId"];
        ci.c1Id = j["ciCtx"][i]["c1Id"];
        ci.c2Id = j["ciCtx"][i]["c2Id"];
        ciCtx.push_back(ci);
    }

    for (uint64_t i = 0; i < j["evMap"].size(); i++)
    {
        EvMap map;
        map.setType(j["evMap"][i]["type"]);
        map.id = j["evMap"][i]["id"];
        map.prime = j["evMap"][i]["prime"];
        evMap.push_back(map);
    }

    for (auto it =  j["exp2pol"].begin(); it != j["exp2pol"].end(); ++it) {
        std::string key = it.key(); 
        uint64_t value = it.value();
         exp2pol.insert(pair(key,value));
    }

    setMapOffsets();
}

void StarkInfo::getPol(void *pAddress, uint64_t idPol, PolInfo &polInfo)
{
    polInfo.map = varPolMap[idPol];
    polInfo.N = mapDeg.section[polInfo.map.section];
    polInfo.offset = mapOffsets.section[polInfo.map.section];
    polInfo.offset += polInfo.map.sectionPos;
    polInfo.size = mapSectionsN.section[polInfo.map.section];
    polInfo.pAddress = ((Goldilocks::Element *)pAddress) + polInfo.offset;
}

uint64_t StarkInfo::getPolSize(uint64_t polId)
{
    VarPolMap p = varPolMap[polId];
    uint64_t N = mapDeg.section[p.section];
    return N * p.dim * sizeof(Goldilocks::Element);
}

Polinomial StarkInfo::getPolinomial(Goldilocks::Element *pAddress, uint64_t idPol)
{
    VarPolMap polInfo = varPolMap[idPol];
    uint64_t dim = polInfo.dim;
    uint64_t N = mapDeg.section[polInfo.section];
    uint64_t nCols = mapSectionsN.section[polInfo.section];
    uint64_t offset = mapOffsets.section[polInfo.section];
    if(polInfo.section == eSection::tmpExp_n) {
        offset += polInfo.sectionPos*N;
        return Polinomial(&pAddress[offset], N, dim, dim, std::to_string(idPol));
    } else {
        offset += polInfo.sectionPos;
        return Polinomial(&pAddress[offset], N, dim, nCols, std::to_string(idPol));
    }    
}

eSection string2section(const string s)
{
    if (s == "cm1_n")
        return cm1_n;
    if (s == "cm2_n")
        return cm2_n;
    if (s == "cm3_n")
        return cm3_n;
    if (s == "cm4_n")
        return cm4_n;
    if (s == "tmpExp_n")
        return tmpExp_n;
    if (s == "f_2ns")
        return f_2ns;
    if (s == "cm1_2ns")
        return cm1_2ns;
    if (s == "cm2_2ns")
        return cm2_2ns;
    if (s == "cm3_2ns")
        return cm3_2ns;
    if (s == "cm4_2ns")
        return cm4_2ns;
    if (s == "q_2ns")
        return q_2ns;
    zklog.error("string2section() found invalid string=" + s);
    exitProcess();
    exit(-1);
}

void StarkInfo::setMapOffsets() {
    uint64_t N = (1 << starkStruct.nBits);
    uint64_t NExtended = (1 << starkStruct.nBitsExt);

    mapTotalN = 0;

    // Set offsets for all stages in the extended field (cm1, cm2, cm3, cm4)
    mapOffsets.section[cm1_2ns] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN.section[cm1_2ns];

    mapOffsets.section[cm2_2ns] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN.section[cm2_2ns];

    mapOffsets.section[cm3_2ns] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN.section[cm3_2ns];

    mapOffsets.section[cm4_2ns] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN.section[cm4_2ns];

    mapOffsets.section[q_2ns] = mapTotalN;
    mapTotalN += NExtended * qDim;
    
    uint64_t offsetPolsBasefield = mapOffsets.section[cm3_2ns];

    // Set offsets for all stages in the basefield field (cm1, cm2, cm3, tmpExp)
    mapOffsets.section[cm3_n] = offsetPolsBasefield;
    offsetPolsBasefield += N * mapSectionsN.section[cm3_n];

    mapOffsets.section[cm1_n] = offsetPolsBasefield;
    offsetPolsBasefield += N * mapSectionsN.section[cm1_n];

    mapOffsets.section[cm2_n] = offsetPolsBasefield;
    offsetPolsBasefield += N * mapSectionsN.section[cm2_n];

    mapOffsets.section[tmpExp_n] = offsetPolsBasefield;
    offsetPolsBasefield += N * mapSectionsN.section[tmpExp_n];

    if(offsetPolsBasefield > mapTotalN) mapTotalN = offsetPolsBasefield;

    // Stage FRIPolynomial
    uint64_t offsetPolsFRI = mapOffsets.section[q_2ns];
    mapOffsets.section[xDivXSubXi_2ns] = offsetPolsFRI;
    offsetPolsFRI += 2 * NExtended * FIELD_EXTENSION;
    
    mapOffsets.section[f_2ns] = offsetPolsFRI;
    offsetPolsFRI += NExtended * FIELD_EXTENSION;

    if(offsetPolsFRI > mapTotalN) mapTotalN = offsetPolsFRI;

    uint64_t offsetPolsEvals = mapOffsets.section[q_2ns];
    mapOffsets.section[LEv] = offsetPolsEvals;
    offsetPolsEvals += 2 * N * FIELD_EXTENSION;
    
    mapOffsets.section[evals] = offsetPolsEvals;
    offsetPolsEvals += evMap.size() * omp_get_max_threads() * FIELD_EXTENSION;

    mapNTTOffsetsHelpers["LEv"] = std::make_pair(offsetPolsEvals, N * FIELD_EXTENSION * 2);
    offsetPolsEvals += N * FIELD_EXTENSION * 2;
   
    if(offsetPolsEvals > mapTotalN) mapTotalN = offsetPolsEvals;

    offsetsExtraMemoryH1H2.resize(puCtx.size());

    uint64_t additionalMemoryOffsetAvailable = offsetPolsBasefield;
    uint64_t limitMemoryOffset = mapOffsets.section[cm3_2ns];

    // Memory H1H2
    uint64_t memoryOffsetH1H2 = mapOffsets.section[cm2_2ns];

    uint64_t numCommited = nCm1;

    for(uint64_t i = 0; i < puCtx.size(); ++i) {
        setMemoryPol(2, exp2pol[to_string(puCtx[i].fExpId)], memoryOffsetH1H2, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(2, exp2pol[to_string(puCtx[i].tExpId)], memoryOffsetH1H2, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(2, cm_n[numCommited + i * 2], memoryOffsetH1H2, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(2, cm_n[numCommited + i * 2 + 1], memoryOffsetH1H2, limitMemoryOffset, additionalMemoryOffsetAvailable);
    }

    if(memoryOffsetH1H2 > mapTotalN) {
        mapTotalN = memoryOffsetH1H2;
    }

    // Set extra memory for H1H2
    for(uint64_t i = 0; i < puCtx.size(); ++i) {
        uint64_t extraMemoryNeeded = 8 * N;
        if(memoryOffsetH1H2 < limitMemoryOffset && memoryOffsetH1H2 + extraMemoryNeeded > limitMemoryOffset) {
            memoryOffsetH1H2 = additionalMemoryOffsetAvailable;
        }
        offsetsExtraMemoryH1H2[i] = memoryOffsetH1H2;
        memoryOffsetH1H2 += extraMemoryNeeded;
    }

    if(memoryOffsetH1H2 > mapTotalN) {
        mapTotalN = memoryOffsetH1H2;
    }

    numCommited = numCommited + puCtx.size() * 2;
    
    // Memory grand product
    uint64_t memoryOffsetGrandProduct = additionalMemoryOffsetAvailable;
    for(uint64_t i = 0; i < puCtx.size(); ++i) {
        setMemoryPol(3, exp2pol[to_string(puCtx[i].numId)], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(3, exp2pol[to_string(puCtx[i].denId)], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(3, cm_n[numCommited + i], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
    }
    numCommited += puCtx.size();

    for(uint64_t i = 0; i < peCtx.size(); ++i) {
        setMemoryPol(3, exp2pol[to_string(peCtx[i].numId)], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(3, exp2pol[to_string(peCtx[i].denId)], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(3, cm_n[numCommited + i], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
    }
    numCommited += peCtx.size();

    for(uint64_t i = 0; i < ciCtx.size(); ++i) {
        setMemoryPol(3, exp2pol[to_string(ciCtx[i].numId)], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(3, exp2pol[to_string(ciCtx[i].denId)], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
        setMemoryPol(3, cm_n[numCommited + i], memoryOffsetGrandProduct, limitMemoryOffset, additionalMemoryOffsetAvailable);
    }
    numCommited += ciCtx.size();
    
    if(memoryOffsetGrandProduct > mapTotalN) {
        mapTotalN = memoryOffsetGrandProduct;
    }

    uint64_t startBuffer;
    uint64_t memoryAvailable;
    uint64_t minBlocks = 4;

    // Stage 1 NTT Memory Helper
    uint64_t startBufferExtended = mapOffsets.section[cm2_2ns];
    uint64_t memoryAvailableEnd = mapTotalN - offsetPolsBasefield;
    uint64_t memoryAvailableExtended =  mapOffsets.section[cm3_n] - startBufferExtended;
    uint64_t nttMemoryHelper = NExtended * mapSectionsN.section[cm1_n];
    if(memoryAvailableExtended > memoryAvailableEnd && memoryAvailableExtended * 8 > nttMemoryHelper) {
        memoryAvailable = memoryAvailableExtended;
        startBuffer = startBufferExtended;
    } else {
        memoryAvailable = memoryAvailableEnd;
        startBuffer = offsetPolsBasefield;
    }

    uint64_t memoryNTTHelperStage1 = NExtended * mapSectionsN.section[cm1_2ns];
    if(startBuffer >= offsetPolsBasefield && memoryAvailable * minBlocks < memoryNTTHelperStage1) {
        memoryAvailable = memoryNTTHelperStage1 / minBlocks;
        if(startBuffer + memoryAvailable > mapTotalN) {
            mapTotalN = startBuffer + memoryAvailable;
        }   
    }
    mapNTTOffsetsHelpers["cm1"] = std::make_pair(startBuffer, memoryAvailable);

    // Stage 2 NTT Memory Helper
    memoryAvailable = mapTotalN - offsetPolsBasefield;
    startBuffer = offsetPolsBasefield;
    uint64_t memoryNTTHelperStage2 = NExtended * mapSectionsN.section[cm2_2ns];
    if(startBuffer >= offsetPolsBasefield && memoryAvailable * minBlocks < memoryNTTHelperStage2) {
        memoryAvailable = memoryNTTHelperStage2 / minBlocks;
        if(startBuffer + memoryAvailable > mapTotalN) {
            mapTotalN = startBuffer + memoryAvailable;
        }   
    }
    mapNTTOffsetsHelpers["cm2"] = std::make_pair(startBuffer, memoryAvailable);

    // Stage 3 NTT Memory Helper
    startBuffer = mapOffsets.section[cm4_2ns];
    memoryAvailable = mapTotalN - startBuffer;
    uint64_t memoryNTTHelperStage3 = NExtended * mapSectionsN.section[cm3_2ns];
    if(startBuffer >= offsetPolsBasefield && memoryAvailable * minBlocks < memoryNTTHelperStage3) {
        memoryAvailable = memoryNTTHelperStage3 / minBlocks;
        if(startBuffer + memoryAvailable > mapTotalN) {
            mapTotalN = startBuffer + memoryAvailable;
        }   
    }
    mapNTTOffsetsHelpers["cm3"] = std::make_pair(startBuffer, memoryAvailable);

    // Stage 4 NTT Memory Helper
    startBuffer = mapOffsets.section[q_2ns] + NExtended * qDim;
    memoryAvailable = mapTotalN - startBuffer;
     uint64_t memoryNTTHelperStage4 = NExtended * mapSectionsN.section[cm4_2ns];
    if(startBuffer >= offsetPolsBasefield && memoryAvailable * minBlocks < memoryNTTHelperStage4) {
        memoryAvailable = memoryNTTHelperStage4 / minBlocks;
        if(startBuffer + memoryAvailable > mapTotalN) {
            mapTotalN = startBuffer + memoryAvailable;
        }   
    }
    mapNTTOffsetsHelpers["cm4"] = std::make_pair(startBuffer, memoryAvailable);
}

void StarkInfo::setMemoryPol(uint64_t stage, uint64_t polId, uint64_t &memoryOffset, uint64_t limitMemoryOffset, uint64_t additionalMemoryOffset) {
    VarPolMap pol = varPolMap[polId];
    if(pol.section == eSection::tmpExp_n) return;
    uint64_t memoryUsed = (1 << starkStruct.nBits) * pol.dim + 8;
    if(memoryOffset < limitMemoryOffset && memoryOffset + memoryUsed > limitMemoryOffset) {
        memoryOffset = additionalMemoryOffset;
    }
    if(stage == 2) {
        mapOffsetsPolsH1H2[polId] = memoryOffset;
    } else {
        mapOffsetsPolsGrandProduct[polId] = memoryOffset;
    }
    
    memoryOffset += memoryUsed;

}

void StarkInfo::setCHelpersInfo(uint64_t stage) {
    uint64_t buffTSize = 5;

    if(stage == 4 || stage == 5) {
        buffTSize++;
    }

    bool domainExtended = stage <= 3 ? false : true;

    buffTOffsetsStages.resize(buffTSize);
    nColsStages.resize(buffTSize);
    nColsStagesAcc.resize(buffTSize);
    offsetsStages.resize(buffTSize);

    // Stage 0: Constants
    nColsStages[0] = nConstants;
    nColsStagesAcc[0] = 0;
    buffTOffsetsStages[0] = 0;
    offsetsStages[0] = 0;

    // Stage 1
    nColsStages[1] = mapSectionsN.section[eSection::cm1_n];
    nColsStagesAcc[1] = nColsStagesAcc[0] + nColsStages[0];
    buffTOffsetsStages[1] = 2*nColsStages[0];
    if(domainExtended) {
        offsetsStages[1] = mapOffsets.section[eSection::cm1_2ns];
    } else {
        offsetsStages[1] = mapOffsets.section[eSection::cm1_n];
    }

    // Stage 2
    nColsStages[2] = mapSectionsN.section[eSection::cm2_n];
    nColsStagesAcc[2] = nColsStagesAcc[1] + nColsStages[1];
    buffTOffsetsStages[2] = buffTOffsetsStages[1] + 2*nColsStages[1];
    if(domainExtended) {
        offsetsStages[2] = mapOffsets.section[eSection::cm2_2ns];
    } else {
        offsetsStages[2] = mapOffsets.section[eSection::cm2_n];
    }

    // Stage 3
    nColsStages[3] = mapSectionsN.section[eSection::cm3_n];
    nColsStagesAcc[3] = nColsStagesAcc[2] + nColsStages[2];
    buffTOffsetsStages[3] = buffTOffsetsStages[2] + 2*nColsStages[2];
    if(domainExtended) {
        offsetsStages[3] = mapOffsets.section[eSection::cm3_2ns];
    } else {
        offsetsStages[3] = mapOffsets.section[eSection::cm3_n];
    }
    
    // Stage 4: Basefield (tmpExp) // Extended (Stage 4)
    if(domainExtended) {
        nColsStages[4] = mapSectionsN.section[eSection::cm4_2ns];
    } else {
        nColsStages[4] = mapSectionsN.section[eSection::tmpExp_n];
    }
    nColsStagesAcc[4] = nColsStagesAcc[3] + nColsStages[3];
    buffTOffsetsStages[4] = buffTOffsetsStages[3] + 2*nColsStages[3];
    if(domainExtended) {
        offsetsStages[4] = mapOffsets.section[eSection::cm4_2ns];
    } else {
        offsetsStages[4] = mapOffsets.section[eSection::tmpExp_n];
    }

    if(stage == 4 || stage == 5) {
        if(stage == 4) {
            nColsStages[buffTSize - 1] = mapSectionsN.section[eSection::q_2ns];
            offsetsStages[buffTSize - 1] = mapOffsets.section[eSection::q_2ns];
        } else if(stage == 5) {
            nColsStages[buffTSize - 1] = mapSectionsN.section[eSection::f_2ns];
            offsetsStages[buffTSize - 1] = mapOffsets.section[eSection::f_2ns];
        }

        nColsStagesAcc[buffTSize - 1] = nColsStagesAcc[buffTSize - 2] + nColsStages[buffTSize - 2];
        buffTOffsetsStages[buffTSize - 1] = buffTOffsetsStages[buffTSize - 2] + 2*nColsStages[buffTSize - 2];
    }
}