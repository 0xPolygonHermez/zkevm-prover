#include "stark_info.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

StarkInfo::StarkInfo(const Config &config, string file) : config(config)
{
    // Avoid initialization if we are not going to generate any proof
    if (!config.generateProof())
        return;

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

    mapOffsets.section[cm1_n] = j["mapOffsets"]["cm1_n"];
    mapOffsets.section[cm2_n] = j["mapOffsets"]["cm2_n"];
    mapOffsets.section[cm3_n] = j["mapOffsets"]["cm3_n"];
    mapOffsets.section[cm4_n] = j["mapOffsets"]["cm4_n"];
    mapOffsets.section[tmpExp_n] = j["mapOffsets"]["tmpExp_n"];
    mapOffsets.section[f_2ns] = j["mapOffsets"]["f_2ns"];
    mapOffsets.section[cm1_2ns] = j["mapOffsets"]["cm1_2ns"];
    mapOffsets.section[cm2_2ns] = j["mapOffsets"]["cm2_2ns"];
    mapOffsets.section[cm3_2ns] = j["mapOffsets"]["cm3_2ns"];
    mapOffsets.section[cm4_2ns] = j["mapOffsets"]["cm4_2ns"];
    mapOffsets.section[q_2ns] = j["mapOffsets"]["q_2ns"];

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
    uint64_t offset = mapOffsets.section[polInfo.section];
    offset += polInfo.sectionPos;
    uint64_t next = mapSectionsN.section[polInfo.section];
    return Polinomial(&pAddress[offset], N, dim, next, std::to_string(idPol));
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