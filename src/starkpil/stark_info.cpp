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
    if(j.contains("pil2")) {
        pil2 = j["pil2"];
    } else {
        pil2 = false;
    }

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

    nPublics = j["nPublics"];
    nConstants = j["nConstants"];
    nCm1 = j["nCm1"];

    if(j.contains("numChallenges")) {
        for(uint64_t i = 0; i < j["numChallenges"].size(); i++) {
            numChallenges.push_back(j["numChallenges"][i]);
        }
    } else {
        numChallenges.push_back(0);
        numChallenges.push_back(2);
        numChallenges.push_back(2);
    }

    nStages = numChallenges.size();

    nChallenges = std::accumulate(numChallenges.begin(), numChallenges.end(), 0);
    nChallenges += 4; // Challenges for FRI, Q and evals

    if(j.contains("openingPoints")) {
        for(uint64_t i = 0; i < j["openingPoints"].size(); i++) {
            openingPoints.push_back(j["openingPoints"][i]);
        }
    } else {
        openingPoints.push_back(0);
        openingPoints.push_back(1);
    }

    if(j.contains("boundaries")) {
       for(uint64_t i = 0; i < j["boundaries"].size(); i++) {
            Boundary b;
            b.name = j["boundaries"][i]["name"];
            if(b.name == string("everyFrame")) {
                b.offsetMin = j["boundaries"][i]["offsetMin"];
                b.offsetMax = j["boundaries"][i]["offsetMax"];
            }
            boundaries.push_back(b);
        }
    } else {
        Boundary b;
        b.name = std::string("everyRow");
        boundaries.push_back(b);
    }

    qDeg = j["qDeg"];
    qDim = j["qDim"];
    for (uint64_t i = 0; i < j["qs"].size(); i++) {
        qs.push_back(j["qs"][i]);
    }

    mapTotalN = j["mapTotalN"];
    
    for(uint64_t s = 1; s <= nStages + 1; s++) {
        string step = "cm" + to_string(s) + "_n";
        string stepExt = "cm" + to_string(s) + "_2ns";

        mapSectionsN.section[string2section(step)] = j["mapSectionsN"][step];
        mapSectionsN.section[string2section(stepExt)] = j["mapSectionsN"][stepExt];

        mapOffsets.section[string2section(step)] = j["mapOffsets"][step];
        mapOffsets.section[string2section(stepExt)] = j["mapOffsets"][stepExt];
    }

    mapSectionsN.section[tmpExp_n] = j["mapSectionsN"]["tmpExp_n"];
    mapOffsets.section[tmpExp_n] = j["mapOffsets"]["tmpExp_n"];

    mapSectionsN.section[q_2ns] = j["mapSectionsN"]["q_2ns"];
    mapOffsets.section[q_2ns] = j["mapOffsets"]["q_2ns"];

    mapSectionsN.section[f_2ns] = j["mapSectionsN"]["f_2ns"];
    mapOffsets.section[f_2ns] = j["mapOffsets"]["f_2ns"];

    for (uint64_t i = 0; i < j["evMap"].size(); i++)
    {
        EvMap map;
        map.setType(j["evMap"][i]["type"]);
        map.id = j["evMap"][i]["id"];
        map.prime = j["evMap"][i]["prime"];
        evMap.push_back(map);
    }


    if(pil2) {
        for (uint64_t i = 0; i < j["varPolMap"].size(); i++) 
        {
            CmPolMap map;
            map.stage = j["varPolMap"][i]["stage"];
            map.stageNum = j["varPolMap"][i]["stageNum"];
            map.name = j["varPolMap"][i]["name"];
            map.dim = j["varPolMap"][i]["dim"];
            map.imPol = j["varPolMap"][i]["imPol"];
            map.stagePos = j["varPolMap"][i]["stagePos"];
            map.stageId = j["varPolMap"][i]["stageId"];
            cmPolsMap.push_back(map);
        }
    } else {
        for (uint64_t i = 0; i < j["varPolMap"].size(); i++)
        {
            VarPolMap map;
            std::string section = j["varPolMap"][i]["section"];
            map.section = string2section(section);
            map.sectionPos = j["varPolMap"][i]["sectionPos"];
            map.dim = j["varPolMap"][i]["dim"];
            varPolMap.push_back(map);
        }

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

        for (auto it =  j["exp2pol"].begin(); it != j["exp2pol"].end(); ++it) {
            uint64_t key = std::stoull(it.key()); 
            uint64_t value = it.value();
            exp2pol.insert(pair(key,value));
        }
    }
}

uint64_t StarkInfo::getPolinomialRef(std::string type, uint64_t index) {
    if(type == "cm_n") 
        return cm_n[index];
    if (type == "cm_2ns") 
        return cm_2ns[index];
    if (type == "exp")
        return exp2pol[index];
    if (type == "q") 
        return qs[index];
    zklog.error("getPolinomialRef() found invalid type=" + type);
    exitProcess();
    exit(-1);
}

Polinomial StarkInfo::getPolinomial(Goldilocks::Element *pAddress, uint64_t idPol, uint64_t deg)
{
    VarPolMap polInfo = varPolMap[idPol];
    uint64_t dim = polInfo.dim;
    uint64_t offset = mapOffsets.section[polInfo.section];
    offset += polInfo.sectionPos;
    uint64_t next = mapSectionsN.section[polInfo.section];
    return Polinomial(&pAddress[offset], deg, dim, next, std::to_string(idPol));
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