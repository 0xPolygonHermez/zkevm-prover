#include "stark_info.hpp"
#include "utils.hpp"

StarkInfo::StarkInfo(const Config &config) : config(config)
{
    json starkInfoJson;
    file2json(config.starkInfoFile, starkInfoJson);
    load(starkInfoJson);
}

void StarkInfo::load (json j)
{
    starkStruct.nBits = j["starkStruct"]["nBits"];
    starkStruct.nBitsExt = j["starkStruct"]["nBitsExt"];
    starkStruct.nQueries = j["starkStruct"]["nQueries"];
    starkStruct.verificationHashType = j["starkStruct"]["verificationHashType"];
    for (uint64_t i=0; i<j["starkStruct"]["steps"].size(); i++)
    {
        Step step;
        step.nBits = j["starkStruct"]["steps"][i]["nBits"];
        starkStruct.steps.push_back(step);
    }

    mapTotalN = j["mapTotalN"];
    nConstants = j["nConstants"];
    nCm1 = j["nCm1"];
    nCm2 = j["nCm2"];
    nCm3 = j["nCm3"];
    nCm4 = j["nCm4"];
    nQ = j["nQ"];
    friExpId = j["friExpId"];
    nExps = j["nExps"];

    mapDeg.cm1_n = j["mapDeg"]["cm1_n"];
    mapDeg.cm2_n = j["mapDeg"]["cm2_n"];
    mapDeg.cm3_n = j["mapDeg"]["cm3_n"];
    mapDeg.exps_withq_n = j["mapDeg"]["exps_withq_n"];
    mapDeg.exps_withoutq_n = j["mapDeg"]["exps_withoutq_n"];
    mapDeg.cm1_2ns = j["mapDeg"]["cm1_2ns"];
    mapDeg.cm2_2ns = j["mapDeg"]["cm2_2ns"];
    mapDeg.cm3_2ns = j["mapDeg"]["cm3_2ns"];
    mapDeg.q_2ns = j["mapDeg"]["q_2ns"];
    mapDeg.exps_withq_2ns = j["mapDeg"]["exps_withq_2ns"];
    mapDeg.exps_withoutq_2ns = j["mapDeg"]["exps_withoutq_2ns"];

    mapOffsets.cm1_n = j["mapOffsets"]["cm1_n"];
    mapOffsets.cm2_n = j["mapOffsets"]["cm2_n"];
    mapOffsets.cm3_n = j["mapOffsets"]["cm3_n"];
    mapOffsets.exps_withq_n = j["mapOffsets"]["exps_withq_n"];
    mapOffsets.exps_withoutq_n = j["mapOffsets"]["exps_withoutq_n"];
    mapOffsets.cm1_2ns = j["mapOffsets"]["cm1_2ns"];
    mapOffsets.cm2_2ns = j["mapOffsets"]["cm2_2ns"];
    mapOffsets.cm3_2ns = j["mapOffsets"]["cm3_2ns"];
    mapOffsets.q_2ns = j["mapOffsets"]["q_2ns"];
    mapOffsets.exps_withq_2ns = j["mapOffsets"]["exps_withq_2ns"];
    mapOffsets.exps_withoutq_2ns = j["mapOffsets"]["exps_withoutq_2ns"];

    for (uint64_t i=0; i<j["mapSections"]["cm1_n"].size(); i++) mapSections.cm1_n.push_back(j["mapSections"]["cm1_n"][i]);
    for (uint64_t i=0; i<j["mapSections"]["cm2_n"].size(); i++) mapSections.cm2_n.push_back(j["mapSections"]["cm2_n"][i]);
    for (uint64_t i=0; i<j["mapSections"]["cm3_n"].size(); i++) mapSections.cm3_n.push_back(j["mapSections"]["cm3_n"][i]);
    for (uint64_t i=0; i<j["mapSections"]["exps_withq_n"].size(); i++) mapSections.exps_withq_n.push_back(j["mapSections"]["exps_withq_n"][i]);
    for (uint64_t i=0; i<j["mapSections"]["exps_withoutq_n"].size(); i++) mapSections.exps_withoutq_n.push_back(j["mapSections"]["exps_withoutq_n"][i]);
    for (uint64_t i=0; i<j["mapSections"]["cm1_2ns"].size(); i++) mapSections.cm1_2ns.push_back(j["mapSections"]["cm1_2ns"][i]);
    for (uint64_t i=0; i<j["mapSections"]["cm2_2ns"].size(); i++) mapSections.cm2_2ns.push_back(j["mapSections"]["cm2_2ns"][i]);
    for (uint64_t i=0; i<j["mapSections"]["cm3_2ns"].size(); i++) mapSections.cm3_2ns.push_back(j["mapSections"]["cm3_2ns"][i]);
    for (uint64_t i=0; i<j["mapSections"]["q_2ns"].size(); i++) mapSections.q_2ns.push_back(j["mapSections"]["q_2ns"][i]);
    for (uint64_t i=0; i<j["mapSections"]["exps_withq_2ns"].size(); i++) mapSections.exps_withq_2ns.push_back(j["mapSections"]["exps_withq_2ns"][i]);
    for (uint64_t i=0; i<j["mapSections"]["exps_withoutq_2ns"].size(); i++) mapSections.exps_withoutq_2ns.push_back(j["mapSections"]["exps_withoutq_2ns"][i]);

    mapSectionsN.cm1_n = j["mapSectionsN"]["cm1_n"];
    mapSectionsN.cm2_n = j["mapSectionsN"]["cm2_n"];
    mapSectionsN.cm3_n = j["mapSectionsN"]["cm3_n"];
    mapSectionsN.exps_withq_n = j["mapSectionsN"]["exps_withq_n"];
    mapSectionsN.exps_withoutq_n = j["mapSectionsN"]["exps_withoutq_n"];
    mapSectionsN.cm1_2ns = j["mapSectionsN"]["cm1_2ns"];
    mapSectionsN.cm2_2ns = j["mapSectionsN"]["cm2_2ns"];
    mapSectionsN.cm3_2ns = j["mapSectionsN"]["cm3_2ns"];
    mapSectionsN.q_2ns = j["mapSectionsN"]["q_2ns"];
    mapSectionsN.exps_withq_2ns = j["mapSectionsN"]["exps_withq_2ns"];
    mapSectionsN.exps_withoutq_2ns = j["mapSectionsN"]["exps_withoutq_2ns"];

    mapSectionsN1.cm1_n = j["mapSectionsN1"]["cm1_n"];
    mapSectionsN1.cm2_n = j["mapSectionsN1"]["cm2_n"];
    mapSectionsN1.cm3_n = j["mapSectionsN1"]["cm3_n"];
    mapSectionsN1.exps_withq_n = j["mapSectionsN1"]["exps_withq_n"];
    mapSectionsN1.exps_withoutq_n = j["mapSectionsN1"]["exps_withoutq_n"];
    mapSectionsN1.cm1_2ns = j["mapSectionsN1"]["cm1_2ns"];
    mapSectionsN1.cm2_2ns = j["mapSectionsN1"]["cm2_2ns"];
    mapSectionsN1.cm3_2ns = j["mapSectionsN1"]["cm3_2ns"];
    mapSectionsN1.q_2ns = j["mapSectionsN1"]["q_2ns"];
    mapSectionsN1.exps_withq_2ns = j["mapSectionsN1"]["exps_withq_2ns"];
    mapSectionsN1.exps_withoutq_2ns = j["mapSectionsN1"]["exps_withoutq_2ns"];

    mapSectionsN3.cm1_n = j["mapSectionsN3"]["cm1_n"];
    mapSectionsN3.cm2_n = j["mapSectionsN3"]["cm2_n"];
    mapSectionsN3.cm3_n = j["mapSectionsN3"]["cm3_n"];
    mapSectionsN3.exps_withq_n = j["mapSectionsN3"]["exps_withq_n"];
    mapSectionsN3.exps_withoutq_n = j["mapSectionsN3"]["exps_withoutq_n"];
    mapSectionsN3.cm1_2ns = j["mapSectionsN3"]["cm1_2ns"];
    mapSectionsN3.cm2_2ns = j["mapSectionsN3"]["cm2_2ns"];
    mapSectionsN3.cm3_2ns = j["mapSectionsN3"]["cm3_2ns"];
    mapSectionsN3.q_2ns = j["mapSectionsN3"]["q_2ns"];
    mapSectionsN3.exps_withq_2ns = j["mapSectionsN3"]["exps_withq_2ns"];
    mapSectionsN3.exps_withoutq_2ns = j["mapSectionsN3"]["exps_withoutq_2ns"];

    for (uint64_t i=0; i<j["varPolMap"].size(); i++)
    {
        VarPolMap map;
        map.section = j["varPolMap"][i]["section"];
        map.sectionPos = j["varPolMap"][i]["sectionPos"];
        map.dim = j["varPolMap"][i]["dim"];
        varPolMap.push_back(map);
    }

}

void StarkInfo::getPol(void * pAddress, uint64_t idPol, PolInfo &polInfo)
{
    polInfo.map = varPolMap[idPol];
    polInfo.N = mapDeg.getSection(polInfo.map.section);
    polInfo.offset = mapOffsets.getSection(polInfo.map.section);
    polInfo.offset += polInfo.map.sectionPos;
    polInfo.size = mapSectionsN.getSection(polInfo.map.section);
    polInfo.pAddress = ((Goldilocks::Element *)pAddress) + polInfo.offset;
}