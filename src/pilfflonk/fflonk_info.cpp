#include "fflonk_info.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"

namespace FflonkInfo {

    FflonkInfo::FflonkInfo(AltBn128::Engine &_E, std::string file): E(_E)
    {   
        // Load contents from json file
        zklog.info("> Reading Fflonk Info");
        TimerStart(FFLONK_INFO_LOAD);
        json fflonkInfoJson;
        file2json(file, fflonkInfoJson);
        load(fflonkInfoJson);
        TimerStopAndLog(FFLONK_INFO_LOAD);
    }

    void FflonkInfo::load(json j)
    {
        nConstants = j["nConstants"];
        nPublics = j["nPublics"];
        nBitsZK = j["nBitsZK"];
        nCm1 = j["nCm1"];
        nCm2 = j["nCm2"];
        nCm3 = j["nCm3"];
        qDim = j["qDim"];
        qDeg = j["qDeg"];

        maxPolsOpenings = j["maxPolsOpenings"];

        mapDeg.section[cm1_n] = j["mapDeg"]["cm1_n"];
        mapDeg.section[cm2_n] = j["mapDeg"]["cm2_n"];
        mapDeg.section[cm3_n] = j["mapDeg"]["cm3_n"];
        mapDeg.section[eSection::tmpExp_n] = j["mapDeg"]["tmpExp_n"];
        mapDeg.section[cm1_2ns] = j["mapDeg"]["cm1_2ns"];
        mapDeg.section[cm2_2ns] = j["mapDeg"]["cm2_2ns"];
        mapDeg.section[cm3_2ns] = j["mapDeg"]["cm3_2ns"];
        mapDeg.section[q_2ns] = j["mapDeg"]["q_2ns"];

        for (uint64_t i = 0; i < j["mapSections"]["cm1_n"].size(); i++)
            mapSections.section[cm1_n].push_back(j["mapSections"]["cm1_n"][i]);
        for (uint64_t i = 0; i < j["mapSections"]["cm2_n"].size(); i++)
            mapSections.section[cm2_n].push_back(j["mapSections"]["cm2_n"][i]);
        for (uint64_t i = 0; i < j["mapSections"]["cm3_n"].size(); i++)
            mapSections.section[cm3_n].push_back(j["mapSections"]["cm3_n"][i]);
        for (uint64_t i = 0; i < j["mapSections"]["tmpExp_n"].size(); i++)
            mapSections.section[eSection::tmpExp_n].push_back(j["mapSections"]["tmpExp_n"][i]);
        for (uint64_t i = 0; i < j["mapSections"]["cm1_2ns"].size(); i++)
            mapSections.section[cm1_2ns].push_back(j["mapSections"]["cm1_2ns"][i]);
        for (uint64_t i = 0; i < j["mapSections"]["cm2_2ns"].size(); i++)
            mapSections.section[cm2_2ns].push_back(j["mapSections"]["cm2_2ns"][i]);
        for (uint64_t i = 0; i < j["mapSections"]["cm3_2ns"].size(); i++)
            mapSections.section[cm3_2ns].push_back(j["mapSections"]["cm3_2ns"][i]);
        for (uint64_t i = 0; i < j["mapSections"]["q_2ns"].size(); i++)
            mapSections.section[q_2ns].push_back(j["mapSections"]["q_2ns"][i]);

        mapSectionsN.section[cm1_n] = j["mapSectionsN"]["cm1_n"];
        mapSectionsN.section[cm2_n] = j["mapSectionsN"]["cm2_n"];
        mapSectionsN.section[cm3_n] = j["mapSectionsN"]["cm3_n"];
        mapSectionsN.section[eSection::tmpExp_n] = j["mapSectionsN"]["tmpExp_n"];
        mapSectionsN.section[cm1_2ns] = j["mapSectionsN"]["cm1_2ns"];
        mapSectionsN.section[cm2_2ns] = j["mapSectionsN"]["cm2_2ns"];
        mapSectionsN.section[cm3_2ns] = j["mapSectionsN"]["cm3_2ns"];
        mapSectionsN.section[q_2ns] = j["mapSectionsN"]["q_2ns"];

        for (uint64_t i = 0; i < j["varPolMap"].size(); i++)
        {
            VarPolMap map;
            map.section = string2section(j["varPolMap"][i]["section"]);
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

        for (uint64_t i = 0; i < j["evMap"].size(); i++)
        {
            EvMap map;
            map.setType(j["evMap"][i]["type"]);
            map.id = j["evMap"][i]["id"];
            map.prime = j["evMap"][i]["prime"];
            evMap.push_back(map);
        }

        for (uint64_t i = 0; i < j["publics"].size(); i++) 
        {
            Publics pub;
            pub.polType = j["publics"][i]["polType"];
            pub.polId = j["publics"][i]["polId"];
            pub.idx = j["publics"][i]["idx"];
            pub.id = j["publics"][i]["id"];
            pub.name = j["publics"][i]["name"];
            publics.push_back(pub);
        }

        for(uint64_t i = 0; i < j["publicsCode"].size(); i++) {
                Step publicsCodeStep;
                publicsCodeStep.tmpUsed = j["publicsCode"][i]["tmpUsed"];
            for (uint64_t l = 0; l < j["publicsCode"][i]["first"].size(); l++)
            {
                StepOperation op;
                op.setOperation(j["publicsCode"][i]["first"][l]["op"]);           // Mandatory field
                op.dest.setType(j["publicsCode"][i]["first"][l]["dest"]["type"]); // Mandatory field
                op.dest.id = j["publicsCode"][i]["first"][l]["dest"]["id"];       // Mandatory field
                if (j["publicsCode"][i]["first"][l]["dest"].contains("prime"))
                    op.dest.prime = j["publicsCode"][i]["first"][l]["dest"]["prime"];
                else
                    op.dest.prime = false;
                if (j["publicsCode"][i]["first"][l]["dest"].contains("p"))
                    op.dest.p = j["publicsCode"][i]["first"][l]["dest"]["p"];
                else
                    op.dest.p = 0;
                for (uint64_t k = 0; k < j["publicsCode"][i]["first"][l]["src"].size(); k++)
                {
                    StepType ty;
                    ty.setType(j["publicsCode"][i]["first"][l]["src"][k]["type"]); // Mandatory field
                    if (j["publicsCode"][i]["first"][l]["src"][k].contains("id"))
                        ty.id = j["publicsCode"][i]["first"][l]["src"][k]["id"];
                    else
                        ty.id = 0; // Mandatory field
                    if (j["publicsCode"][i]["first"][l]["src"][k].contains("prime"))
                        ty.prime = j["publicsCode"][i]["first"][l]["src"][k]["prime"];
                    else
                        ty.prime = false;
                    if (j["publicsCode"][i]["first"][l]["src"][k].contains("p"))
                        ty.p = j["publicsCode"][i]["first"][l]["src"][k]["p"];
                    else
                        ty.p = 0;
                    if (j["publicsCode"][i]["first"][l]["src"][k].contains("value"))
                        ty.value = j["publicsCode"][i]["first"][l]["src"][k]["value"];
                    op.src.push_back(ty);
                }
                publicsCodeStep.first.push_back(op);
                publicsCode.push_back(publicsCodeStep);
            }
        }

        step2prev.tmpUsed = j["step2prev"]["tmpUsed"];
        for (uint64_t i = 0; i < j["publicsCode"][i]["first"].size(); i++)
        {
            StepOperation op;
            op.setOperation(j["publicsCode"][i]["first"][i]["op"]);           // Mandatory field
            op.dest.setType(j["publicsCode"][i]["first"][i]["dest"]["type"]); // Mandatory field
            op.dest.id = j["publicsCode"][i]["first"][i]["dest"]["id"];       // Mandatory field
            if (j["publicsCode"][i]["first"][i]["dest"].contains("prime"))
                op.dest.prime = j["publicsCode"][i]["first"][i]["dest"]["prime"];
            else
                op.dest.prime = false;
            if (j["publicsCode"][i]["first"][i]["dest"].contains("p"))
                op.dest.p = j["publicsCode"][i]["first"][i]["dest"]["p"];
            else
                op.dest.p = 0;
            for (uint64_t k = 0; k < j["publicsCode"][i]["first"][i]["src"].size(); k++)
            {
                StepType ty;
                ty.setType(j["publicsCode"][i]["first"][i]["src"][k]["type"]); // Mandatory field
                if (j["publicsCode"][i]["first"][i]["src"][k].contains("id"))
                    ty.id = j["publicsCode"][i]["first"][i]["src"][k]["id"];
                else
                    ty.id = 0; // Mandatory field
                if (j["publicsCode"][i]["first"][i]["src"][k].contains("prime"))
                    ty.prime = j["publicsCode"][i]["first"][i]["src"][k]["prime"];
                else
                    ty.prime = false;
                if (j["publicsCode"][i]["first"][i]["src"][k].contains("p"))
                    ty.p = j["publicsCode"][i]["first"][i]["src"][k]["p"];
                else
                    ty.p = 0;
                if (j["publicsCode"][i]["first"][i]["src"][k].contains("value"))
                    ty.value = j["publicsCode"][i]["first"][i]["src"][k]["value"];
                op.src.push_back(ty);
            }
            step2prev.first.push_back(op);
        }

        step3prev.tmpUsed = j["step3prev"]["tmpUsed"];
        for (uint64_t i = 0; i < j["step3prev"]["first"].size(); i++)
        {
            StepOperation op;
            op.setOperation(j["step3prev"]["first"][i]["op"]);           // Mandatory field
            op.dest.setType(j["step3prev"]["first"][i]["dest"]["type"]); // Mandatory field
            op.dest.id = j["step3prev"]["first"][i]["dest"]["id"];       // Mandatory field
            if (j["step3prev"]["first"][i]["dest"].contains("prime"))
                op.dest.prime = j["step3prev"]["first"][i]["dest"]["prime"];
            else
                op.dest.prime = false;
            if (j["step3prev"]["first"][i]["dest"].contains("p"))
                op.dest.p = j["step3prev"]["first"][i]["dest"]["p"];
            else
                op.dest.p = 0;
            for (uint64_t k = 0; k < j["step3prev"]["first"][i]["src"].size(); k++)
            {
                StepType ty;
                ty.setType(j["step3prev"]["first"][i]["src"][k]["type"]); // Mandatory field
                if (j["step3prev"]["first"][i]["src"][k].contains("id"))
                    ty.id = j["step3prev"]["first"][i]["src"][k]["id"];
                else
                    ty.id = 0; // Mandatory field
                if (j["step3prev"]["first"][i]["src"][k].contains("prime"))
                    ty.prime = j["step3prev"]["first"][i]["src"][k]["prime"];
                else
                    ty.prime = false;
                if (j["step3prev"]["first"][i]["src"][k].contains("p"))
                    ty.p = j["step3prev"]["first"][i]["src"][k]["p"];
                else
                    ty.p = 0;
                if (j["step3prev"]["first"][i]["src"][k].contains("value"))
                    ty.value = j["step3prev"]["first"][i]["src"][k]["value"];
                op.src.push_back(ty);
            }
            step3prev.first.push_back(op);
        }

        step3.tmpUsed = j["step3"]["tmpUsed"];
        for (uint64_t i = 0; i < j["step3"]["first"].size(); i++)
        {
            StepOperation op;
            op.setOperation(j["step3"]["first"][i]["op"]);           // Mandatory field
            op.dest.setType(j["step3"]["first"][i]["dest"]["type"]); // Mandatory field
            op.dest.id = j["step3"]["first"][i]["dest"]["id"];       // Mandatory field
            if (j["step3"]["first"][i]["dest"].contains("prime"))
                op.dest.prime = j["step3"]["first"][i]["dest"]["prime"];
            else
                op.dest.prime = false;
            if (j["step3"]["first"][i]["dest"].contains("p"))
                op.dest.p = j["step3"]["first"][i]["dest"]["p"];
            else
                op.dest.p = 0;
            for (uint64_t k = 0; k < j["step3"]["first"][i]["src"].size(); k++)
            {
                StepType ty;
                ty.setType(j["step3"]["first"][i]["src"][k]["type"]); // Mandatory field
                if (j["step3"]["first"][i]["src"][k].contains("id"))
                    ty.id = j["step3"]["first"][i]["src"][k]["id"];
                else
                    ty.id = 0; // Mandatory field
                if (j["step3"]["first"][i]["src"][k].contains("prime"))
                    ty.prime = j["step3"]["first"][i]["src"][k]["prime"];
                else
                    ty.prime = false;
                if (j["step3"]["first"][i]["src"][k].contains("p"))
                    ty.p = j["step3"]["first"][i]["src"][k]["p"];
                else
                    ty.p = 0;
                if (j["step3"]["first"][i]["src"][k].contains("value"))
                    ty.value = j["step3"]["first"][i]["src"][k]["value"];
                op.src.push_back(ty);
            }
            step3.first.push_back(op);
        }

        step42ns.tmpUsed = j["step42ns"]["tmpUsed"];
        for (uint64_t i = 0; i < j["step42ns"]["first"].size(); i++)
        {
            StepOperation op;
            op.setOperation(j["step42ns"]["first"][i]["op"]);           // Mandatory field
            op.dest.setType(j["step42ns"]["first"][i]["dest"]["type"]); // Mandatory field
            op.dest.id = j["step42ns"]["first"][i]["dest"]["id"];       // Mandatory field
            if (j["step42ns"]["first"][i]["dest"].contains("prime"))
                op.dest.prime = j["step42ns"]["first"][i]["dest"]["prime"];
            else
                op.dest.prime = false;
            if (j["step42ns"]["first"][i]["dest"].contains("p"))
                op.dest.p = j["step42ns"]["first"][i]["dest"]["p"];
            else
                op.dest.p = 0;
            for (uint64_t k = 0; k < j["step42ns"]["first"][i]["src"].size(); k++)
            {
                StepType ty;
                ty.setType(j["step42ns"]["first"][i]["src"][k]["type"]); // Mandatory field
                if (j["step42ns"]["first"][i]["src"][k].contains("id"))
                    ty.id = j["step42ns"]["first"][i]["src"][k]["id"];
                else
                    ty.id = 0; // Mandatory field
                if (j["step42ns"]["first"][i]["src"][k].contains("prime"))
                    ty.prime = j["step42ns"]["first"][i]["src"][k]["prime"];
                else
                    ty.prime = false;
                if (j["step42ns"]["first"][i]["src"][k].contains("p"))
                    ty.p = j["step42ns"]["first"][i]["src"][k]["p"];
                else
                    ty.p = 0;
                if (j["step42ns"]["first"][i]["src"][k].contains("value"))
                    ty.value = j["step42ns"]["first"][i]["src"][k]["value"];
                op.src.push_back(ty);
            }
            step42ns.first.push_back(op);
        }



        for (uint64_t i = 0; i < j["exps_n"].size(); i++)
        {
            uint64_t value = 0;
            if (!j["exps_n"][i].is_null())
            {
                value = j["exps_n"][i];
            }
            exps_n.push_back(value);
        }

        for (uint64_t i = 0; i < j["q_2ns"].size(); i++)
        {
            uint64_t value = 0;
            if (!j["q_2ns"][i].is_null())
            {
                value = j["q_2ns"][i];
            }
            q_2nsVector.push_back(value);
        }

        for (uint64_t i = 0; i < j["tmpExp_n"].size(); i++)
        {
            uint64_t value = 0;
            if (!j["tmpExp_n"][i].is_null())
            {
                value = j["tmpExp_n"][i];
            }
            tmpExp_n.push_back(value);
        }

        for (auto it =  j["exp2pol"].begin(); it != j["exp2pol"].end(); ++it) {
            std::string key = it.key(); 
            uint64_t value = it.value();
            exp2pol.insert(std::pair(key,value));
        }

    }


    eSection string2section(const std::string s)
    {
        if (s == "cm1_n")
            return cm1_n;
        if (s == "cm2_n")
            return cm2_n;
        if (s == "cm3_n")
            return cm3_n;
        if (s == "tmpExp_n")
            return tmpExp_n;
        if (s == "cm1_2ns")
            return cm1_2ns;
        if (s == "cm2_2ns")
            return cm2_2ns;
        if (s == "cm3_2ns")
            return cm3_2ns;
        if (s == "q_2ns")
            return q_2ns;
        zklog.error("string2section() found invalid string=" + s);
        exitProcess();
        exit(-1);
    }

    std::string FflonkInfo::getSectionName(eSection section) {
        switch (section) {
            case cm1_n:
                return "cm1_n";
            case cm1_2ns:
                return "cm1_2ns";
            case cm2_n:
                return "cm2_n";
            case cm2_2ns:
                return "cm2_2ns";
            case cm3_n:
                return "cm3_n";
            case cm3_2ns:
                return "cm3_2ns";
            case eSection::tmpExp_n:
                return "tmpExp_n";
            case q_2ns:
                return "q_2ns";
            default:
                zklog.error("Invalid section");
                exitProcess();
                exit(-1);
        }
    }

    PolInfo FflonkInfo::getPolInfo(u_int64_t polId) {
        eSection section = varPolMap[polId].section;

        PolInfo polInfo;

        polInfo.sectionName = getSectionName(section);
        polInfo.id = varPolMap[polId].sectionPos;
        polInfo.nPols = mapSections.section[section].size();

        return polInfo;
    }
}