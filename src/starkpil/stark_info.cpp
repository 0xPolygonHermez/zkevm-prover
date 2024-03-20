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

    if(j.contains("nSubAirValues")) {
        nSubProofValues = j["nSubAirValues"];
    } else {
        nSubProofValues = 0;
    }

    if(j.contains("numChallenges")) {
        nChallenges = 0;
        for(uint64_t i = 0; i < j["numChallenges"].size(); i++) {
            numChallenges.push_back(j["numChallenges"][i]);
            nChallenges += numChallenges[i];
            if(i == 0) {
                stageChallengeIndex.push_back(0);
            } else {
                stageChallengeIndex.push_back(stageChallengeIndex[i - 1] + numChallenges[i - 1]);
            }
        }
        qChallengeIndex = nChallenges++;
        xiChallengeIndex = nChallenges++;
        fri1ChallengeIndex = nChallenges++;
        fri2ChallengeIndex = nChallenges++;
    } else {
        numChallenges.push_back(0);
        numChallenges.push_back(2);
        numChallenges.push_back(2);
        stageChallengeIndex.push_back(0);
        stageChallengeIndex.push_back(0);
        stageChallengeIndex.push_back(2);
        qChallengeIndex = 4;
        xiChallengeIndex = 7;
        fri1ChallengeIndex = 5;
        fri2ChallengeIndex = 6; 
        nChallenges = 8;
    }

    nStages = numChallenges.size();

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
    
    std::string ext = pil2 ? "_ext" : "_2ns";

    for(uint64_t i = 1; i <= nStages + 1; i++) {
        string s = i == nStages + 1 && pil2 ? "Q" : to_string(i);
        string step = "cm" + s + "_n";
        string stepExt = "cm" + s + ext;
        
        eSection section = string2section("cm" + to_string(i) + "_n");
        eSection sectionExt = string2section("cm" + to_string(i) + "_2ns");

        mapSectionsN.section[section] = j["mapSectionsN"][step];
        mapSectionsN.section[sectionExt] = j["mapSectionsN"][stepExt];

        mapOffsets.section[section] = j["mapOffsets"][step];
        mapOffsets.section[sectionExt] = j["mapOffsets"][stepExt];
    }

    mapSectionsN.section[tmpExp_n] = j["mapSectionsN"]["tmpExp_n"];
    mapOffsets.section[tmpExp_n] = j["mapOffsets"]["tmpExp_n"];

    mapSectionsN.section[q_2ns] = j["mapSectionsN"]["q" + ext];
    mapOffsets.section[q_2ns] = j["mapOffsets"]["q" + ext];

    mapSectionsN.section[f_2ns] = j["mapSectionsN"]["f" + ext];
    mapOffsets.section[f_2ns] = j["mapOffsets"]["f" + ext];

    for (uint64_t i = 0; i < j["evMap"].size(); i++)
    {
        EvMap map;
        map.setType(j["evMap"][i]["type"]);
        map.id = j["evMap"][i]["id"];
        if(pil2) {
            map.prime = j["evMap"][i]["prime"];
        } else {
            map.prime = j["evMap"][i]["prime"] ? 1 : 0; 
        }
        evMap.push_back(map);
    }

    if(pil2) {
        for (uint64_t i = 0; i < j["cmPolsMap"].size(); i++) 
        {
            CmPolMap map;
            map.stage = j["cmPolsMap"][i]["stage"];
            map.stageNum = j["cmPolsMap"][i]["stageNum"];
            map.name = j["cmPolsMap"][i]["name"];
            map.dim = j["cmPolsMap"][i]["dim"];
            map.imPol = j["cmPolsMap"][i]["imPol"];
            map.stagePos = j["cmPolsMap"][i]["stagePos"];
            map.stageId = j["cmPolsMap"][i]["stageId"];
            cmPolsMap.push_back(map);
        }

        for (uint64_t i = 0; i < j["symbolsStage"].size(); i++) 
        {
            symbolsStage.push_back(std::vector<Symbol>());
            for(uint64_t k = 0; k < j["symbolsStage"][i].size(); k++) 
            {
                Symbol symbol;
                symbol.setSymbol(j["symbolsStage"][i][k]);
                symbolsStage[i].push_back(symbol);
            }
        }
        
        for(uint64_t i = 0; i < nStages; ++i) {
            std::string stage = "stage" + to_string(i + 1);
            stageCodeSymbols.push_back(std::vector<Symbol>());
            expressionsCodeSymbols.push_back(std::vector<ExpressionCodeSymbol>());
            for(uint64_t k = 0; k < j["code"][stage]["symbolsCalculated"].size(); k++) {
                Symbol symbol;
                symbol.setSymbol(j["code"][stage]["symbolsCalculated"][k]);
                stageCodeSymbols[i].push_back(symbol);
            }
        }

        for(uint64_t i = 0; i < j["expressionsCode"].size(); i++) {
            ExpressionCodeSymbol expSymbol;
            expSymbol.stage = j["expressionsCode"][i]["stage"];
            expSymbol.expId = j["expressionsCode"][i]["expId"];
            for(uint64_t k = 0; k < j["expressionsCode"][i]["code"]["symbolsUsed"].size(); k++) {
                Symbol symbol; 
                symbol.setSymbol(j["expressionsCode"][i]["code"]["symbolsUsed"][k]);
                expSymbol.symbolsUsed.push_back(symbol);
            }
            expressionsCodeSymbols[expSymbol.stage - 1].push_back(expSymbol);
        }

        for(uint64_t i = 0; i < j["hints"].size(); i++) {
            Hint hint;
            if(j["hints"][i]["name"] == string("public")) continue;
            hint.type = string2hintType(j["hints"][i]["name"]);
            
            uint64_t stage = j["hints"][i]["dest"][0]["stage"];
            
            hint.fields = std::vector<string>();
            for(uint64_t k = 0; k < j["hints"][i]["fields"].size(); k++) {
                std::string field = j["hints"][i]["fields"][k];
                hint.fields.push_back(field);
                Symbol symbol;
                symbol.setSymbol(j["hints"][i][field]);
                hint.fieldSymbols.insert(pair(field,symbol));
            }

            hint.destSymbols = std::vector<Symbol>();
            for(uint64_t k = 0; k < j["hints"][i]["dest"].size(); k++) {
                Symbol symbol;
                symbol.setSymbol(j["hints"][i]["dest"][k]);
                hint.destSymbols.push_back(symbol);
            }

            hint.symbols = std::vector<Symbol>();
            for(uint64_t k = 0; k < j["hints"][i]["symbols"].size(); k++) {
                Symbol symbol;
                symbol.setSymbol(j["hints"][i]["symbols"][k]);
                hint.symbols.push_back(symbol);
            }

            hints[stage].push_back(hint);
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
        

        uint64_t indxStage2 = 0;
        uint64_t indxStage3 = 0;

        for (uint64_t i = 0; i < j["puCtx"].size(); i++)
        {
            Hint hintH1H2;
            hintH1H2.type = hintType::h1h2;

            Symbol fExp;
            fExp.setSymbol(2, j["puCtx"][i]["fExpId"]);
            hintH1H2.fields.push_back("fExpId");           
            hintH1H2.fieldSymbols.insert(pair("fExpId", fExp));

            Symbol tExp;
            tExp.setSymbol(2, j["puCtx"][i]["tExpId"]);
            hintH1H2.fields.push_back("tExpId");           
            hintH1H2.fieldSymbols.insert(pair("tExpId", tExp));

            Symbol h1;
            Symbol h2;
            h1.setSymbol(2, j["puCtx"][i]["h1Id"]);
            h2.setSymbol(2, j["puCtx"][i]["h2Id"]);
            hintH1H2.destSymbols.push_back(h1);
            hintH1H2.destSymbols.push_back(h2);
  
            hintH1H2.index = indxStage2;
            indxStage2 += 4;

            hints[2].push_back(hintH1H2);

            Hint hintGProd;
            hintGProd.type = hintType::gprod;

            Symbol num;
            num.setSymbol(2, j["puCtx"][i]["numId"]);
            hintGProd.fields.push_back("numId");
            hintGProd.fieldSymbols.insert(pair("numId", num));

            Symbol den;
            den.setSymbol(2, j["puCtx"][i]["denId"]);
            hintGProd.fields.push_back("denId");
            hintGProd.fieldSymbols.insert(pair("denId", den));

            Symbol z;
            z.setSymbol(2, j["puCtx"][i]["zId"]);
            hintGProd.destSymbols.push_back(z);

            hintGProd.index = indxStage3;
            indxStage3 += 3;

            hints[3].push_back(hintGProd);
        }

        for (uint64_t i = 0; i < j["peCtx"].size(); i++) 
        {
            Hint hintGProd;
            hintGProd.type = hintType::gprod;

            Symbol num;
            num.setSymbol(2, j["peCtx"][i]["numId"]);
            hintGProd.fields.push_back("numId");
            hintGProd.fieldSymbols.insert(pair("numId", num));

            Symbol den;
            den.setSymbol(2, j["peCtx"][i]["denId"]);
            hintGProd.fields.push_back("denId");
            hintGProd.fieldSymbols.insert(pair("denId", den));

            Symbol z;
            z.setSymbol(2, j["peCtx"][i]["zId"]);
            hintGProd.destSymbols.push_back(z);

            hintGProd.index = indxStage3;
            indxStage3 += 3;

            hints[3].push_back(hintGProd);
        }

        for (uint64_t i = 0; i < j["ciCtx"].size(); i++) 
        {
            Hint hintGProd;
            hintGProd.type = hintType::gprod;

            Symbol num;
            num.setSymbol(2, j["ciCtx"][i]["numId"]);
            hintGProd.fields.push_back("numId");
            hintGProd.fieldSymbols.insert(pair("numId", num));

            Symbol den;
            den.setSymbol(2, j["ciCtx"][i]["denId"]);
            hintGProd.fields.push_back("denId");
            hintGProd.fieldSymbols.insert(pair("denId", den));

            Symbol z;
            z.setSymbol(2, j["ciCtx"][i]["zId"]);
            hintGProd.destSymbols.push_back(z);

            hintGProd.index = indxStage3;
            indxStage3 += 3;

            hints[3].push_back(hintGProd);
        }

        for (auto it =  j["exp2pol"].begin(); it != j["exp2pol"].end(); ++it) {
            uint64_t key = std::stoull(it.key()); 
            uint64_t value = it.value();
            exp2pol.insert(pair(key,value));
        }
    }
}

uint64_t StarkInfo::getPolinomialRef(std::string type, uint64_t index) {

    if(pil2) {
        return index;
    } else {
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
}

Polinomial StarkInfo::getPolinomial(Goldilocks::Element *pAddress, uint64_t idPol, uint64_t deg)
{
    if(pil2) {
        CmPolMap polInfo = cmPolsMap[idPol];
        uint64_t dim = polInfo.dim;
        uint64_t N = (1 << starkStruct.nBits);
        string stage = polInfo.stage == string("cmQ") ? "cm" + to_string(nStages + 1) : polInfo.stage;
        eSection section = N == deg ? string2section(stage + "_n") : string2section(stage + "_2ns");
        uint64_t nCols = mapSectionsN.section[section];
        uint64_t offset = mapOffsets.section[section];
        offset += polInfo.stagePos;
        return Polinomial(&pAddress[offset], deg, dim, nCols, std::to_string(idPol));
    } else {
        VarPolMap polInfo = varPolMap[idPol];
        uint64_t dim = polInfo.dim;
        uint64_t offset = mapOffsets.section[polInfo.section];
        uint64_t nCols = mapSectionsN.section[polInfo.section];
        offset += polInfo.sectionPos;
        return Polinomial(&pAddress[offset], deg, dim, nCols, std::to_string(idPol));
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
    if(s == "subproofvalue")
        return subproofvalue;
    if(s == "challenge")
        return challenge;
    if(s == "number")
        return number;
    zklog.error("string2opType() found invalid string=" + s);
    exitProcess();
    exit(-1);
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

hintType string2hintType(const string s)
{
    if(s == "h1h2")
        return h1h2;
    if(s == "gprod")
        return gprod;
    if(s == "public")
        return publicValue;
    if(s == "gsum") 
        return gsum;
    if(s == "subproofvalue")
        return subproofValue;
    zklog.error("string2hintType() found invalid string=" + s);
    exitProcess();
    exit(-1);
}