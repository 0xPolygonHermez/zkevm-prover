#include "fflonk_shkey.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"

FflonkShKey::FflonkShKey(AltBn128::Engine &_E, std::string file): E(_E)
{   
    // Load contents from json file
    zklog.info("> Reading Fflonk Info");
    TimerStart(FFLONK_SHKEY_LOAD);
    json fflonkShKeyJson;
    file2json(file, fflonkShKeyJson);
    load(fflonkShKeyJson);
    TimerStopAndLog(FFLONK_SHKEY_LOAD);
}

void FflonkShKey::load(json j)
{
    power = j["power"];
    powerW = j["powerW"];
    maxQDegree = j["maxQDegree"];
    nPublics = j["nPublics"];

    for(u_int32_t i = 0; i <  j["roots"].size(); ++i) {
        std::string name = j["roots"][i];
        omegas[name] = E.fr.set(j[name]);
    }

    std::vector<std::string> polsNamesStage0;
    for(u_int32_t i = 0; i <  j["polsNamesStage"][0]; ++i) {
        std::string name = j["polsNamesStage"][0][i];

        polsNamesStage0.push_back(name);
    }

    std::vector<std::string> polsNamesStage1;
    for(u_int32_t i = 0; i <  j["polsNamesStage"][1]; ++i) {
        std::string name = j["polsNamesStage"][1][i];

        polsNamesStage1.push_back(name);
    }


    std::vector<std::string> polsNamesStage2;
    for(u_int32_t i = 0; i <  j["polsNamesStage"][2]; ++i) {
        std::string name = j["polsNamesStage"][2][i];

        polsNamesStage2.push_back(name);
    }


    std::vector<std::string> polsNamesStage3;
    for(u_int32_t i = 0; i <  j["polsNamesStage"][3]; ++i) {
        std::string name = j["polsNamesStage"][3][i];

        polsNamesStage3.push_back(name);
    }

    polsNamesStage[0] = polsNamesStage0;
    polsNamesStage[1] = polsNamesStage1;
    polsNamesStage[2] = polsNamesStage2;
    polsNamesStage[3] = polsNamesStage3;

    // Load f
    for(u_int32_t i = 0; i < j["f"].size(); ++i) {
        F f_i;
        f_i.index = j["f"][i]["index"];
        f_i.degree = j["f"][i]["degree"];

        for(u_int32_t l = 0; l < j["f"][i]["pols"].size(); ++l) {
            std::string name = j["f"][i]["pols"][l];
            f_i.pols.push_back(name);
        }

        for(u_int32_t l = 0; l < j["f"][i]["openingPoints"].size(); ++l) {
            uint64_t point = j["f"][i]["openingPoints"][l];
            f_i.openingPoints.push_back(point);
        }

        for(u_int32_t l = 0; l < j["f"][i]["stages"].size(); ++l) {
            FStage stage;
            stage.stage = j["f"][i]["stages"][l]["stage"];

            for(u_int32_t k = 0; k < j["f"][i]["stages"][l]["pols"].size(); ++k) {
                FStagePol stageInfo;
                stageInfo.name = j["f"][i]["stages"][l]["pols"][k]["name"];
                stageInfo.degree = j["f"][i]["stages"][l]["pols"][k]["degree"];

                stage.pols.push_back(stageInfo);
            }

            f_i.stages.push_back(stage);
        }

        f.push_back(f_i);
    }
}