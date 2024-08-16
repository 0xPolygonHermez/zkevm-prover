#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"

class ArgumentParser {
private:
    vector <string> arguments;
public:
    ArgumentParser (int &argc, char **argv)
    {
        for (int i=1; i < argc; ++i)
            arguments.push_back(string(argv[i]));
    }

    string getArgumentValue (const string argshort, const string arglong) 
    {
        for (size_t i=0; i<arguments.size(); i++) {
            if (argshort==arguments[i] || arglong==arguments[i]) {
                if (i+1 < arguments.size()) return (arguments[i+1]);
                else return "";
            }
        }
        return "";
    }

    bool argumentExists (const string argshort, const string arglong) 
    {
        bool found = false;
        for (size_t i=0; i<arguments.size(); i++) {
            if (argshort==arguments[i] || arglong==arguments[i]) {
                if (found) {
                    throw runtime_error("bctree: cannot use "+argshort+"/"+arglong+" parameter twice!");
                } else found = true;
            }
        }
        return found;
    }
};


int main(int argc, char **argv)
{
    Config config;
    
    string constFile = "";
    string starkInfoFile = "";
    string cHelpersFile = "";
    string commitPols = "";
    string publicsFile = "";

    ArgumentParser aParser (argc, argv);

    HintHandlerBuilder::registerBuilder(H1H2HintHandler::getName(), std::make_unique<H1H2HintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GProdHintHandler::getName(), std::make_unique<GProdHintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GSumHintHandler::getName(), std::make_unique<GSumHintHandlerBuilder>());
    
    try {
        //Input arguments
        if (aParser.argumentExists("-c","--const")) {
            constFile = aParser.getArgumentValue("-c", "--const");
            if (!fileExists(constFile)) throw runtime_error("constraint_checker: constants file doesn't exist ("+constFile+")");
        } else throw runtime_error("constraint_checker: constants file argument not specified <-c/--const> <const_file>");
        
        if (aParser.argumentExists("-s","--starkinfo")) {
            starkInfoFile = aParser.getArgumentValue("-s", "--starkinfo");
            if (!fileExists(starkInfoFile)) throw runtime_error("constraint_checker: starkinfo file doesn't exist ("+starkInfoFile+")");
        } else throw runtime_error("constraint_checker: starkinfo file argument not specified <-s/--stark> <starkinfo_file>");
    
        if (aParser.argumentExists("-h","--chelpers")) {
            cHelpersFile = aParser.getArgumentValue("-h","--chelpers");
            if (cHelpersFile =="") throw runtime_error("constraint_checker: chelpers file not specified");
        } else throw runtime_error("constraint_checker: chelpers file argument not specified <-h/--chelpers> <chelpers_file>");

        if (aParser.argumentExists("-t","--commit")) {
            commitPols = aParser.getArgumentValue("-t","--commit");
            if (commitPols =="") throw runtime_error("constraint_checker: commit file not specified");
        } else throw runtime_error("constraint_checker: commit file argument not specified <-t/--commit> <commit_file>");

        if (aParser.argumentExists("-p","--publics")) {
            publicsFile = aParser.getArgumentValue("-p","--publics");
            if (publicsFile=="") throw runtime_error("constraint_checker: pubics file not specified");
        } else throw runtime_error("constraint_checker: publics file argument not specified <-p/--publics> <public_file>");

        StarkInfo starkInfo(starkInfoFile);
        CHelpers cHelpers(cHelpersFile);
        ConstPols<Goldilocks::Element> constPols(starkInfo, constFile);

        void *pCommit = copyFile(commitPols, starkInfo.mapSectionsN["cm1"] * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
        void *pAddress = (void *)malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));

        uint64_t N = (1 << starkInfo.starkStruct.nBits);
        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i += 1)
        {
            std::memcpy((uint8_t*)pAddress + starkInfo.mapOffsets[std::make_pair("cm1", false)]*sizeof(Goldilocks::Element) + i*starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), 
                (uint8_t*)pCommit + i*starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), 
                starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element));
        }

        json publics;
        file2json(publicsFile, publics);

        Goldilocks::Element publicInputs[starkInfo.nPublics];

        for(uint64_t i = 0; i < starkInfo.nPublics; i++) {
            publicInputs[i] = Goldilocks::fromU64(publics[i]);
        }

        json publicStarkJson;
        for (uint64_t i = 0; i < starkInfo.nPublics; i++)
        {
            publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
        }

        nlohmann::ordered_json jProof;

        FRIProof<Goldilocks::Element> fproof(starkInfo);

        Starks<Goldilocks::Element> starks(config, pAddress, starkInfo, cHelpers, constPols, true);

        starks.genProof(fproof, &publicInputs[0]); 

        return EXIT_SUCCESS;
    } catch (const exception &e) {
        cerr << e.what() << endl;
        cerr << "usage: constraint_checker <-c|--const> <const_file> <-s|--starkinfo> <starkinfo_file> <-h|--chelpers> <chelpers_file> <-t|--commit> <commit_file> <-p|--publics> <public_file>" << endl;
        cerr << "example: constraint_checker -c zkevm.const -s zkevm.starkinfo.json -h zkevm.chelpers.bin -t zkevm.commit -p zkevm.publics.json" << endl;
        return EXIT_FAILURE;        
    }        
}