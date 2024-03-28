#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "chelpers_steps.hpp"
#include "AllPil2Steps.hpp"

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
    config.mapConstPolsFile = false;
    config.mapConstantsTreeFile = false;
    
    string constFile = "";
    string constTreeFile = "";
    string starkInfoFile = "";
    string verKeyFile = "";
    string cHelpersFile = "";
    string commitPols = "";
    string publicsFile = "";

    ArgumentParser aParser (argc, argv);

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

        if (aParser.argumentExists("-t","--tree")) {
            constTreeFile = aParser.getArgumentValue("-t","--tree");
            if (constTreeFile=="") throw runtime_error("constraint_checker: constants tree file not specified");
        } else throw runtime_error("constraint_checker: constants tree file argument not specified <-t/--tree> <consttree_file>");
        
        if (aParser.argumentExists("-v","--verkey")) {
            verKeyFile = aParser.getArgumentValue("-v","--verkey");
            if (verKeyFile=="") throw runtime_error("constraint_checker: chelpers file not specified");
        } else throw runtime_error("constraint_checker: verkey file argument not specified <-v/--verkey> <verkey_file>");

        if (aParser.argumentExists("-h","--chelpers")) {
            cHelpersFile = aParser.getArgumentValue("-h","--chelpers");
            if (cHelpersFile =="") throw runtime_error("constraint_checker: chelpers file not specified");
        } else throw runtime_error("constraint_checker: chelpers file argument not specified <-h/--chelpers> <chelpers_file>");

        if (aParser.argumentExists("-w","--commit")) {
            commitPols = aParser.getArgumentValue("-w","--commit");
            if (commitPols =="") throw runtime_error("constraint_checker: commit file not specified");
        } else throw runtime_error("constraint_checker: commit file argument not specified <-w/--commit> <commit_file>");

        if (aParser.argumentExists("-p","--publics")) {
            publicsFile = aParser.getArgumentValue("-p","--publics");
            if (publicsFile=="") throw runtime_error("constraint_checker: pubics file not specified");
        } else throw runtime_error("constraint_checker: publics file argument not specified <-p/--publics> <public_file>");

        StarkInfo starkInfo(starkInfoFile);

        void *pCommit = copyFile(commitPols, starkInfo.nCm1 * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));

        // TODO: pAddress should be properly calculated!!!!!
        void *pAddress = (void *)malloc((starkInfo.mapTotalN + starkInfo.mapSectionsN.section[eSection::cm3_n] * (1 << starkInfo.starkStruct.nBitsExt)) * sizeof(Goldilocks::Element));

        uint64_t N = (1 << starkInfo.starkStruct.nBits);
        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i += 1)
        {
            std::memcpy((uint8_t*)pAddress + i*starkInfo.nCm1*sizeof(Goldilocks::Element), (uint8_t*)pCommit + i*starkInfo.nCm1*sizeof(Goldilocks::Element), starkInfo.nCm1*sizeof(Goldilocks::Element));
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

        uint64_t hashSize = starkInfo.starkStruct.verificationHashType == "GL" ? 4 : 1;
        FRIProof<Goldilocks::Element> fproof(starkInfo, hashSize);

        Starks<Goldilocks::Element> starks(config, {constFile, false, constTreeFile, starkInfoFile, cHelpersFile}, pAddress);
        starks.debug = true;

        // CHelpersSteps cHelpersSteps;
        AllPil2Steps cHelpersSteps;
        starks.genProof(fproof, &publicInputs[0], &cHelpersSteps); 

        // TODO PRINT REPORT

        return EXIT_SUCCESS;
    } catch (const exception &e) {
        cerr << e.what() << endl;
        cerr << "usage: constraint_checker <-c|--const> <const_file> <-s|--starkinfo> <starkinfo_file> <-t|--tree> <consttree_file> <-h|--chelpers> <chelpers_file> <-w|--commit> <commit_file> <-p|--publics> <public_file>" << endl;
        cerr << "example: constraint_checker -c zkevm.const -s zkevm.starkinfo.json -t zkevm.consttree -v zkevm.verkey -h zkevm.chelpers.bin -w zkevm.commit -p zkevm.publics.json" << endl;
        return EXIT_FAILURE;        
    }        
}