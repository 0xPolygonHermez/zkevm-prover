#include <sys/stat.h>
#include "build_const_tree.hpp"

#define BCTREE_VERSION "0.1.0.0"

using namespace std;

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

void showVersion() 
{
    cout << "bctree: version " << string(BCTREE_VERSION) << endl;
}

int main(int argc, char **argv)
{
    string constFile = "";
    string starkInfoFile = "";
    string constTreeFile = "";
    string verKeyFile = "";

    ArgumentParser aParser (argc, argv);

    try {
        //Input arguments
        if (aParser.argumentExists("-c","--const")) {
            constFile = aParser.getArgumentValue("-c", "--const");
            if (!fileExists(constFile)) throw runtime_error("bctree: constants file doesn't exist ("+constFile+")");
        } else throw runtime_error("bctree: constants input file argument not specified <-c/--const> <const_file>");
        if (aParser.argumentExists("-s","--stark")) {
            starkInfoFile = aParser.getArgumentValue("-s", "--stark");
            if (!fileExists(starkInfoFile)) throw runtime_error("bctree: starkinfo file doesn't exist ("+starkInfoFile+")");
        } else throw runtime_error("bctree: starkinfo input file argument not specified <-s/--stark> <starkinfo_file>");

        //Output arguments
        if (aParser.argumentExists("-v","--verkey")) {
            verKeyFile = aParser.getArgumentValue("-v","--verkey");
            if (verKeyFile=="") throw runtime_error("bctree: key ouput file not specified");
        } else throw runtime_error("bctree: constants tree verkey file argument not specified <-v/--verkey> <verkey_file>");
        if (aParser.argumentExists("-t","--tree")) {
            constTreeFile = aParser.getArgumentValue("-t","--tree");
            if (constTreeFile=="") throw runtime_error("bctree: constants tree output file not specified");
        }
        

        showVersion();

        buildConstTree(constFile, starkInfoFile, constTreeFile, verKeyFile);
        
        return EXIT_SUCCESS;
    } catch (const exception &e) {
        cerr << e.what() << endl;
        showVersion();
        cerr << "usage: bctree <-c|--const> <const_file> <-s|--stark> <starkinfo_file> <-t|--tree> <consttree_file> [<-v|--verkey> <verkey_file>]" << endl;
        cerr << "example: bctree -c zkevm.const -s zkevm.starkinfo.json -t zkevm.consttree -v zkevm.verkey" << endl;
        return EXIT_FAILURE;        
    }    
}
