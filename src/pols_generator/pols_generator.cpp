#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <gmpxx.h>
#include "../config/definitions.hpp" // This is the only project file allowed to be included

using namespace std;
using namespace nlohmann;

// Fork namespace
const string forkNamespace = PROVER_FORK_NAMESPACE_STRING;

// Forward declaration
void file2json (ordered_json &rom, string &romFileName);
string generate(const ordered_json &pols, const string &type, const string &namespaceName);
void string2file (const string & s, const string & fileName);

int main(int argc, char **argv)
{
    cout << "Pols generator" << endl;

    // Load main.pil.json
    string pilFileName = "src/main_sm/" + forkNamespace + "/scripts/main.pil.json";
    ordered_json pols;
    file2json(pols, pilFileName);

    string directoryName = "src/main_sm/" + forkNamespace + "/pols_generated";

    string code;
    code = generate(pols, "cmP", forkNamespace);
    string2file(code, directoryName + "/commit_pols.hpp");
    code = generate(pols, "constP", forkNamespace);
    string2file(code, directoryName + "/constant_pols.hpp");

    return 0;
}

void file2json (ordered_json &pols, string &polsFileName)
{
    std::ifstream inputStream(polsFileName);
    if (!inputStream.good())
    {
        cerr << "Error: Main generator failed loading input JSON file " << polsFileName << endl;
        exit(-1);
    }
    try
    {
        pols = nlohmann::ordered_json::parse(inputStream);
    }
    catch (exception &e)
    {
        cerr << "Error: Main generator failed parsing input JSON file " << polsFileName << " exception=" << e.what() << endl;
        exit(-1);
    }
    inputStream.close();
}

void string2file (const string & s, const string & fileName)
{
    ofstream outfile;
    outfile.open(fileName);
    outfile << s << endl;
    outfile.close();
}

string filter_name(const string & name)
{
    if (name == "assert") return "assert_pol";
    if (name == "return") return "return_pol";
    return name;
}

vector<string> splitString (const string &s)
{
    vector<string> result;
    istringstream ss(s);
    string aux;    
    while (getline(ss, aux, '.'))
    {
        result.push_back(aux);
    }
    return result;
}

#define max(a,b) (a>b?a:b)

string generate(const ordered_json &pols, const string &type, const string &namespaceName)
{

    string code = "";

    // List all cmP pols namespaces
    vector<string> namespaces;
    vector<uint64_t> numberOfPols;
    vector<uint64_t> degree;

    for (ordered_json::const_iterator it = pols["references"].begin(); it != pols["references"].end(); ++it)
    {
        string key = it.key();
        if (pols["references"][key]["type"] == type)
        {
            vector<string> vs = splitString(key);
            string namespaceString = vs[0];
            uint64_t i;
            for (i=0; i<namespaces.size(); i++)
            {
                if (namespaces[i] == namespaceString)
                {
                    break;
                }
            }
            if (i == namespaces.size())
            {
                namespaces.push_back(namespaceString);
                numberOfPols.push_back(0);
                degree.push_back(0);
            }
        }
    }


    vector<string> declaration;
    vector<string> initialization;
    uint64_t maxDegree = 0;
    uint64_t offset = 0;
    uint64_t offset_transpositioned = 0;
    vector<uint64_t> localOffset;

    // Init the declaration and initialization arrays
    for (uint64_t i=0; i<namespaces.size(); i++)
    {
        declaration.push_back("");
        initialization.push_back("");
        localOffset.push_back(0);
    }

    // Calculate the number of polynomials of the requested type and the sufix
    uint64_t numPols = 0;
    string sufix;
    string fileDefine;
    if (type == "cmP")
    {
        numPols = pols["nCommitments"];
        sufix = "Commit";
        fileDefine = "COMMIT_POLS_HPP_" + namespaceName;
    }
    else if (type == "constP")
    {
        numPols = pols["nConstants"];
        sufix = "Constant";
        fileDefine = "CONSTANT_POLS_HPP_" + namespaceName;
    }

    code += "#ifndef " + fileDefine + "\n";
    code += "#define " + fileDefine + "\n";
    code += "\n";
    code += "#include <cstdint>\n";
    code += "#include \"goldilocks_base_field.hpp\"\n";
    code += "\n";

    code += "namespace " + namespaceName + "\n";
    code += "{\n\n";

    code += "class " + sufix + "Pol\n";
    code += "{\n";
    code += "private:\n";
    code += "    Goldilocks::Element * _pAddress;\n";
    code += "    uint64_t _degree;\n";
    code += "    uint64_t _index;\n";
    code += "public:\n";
    code += "    " + sufix + "Pol(Goldilocks::Element * pAddress, uint64_t degree, uint64_t index) : _pAddress(pAddress), _degree(degree), _index(index) {};\n";

    code += "    inline Goldilocks::Element & operator[](uint64_t i) { return _pAddress[i*" + to_string(numPols) + "]; };\n";
    code += "    inline Goldilocks::Element * operator=(Goldilocks::Element * pAddress) { _pAddress = pAddress; return _pAddress; };\n\n";
    code += "    inline Goldilocks::Element * address (void) { return _pAddress; }\n";
    code += "    inline uint64_t degree (void) { return _degree; }\n";
    code += "    inline uint64_t index (void) { return _index; }\n";

    code += "};\n\n";

    // For each cmP pol, add it to the proper namespace array
    for (uint64_t i = 0; i < numPols; i++)
    {
        for (ordered_json::const_iterator it = pols["references"].begin(); it != pols["references"].end(); ++it)
        {
            string key = it.key();
            json pol = pols["references"][key];
            if ( (pol["type"] == type) && (pol["id"]==i) )
            {
                vector<string> nameArray = splitString(key);
                string namespaceString = nameArray[0];
                string name = nameArray[1];
                uint64_t namespaceId = 0;
                for (; namespaceId < namespaces.size(); namespaceId++)
                {
                    if (namespaces[namespaceId] == namespaceString)
                    {
                        break;
                    }
                }
                string ctype = "";
                uint64_t csize = 0;
                if (pol["elementType"]=="field") { ctype="Goldilocks::Element"; csize=8; }
                else if (pol["elementType"]=="u8") { ctype="uint8_t"; csize=1; }
                else if (pol["elementType"]=="u16") { ctype="uint16_t"; csize=2; }
                else if (pol["elementType"]=="u32") { ctype="uint32_t"; csize=4; }
                else if (pol["elementType"]=="u64") { ctype="uint64_t"; csize=8; }
                else if (pol["elementType"]=="s8") { ctype="int8_t"; csize=1; }
                else if (pol["elementType"]=="s16") { ctype="int16_t"; csize=2; }
                else if (pol["elementType"]=="s32") { ctype="int32_t"; csize=4; }
                else if (pol["elementType"]=="s64") { ctype="int64_t"; csize=8; }
                else if (pol["elementType"]=="bool") { ctype="uint8_t"; csize=1; }
                else {
                    //console.log("elementType="+pol["elementType"]);
                    ctype="Goldilocks::Element"; csize=8;
                }

                string array;
                if (pol["isArray"])
                {
                    array = "["+to_string(pol["len"])+"]";
                }
                //declaration[namespaceId] += "    " + ctype + " * " + name + array + ";\n";
                declaration[namespaceId] += "    " + sufix + "Pol " + filter_name(name) + array + ";\n";
                if (pol["isArray"])
                {
                    initialization[namespaceId] += "        " + filter_name(name) + "{\n";
                    for (uint64_t a = 0; a < pol["len"]; a++)
                    {
                        string comma = ",";
                        if (a == uint64_t(pol["len"])-1)
                        {
                            comma = "";
                        }
                        initialization[namespaceId] += "            " + sufix + "Pol((" + ctype + " *)((uint8_t *)pAddress + " + to_string(offset_transpositioned) + "), degree, " + to_string(i+a) + ")" + comma + "\n";
                        offset += csize*uint64_t(pol["polDeg"]);
                        offset_transpositioned += csize;
                        localOffset[namespaceId] += csize;
                        numberOfPols[namespaceId] += 1;
                    }
                    initialization[namespaceId] += "        },\n";
                } else {
                    initialization[namespaceId] += "        " + filter_name(name) + "((" + ctype + " *)((uint8_t *)pAddress + " + to_string(offset_transpositioned) + "), degree, " + to_string(i) + "),\n";
                    offset += csize*uint64_t(pol["polDeg"]);
                    offset_transpositioned += csize;
                    localOffset[namespaceId] += csize;
                    numberOfPols[namespaceId] += 1;
                }
                degree[namespaceId] = pol["polDeg"];
                maxDegree = max(maxDegree, uint64_t(pol["polDeg"]));
                break;
            }
        }
    }
    for (uint64_t i=0; i<namespaces.size(); i++)
    {
        code += "class " + namespaces[i] + sufix + "Pols\n";
        code += "{\n";
        code += "public:\n";
        code += declaration[i];
        code += "private:\n";
        code += "    void * _pAddress;\n";
        code += "    uint64_t _degree;\n";
        code += "public:\n";
        code += "\n";
        code += "    " + namespaces[i] + sufix + "Pols (void * pAddress, uint64_t degree) :\n";
        code += initialization[i];
        code += "        _pAddress(pAddress),\n";
        code += "        _degree(degree) {};\n";
        code += "\n";
        code += "    inline static uint64_t pilDegree (void) { return " + to_string(degree[i]) + "; }\n";
        code += "    inline static uint64_t pilSize (void) { return " + to_string(localOffset[i]) + "; }\n";
        code += "    inline static uint64_t numPols (void) { return " + to_string(numberOfPols[i]) + "; }\n\n";
        code += "    inline void * address (void) { return _pAddress; }\n";
        code += "    inline uint64_t degree (void) { return _degree; }\n";
        code += "    inline uint64_t size (void) { return _degree*" + to_string(numberOfPols[i]) + "*sizeof(Goldilocks::Element); }\n";
        code += "};\n";
        code += "\n";
    }

    code += "class " + sufix + "Pols\n";
    code += "{\n";
    code += "public:\n";

    for (uint64_t i=0; i<namespaces.size(); i++)
    {
        code += "    " + namespaces[i] + sufix + "Pols " + namespaces[i] + ";\n";
    }

    code += "private:\n";
    code += "    void * _pAddress;\n";
    code += "    uint64_t _degree;\n";

    code += "public:\n";
    code += "\n";
    code += "    " + sufix + "Pols (void * pAddress, uint64_t degree) :\n";
    for (uint64_t i=0; i<namespaces.size(); i++)
    {
        code += "        " + namespaces[i] + "(pAddress, degree),\n";
    }
    code += "        _pAddress(pAddress),\n";
    code += "        _degree(degree) {}\n";
    code += "\n";
    code += "    inline static uint64_t pilSize (void) { return " + to_string(offset) + "; }\n";
    code += "    inline static uint64_t pilDegree (void) { return " + to_string(maxDegree) + "; }\n";
    code += "    inline static uint64_t numPols (void) { return " + to_string(numPols) + "; }\n\n";
    code += "    inline void * address (void) { return _pAddress; }\n";
    code += "    inline uint64_t degree (void) { return _degree; }\n";
    code += "    inline uint64_t size (void) { return _degree*" + to_string(numPols) + "*sizeof(Goldilocks::Element); }\n\n";
    code += "    inline Goldilocks::Element &getElement (uint64_t pol, uint64_t evaluation)\n";
    code += "    {\n";
    code += "        return ((Goldilocks::Element *)_pAddress)[pol + evaluation * numPols()];\n";
    code += "    }\n";
    code += "};\n";
    code += "\n";
    
    code += "} // namespace\n\n"; // namespace name

    code += "#endif // " + fileDefine + "\n";
    return code;
}
