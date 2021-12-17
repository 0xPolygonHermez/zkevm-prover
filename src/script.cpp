#include "script.hpp"
#include "config.hpp"

void Script::parse (json &script)
{
    // Check it has not been parsed before
    if (bParsed)
    {
        cerr << "Error: Script::parse() already called" << endl;
        exit(-1);
    }

    // Parse the different parts of the script JSON file
    parseReferences(script);
    parseProgram(script);
}

void Script::parseReferences (json &script)
{
    // Script JSON file must contain a refs array at the root level
    if ( !script.contains("refs") ||
         !script["refs"].is_array() )
    {
        cerr << "Error: Script::parseReferences() refs key not found in PIL JSON file" << endl;
        exit(-1);
    }

    // Parse and store every reference
    uint64_t size = script["refs"].size();
    for (uint64_t i=0; i<size; i++)
    {
        Reference ref;
        json refJson = script["refs"][i];

        // Parse the mandatory element id
        ref.id = refJson["id"];
        if (ref.id != i)
        {
            cerr << "Error: Script::parseReferences() found unexpected id " << ref.id << " in loop " << i << endl;
            exit(-1);
        }

        // Parse the mandatory element type
        string type = refJson["type"];
        if (type == "pol")
        {
            ref.type = rt_pol;
            string elementType = refJson["elementType"]; // TODO: uncomment this when the elementType is present always in a pol
            ref.elementType = string2et(elementType);
            //if (!refJson.contains("elementType")) cout << "Missing elementType in reference with id: " << ref.id << endl;
            ref.N = refJson["N"];
        }
        else if (type == "field")
        {
            ref.type = rt_field;
        }
        else if (type == "treeGroup")
        {
            ref.type = rt_treeGroup;
            ref.nGroups = refJson["nGroups"];
            ref.groupSize = refJson["groupSize"];
        }
        else if (type == "treeGroupMultipol")
        {
            ref.type = rt_treeGroupMultipol;
            ref.nGroups = refJson["nGroups"];
            ref.groupSize = refJson["groupSize"];
            ref.nPols = refJson["nPols"];
        }
        else if (type == "treeGroup_elementProof")
        {
            ref.type = rt_treeGroup_elementProof;
            ref.nGroups = refJson["nGroups"];
            ref.groupSize = refJson["groupSize"];
        }
        else if (type == "treeGroup_groupProof")
        {
            ref.type = rt_treeGroup_groupProof;
            ref.nGroups = refJson["nGroups"];
            ref.groupSize = refJson["groupSize"];
        }
        else if (type == "treeGroupMultipol_groupProof")
        {
            ref.type = rt_treeGroupMultipol_groupProof;
            ref.nGroups = refJson["nGroups"];
            ref.groupSize = refJson["groupSize"];
            ref.nPols = refJson["nPols"];
        }
        else if (type == "idxArray")
        {
            ref.type = rt_idxArray;
            string elementType = refJson["elementType"];
            ref.elementType = string2et(elementType);
            ref.N = refJson["N"];
        }
        else if (type == "int")
        {
            ref.type = rt_int;
        }
        else
        {
            cerr << "Error: Script::parseReferences() found an unknown type: " << type << endl;
            exit(-1);
        }

        // Store the reference
        refs.push_back(ref);
    }
}

void Script::parseProgram (json &script)
{
    // Script JSON file must contain a program array at the root level
    if ( !script.contains("program") ||
         !script["program"].is_array() )
    {
        cerr << "Error: Script::parseProgram() program key not found in PIL JSON file" << endl;
        exit(-1);
    }

    // Parse and store every reference
    uint64_t size = script["program"].size();
    for (uint64_t i=0; i<size; i++)
    {
        Program program;
        json programJson = script["program"][i];

        // Parse the mandatory element op
        program.op = programJson["op"];

        // Parse the optional elements
        if (programJson.contains("msg")) program.msg = programJson["msg"];
        if (programJson.contains("result")) program.result = programJson["result"];
        if (programJson.contains("tree")) program.tree = programJson["tree"];
        if (programJson.contains("nGroups")) program.nGroups = programJson["nGroups"];
        if (programJson.contains("groupSize")) program.groupSize = programJson["groupSize"];
        if (programJson.contains("nPols")) program.nPols = programJson["nPols"];
        if (programJson.contains("polIdx")) program.polIdx = programJson["polIdx"];
        if (programJson.contains("value")) program.value = programJson["value"];
        if (programJson.contains("extendBits")) program.extendBits = programJson["extendBits"];
        if (programJson.contains("N")) program.N = programJson["N"];
        if (programJson.contains("f")) program.f = programJson["f"];
        if (programJson.contains("t")) program.t = programJson["t"];
        if (programJson.contains("resultH1")) program.resultH1 = programJson["resultH1"];
        if (programJson.contains("resultH2")) program.resultH2 = programJson["resultH2"];
        if (programJson.contains("constant")) program.constant = programJson["constant"];
        if (programJson.contains("shift")) program.shift = programJson["shift"];
        if (programJson.contains("idx")) program.idx = programJson["idx"];
        if (programJson.contains("pos")) program.pos = programJson["pos"];
        if (programJson.contains("idxArray")) program.idxArray = programJson["idxArray"];
        if (programJson.contains("w")) program.w = programJson["w"];
        if (programJson.contains("add")) program.add = programJson["add"];
        if (programJson.contains("mod")) program.mod = programJson["mod"];
        if (programJson.contains("shiftInv")) program.shiftInv = programJson["shiftInv"];
        if (programJson.contains("specialX")) program.specialX = programJson["specialX"];
        if (programJson.contains("values"))
        {
            json values = programJson["values"];
            uint64_t size = values.size();
            for (uint64_t j=0; j<size; j++)
            {
                uint64_t value = values[j];
                program.values.push_back(value);
            }
        }
        if (programJson.contains("pols"))
        {
            json pols = programJson["pols"];
            uint64_t size = pols.size();
            for (uint64_t j=0; j<size; j++)
            {
                uint64_t pol = pols[j];
                program.pols.push_back(pol);
            }
        }
        if (programJson.contains("fields"))
        {
            json fields = programJson["fields"];
            uint64_t size = fields.size();
            for (uint64_t j=0; j<size; j++)
            {
                uint64_t field = fields[j];
                program.pols.push_back(field);
            }
        }

        // Store the program
        this->program.push_back(program);
    }
}