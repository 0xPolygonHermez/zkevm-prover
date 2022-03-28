#ifndef SCRIPT_HPP
#define SCRIPT_HPP

#include <string>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include "reference.hpp"
#include "program.hpp"
#include "output.hpp"
#include "pol_types.hpp"
#include "ff/ff.hpp"

using namespace std;
using json = nlohmann::json;

class Script
{
private:
    FiniteField &fr;
    bool bParsed;

public:

    vector<Reference> refs;
    vector<Program> program;
    Output output;

    Script(FiniteField &fr): fr(fr), bParsed(false) {};
    void parse (json &scriptJson);

private:
    void parseReference (json &refJson, Reference &ref);
    void parseReferences (json &scriptJson);
    void parseProgram (json &scriptJson);
    void parseOutput (json &scriptJson);
    void parseOutput (json &json, Output &output);
};

#endif