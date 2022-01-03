#ifndef INPUT_HPP
#define INPUT_HPP

#include <nlohmann/json.hpp>
#include "context.hpp"
#include "public_inputs.hpp"

using json = nlohmann::json;

// Loads the input JSON file transactions into memory
void loadInput (Context &ctx, json &input);

class Input
{
public:
    string message; // used in gRPC: "calculate", "cancel"
    PublicInputs publicInputs;
    string globalExitRoot;
    vector<string> txs;
    map<string, string> keys;
};

#endif