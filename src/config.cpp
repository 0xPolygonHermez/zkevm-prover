#include <string>
#include <nlohmann/json.hpp>
#include "config.hpp"

using namespace std;
using json = nlohmann::json;

void Config::load(json &config)
{
    bServer = true;
    if (config.contains("inputFile") && 
        config["inputFile"].is_string())
    {
        inputFile = config["inputFile"];
        bServer = false;
    }
    if (config.contains("romFile") && 
        config["romFile"].is_string())
    {
        romFile = config["romFile"];
    }
    if (config.contains("outputPath") && 
        config["outputPath"].is_string())
    {
        outputPath = config["outputPath"];
    }
    if (config.contains("pilFile") && 
        config["pilFile"].is_string())
    {
        pilFile = config["pilFile"];
    }
    if (config.contains("cmPolsFile") && 
        config["cmPolsFile"].is_string())
    {
        cmPolsFile = config["cmPolsFile"];
    }
    if (config.contains("constPolsFile") && 
        config["constPolsFile"].is_string())
    {
        constPolsFile = config["constPolsFile"];
    }
    if (config.contains("constantsTreeFile") && 
        config["constantsTreeFile"].is_string())
    {
        constantsTreeFile = config["constantsTreeFile"];
    }
    if (config.contains("scriptFile") && 
        config["scriptFile"].is_string())
    {
        scriptFile = config["scriptFile"];
    }
    if (config.contains("starkFile") && 
        config["starkFile"].is_string())
    {
        starkFile = config["starkFile"];
    }
    if (config.contains("verifierFile") && 
        config["verifierFile"].is_string())
    {
        verifierFile = config["verifierFile"];
    }
    if (config.contains("witnessFile") && 
        config["witnessFile"].is_string())
    {
        witnessFile = config["witnessFile"];
    }
    if (config.contains("starkVerifierFile") && 
        config["starkVerifierFile"].is_string())
    {
        starkVerifierFile = config["starkVerifierFile"];
    }
    if (config.contains("dbHost") && 
        config["dbHost"].is_string())
    {
        dbHost = config["dbHost"];
    }
    if (config.contains("dbPort") && 
        config["dbPort"].is_number())
    {
        dbPort = config["dbPort"];
    }
    if (config.contains("dbUser") && 
        config["dbUser"].is_string())
    {
        dbUser = config["dbUser"];
    }
    if (config.contains("dbPassword") && 
        config["dbPassword"].is_string())
    {
        dbPassword = config["dbPassword"];
    }
    if (config.contains("dbDatabaseName") && 
        config["dbDatabaseName"].is_string())
    {
        dbDatabaseName = config["dbDatabaseName"];
    }
    if (config.contains("dbTableName") && 
        config["dbTableName"].is_string())
    {
        dbTableName = config["dbTableName"];
    }
}