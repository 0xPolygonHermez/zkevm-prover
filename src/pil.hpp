#ifndef PIL_HPP
#define PIL_HPP

#include <string>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

class PolJsonData
{
public:
    string   name;
    uint64_t id;
    string   elementType;
};

class Expression
{
public:
    string op;
    uint64_t deg;
    bool bIdPresent; // if deg==1, then id is present
    uint64_t id;
    bool bNextPresent; // if deg==1, then next is present
    bool next;
    vector<Expression> values;
};

class PlookupIdentity
{
public:
    vector<uint64_t> f;
    vector<uint64_t> t;
    bool bSelFPresent;
    uint64_t selF;
    bool bSelTPresent;
    uint64_t selT;
};

class Pil
{
private:
    bool bParsed;
public:
    uint64_t nCommitments;
    uint64_t nConstants;
    uint64_t nQ;
    uint64_t nIm;

    vector<PolJsonData> cmPols;
    vector<PolJsonData> constPols;

    vector<uint64_t> polIdentities;
    vector<PlookupIdentity> plookupIdentities;
    vector<Expression> expressions;
    Pil(): bParsed(false) {};
    void parse (json &pilJson);
private:
    void parseRootElements(json &pilJson);
    void parsePolynomials(json &pilJson);
    void parsePolIdentities(json &pilJson);
    void parsePlookupIdentities(json &pilJson);
    void parseExpressions(json &pilJson);
    void parseExpressionArray(json &expressionArrayJson, vector<Expression> &expressionsVector);
    void parseExpression(json &expressionJson, Expression &expression);
};

#endif