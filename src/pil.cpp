#include "pil.hpp"
#include "config.hpp"

void Pil::parse (json &pil)
{
    // Check it has not been parsed before
    if (bParsed)
    {
        cerr << "Error: Pil::parse() already called" << endl;
        exit(-1);
    }

    // Parse the different parts of the PIL JSON file
    parseRootElements(pil);
    parsePolynomials(pil);
    parsePolIdentities(pil);
    parsePlookupIdentities(pil);
    parseExpressions(pil);
}

void Pil::parseRootElements (json &pil)
{
    // PIL JSON file must contain a nCommitments key at the root level
    if ( !pil.contains("nCommitments") ||
         !pil["nCommitments"].is_number_unsigned() )
    {
        cerr << "Error: Pil::parseRootElements() nCommitments key not found in PIL JSON file" << endl;
        exit(-1);
    }
    nCommitments = pil["nCommitments"];

    // PIL JSON file must contain a nConstants key at the root level
    if ( !pil.contains("nConstants") ||
         !pil["nConstants"].is_number_unsigned() )
    {
        cerr << "Error: Pil::parseRootElements() nConstants key not found in PIL JSON file" << endl;
        exit(-1);
    }
    nConstants = pil["nConstants"];

    // PIL JSON file must contain a nQ key at the root level
    if ( !pil.contains("nQ") ||
         !pil["nQ"].is_number_unsigned() )
    {
        cerr << "Error: Pil::parseRootElements() nQ key not found in PIL JSON file" << endl;
        exit(-1);
    }
    nQ = pil["nQ"];

    // PIL JSON file must contain a nIm key at the root level
    if ( !pil.contains("nIm") ||
         !pil["nIm"].is_number_unsigned() )
    {
        cerr << "Error: Pil::parseRootElements() nIm key not found in PIL JSON file" << endl;
        exit(-1);
    }
    nIm = pil["nIm"];
}

void Pil::parsePolynomials (json &pil)
{
    // PIL JSON file must contain a references structure at the root level
    if ( !pil.contains("references") ||
         !pil["references"].is_structured() )
    {
        cerr << "Error: Pil::parsePolynomials() references key not found in PIL JSON file" << endl;
        exit(-1);
    }

    // Iterate the PIL JSON references array
    json references = pil["references"];
    uint64_t addedCmPols = 0;
    uint64_t addedConstPols = 0;
    for (json::iterator it = references.begin(); it != references.end(); ++it) {
        string key = it.key();
        json value = it.value();
        if ( value.is_object() &&
             value.contains("type") && 
             value["type"].is_string() &&
             value.contains("id") &&
             value["id"].is_number_unsigned() ) 
        {
            string type = it.value()["type"];
            uint64_t id = it.value()["id"];
            if (type=="cmP" || type=="constP") {
                if (id>=NPOLS)
                {
                    cerr << "Error: Pil::parsePolynomials() polynomial " << key << " id(" << id << ") >= NPOLS(" << NPOLS << ")" << endl;
                    exit(-1);
                }
                string elementType = it.value()["elementType"];
                PolJsonData data;
                data.name = key;
                data.id = id;
                data.elementType = elementType;
                if (type=="cmP")
                {
                    cmPols.push_back(data);
#ifdef LOG_POLS
                    cout << "Added committed polynomial " << addedCmPols << ": " << key << " with ID " << id << " and type " << elementType << endl;
#endif
                    addedCmPols++;
                }
                else
                {
                    constPols.push_back(data);
#ifdef LOG_POLS
                    cout << "Added constant polynomial " << addedConstPols << ": " << key << " with ID " << id << " and type " << elementType << endl;
#endif
                    addedConstPols++;
                }
            }
        }
    }
#ifdef LOG_POLS
    cout << "Added " << addedCmPols << " committed polynomials and " << addedConstPols << " constant polynomials" << endl;
#endif

    // Consistency check
    if (cmPols.size() != nCommitments)
    {
        cerr << "Error: Pil::parsePolynomials() found cmPols.size()=" << cmPols.size() << " different from nCommitments=" << nCommitments << endl;
        exit(-1);
    }
    if (constPols.size() != nConstants)
    {
        cerr << "Error: Pil::parsePolynomials() found constPols.size()=" << constPols.size() << " different from nConstants=" << nConstants << endl;
        exit(-1);
    }
}

void Pil::parsePolIdentities(json &pil)
{
    // PIL JSON file must contain a polIdentities array at the root level
    if ( !pil.contains("polIdentities") ||
         !pil["polIdentities"].is_array() )
    {
        cerr << "Error: Pil::parsePolIdentities() polIdentities key not found in PIL JSON file" << endl;
        exit(-1);
    }

    // Parse and store every identity
    uint64_t size = pil["polIdentities"].size();
    for (uint64_t i=0; i<size; i++)
    {
        uint64_t aux = pil["polIdentities"][i];
        polIdentities.push_back(aux);
    }
}

void Pil::parsePlookupIdentities(json &pil)
{
    // PIL JSON file must contain a plookupIdentities array at the root level
    if ( !pil.contains("plookupIdentities") ||
         !pil["plookupIdentities"].is_array() )
    {
        cerr << "Error: Pil::parsePlookupIdentities() plookupIdentities key not found in PIL JSON file" << endl;
        exit(-1);
    }

    // Parse and store every identity
    uint64_t size = pil["plookupIdentities"].size();
    for (uint64_t i=0; i<size; i++)
    {
        // Check the existance of the root element
        json plookupIdentity = pil["plookupIdentities"][i];
        if ( !plookupIdentity.is_structured() )
        {
            cerr << "Error: Pil::parsePlookupIdentities() plookupIdentities key not structured" << endl;
            exit(-1);           
        }

        // Store plookup identity data in this instance
        PlookupIdentity identity;

        // Store the f array
        if (!plookupIdentity.contains("f") ||
            !plookupIdentity["f"].is_array())
        {
            cerr << "Error: Pil::parsePlookupIdentities() f is not an array" << endl;
            exit(-1);           
        }
        for (uint64_t j=0; j<plookupIdentity["f"].size(); j++)
        {
            json f=plookupIdentity["f"][j];
            if (!f.is_number_unsigned())
            {
                cerr << "Error: Pil::parsePlookupIdentities() f[j] is not an unsigned number" << endl;
                exit(-1);           
            }
            uint64_t aux = f;
            identity.f.push_back(aux);
        }

        // Store the t array
        if (!plookupIdentity.contains("t") ||
            !plookupIdentity["t"].is_array())
        {
            cerr << "Error: Pil::parsePlookupIdentities() t is not an array" << endl;
            exit(-1);           
        }
        for (uint64_t j=0; j<plookupIdentity["t"].size(); j++)
        {
            json t=plookupIdentity["t"][j];
            if (!t.is_number_unsigned())
            {
                cerr << "Error: Pil::parsePlookupIdentities() t[j] is not an unsigned number" << endl;
                exit(-1);           
            }
            uint64_t aux = t;
            identity.t.push_back(aux);
        }

        // Store the selF element
        if (!plookupIdentity.contains("selF"))
        {
            cerr << "Error: Pil::parsePlookupIdentities() selF element not found" << endl;
            exit(-1);
        }
        json selF = plookupIdentity["selF"];
        if (selF.is_null())
        {
            identity.bSelFPresent = false;
            identity.selF = 0;
        }
        else if (selF.is_number_unsigned())
        {
            identity.bSelFPresent = true;
            identity.selF = selF;
        }
        else
        {
            cerr << "Error: Pil::parsePlookupIdentities() selF element is neither null nor unsigned number" << endl;
            exit(-1);
        }

        // Store the selF element
        if (!plookupIdentity.contains("selT"))
        {
            cerr << "Error: Pil::parsePlookupIdentities() selT element not found" << endl;
            exit(-1);
        }
        json selT = plookupIdentity["selT"];
        if (selF.is_null())
        {
            identity.bSelTPresent = false;
            identity.selT = 0;
        }
        else if (selF.is_number_unsigned())
        {
            identity.bSelTPresent = true;
            identity.selT = selT;
        }
        else
        {
            cerr << "Error: Pil::parsePlookupIdentities() selT element is neither null nor unsigned number" << endl;
            exit(-1);
        }

        // Store the identity data
        plookupIdentities.push_back(identity);
    }
}

void Pil::parseExpressions(json &pil)
{
    // PIL JSON file must contain a expressions array at the root level
    if ( !pil.contains("expressions") ||
         !pil["expressions"].is_array() )
    {
        cerr << "Error: Pil::parseExpressions() expressions key not found in PIL JSON file" << endl;
        exit(-1);
    }

    json expressionArray = pil["expressions"];
    parseExpressionArray( expressionArray, expressions);
}


void Pil::parseExpressionArray(json &expressionArrayJson, vector<Expression> &expressionsVector)
{
    // Parse and store every identity
    uint64_t size = expressionArrayJson.size();
    for (uint64_t i=0; i<size; i++)
    {
        json expressionJson = expressionArrayJson[i];
        Expression expression;
        parseExpression(expressionJson, expression);
        expressionsVector.push_back(expression);
    }
}

void Pil::parseExpression(json &expressionJson, Expression &expression)
{
    // Expression must be structured
    if (!expressionJson.is_structured())
    {
        cerr << "Error: Pil::parseExpression() expression element is not structured" << endl;
        exit(-1);
    }

    // Store op element
    if (!expressionJson.contains("op"))
    {
        cerr << "Error: Pil::parseExpression() op element not found" << endl;
        exit(-1);
    }
    json op = expressionJson["op"]; 
    if (!op.is_string())
    {
        cerr << "Error: Pil::parseExpression() op element not a string" << endl;
        exit(-1);
    }
    expression.op = op;

    // Store deg element
    if (!expressionJson.contains("deg"))
    {
        cerr << "Error: Pil::parseExpression() deg element not found" << endl;
        exit(-1);
    }
    json deg = expressionJson["deg"]; 
    if (!deg.is_number_unsigned())
    {
        cerr << "Error: Pil::parseExpression() deg element not an unsigned number" << endl;
        exit(-1);
    }
    expression.deg = deg;

    // Store id element, if present
    if (!expressionJson.contains("id"))
    {
        expression.bIdPresent = false;
        expression.id = 0;
    }
    else
    {
        json id = expressionJson["id"];
        if (!id.is_number_unsigned())
        {
            cerr << "Error: Pil::parseExpression() id element not an unsigned number" << endl;
            exit(-1);
        }
        expression.bIdPresent = true;
        expression.id = id;
    }

    // Store idQ element, if present
    if (!expressionJson.contains("idQ"))
    {
        expression.bIdQPresent = false;
        expression.idQ = 0;
    }
    else
    {
        json idQ = expressionJson["idQ"];
        if (!idQ.is_number_unsigned())
        {
            cerr << "Error: Pil::parseExpression() idQ element not an unsigned number" << endl;
            exit(-1);
        }
        expression.bIdQPresent = true;
        expression.idQ = idQ;
    }

    // Store next element, if present
    if (!expressionJson.contains("next"))
    {
        expression.bNextPresent = false;
        expression.next = false;
    }
    else
    {
        json next = expressionJson["next"];
        if (!next.is_boolean())
        {
            cerr << "Error: Pil::parseExpression() next element not a boolean" << endl;
            exit(-1);
        }
        expression.bNextPresent = true;
        expression.next = next;
    }

    // Store const element, if present
    if (!expressionJson.contains("const"))
    {
        expression.bConstPresent = false;
        expression.constant = "";
    }
    else
    {
        json constant = expressionJson["const"];
        if (!constant.is_string())
        {
            cerr << "Error: Pil::parseExpression() constant element not a string" << endl;
            exit(-1);
        }
        expression.bConstPresent = true;
        expression.constant = constant;
    }

    // Store the values array, if present
    if (expressionJson.contains("values"))
    {
        json values = expressionJson["values"];
        if (!values.is_array())
        {
            cerr << "Error: Pil::parseExpression() values element not an array" << endl;
            exit(-1);
        }
        parseExpressionArray(values, expression.values);
    }
}