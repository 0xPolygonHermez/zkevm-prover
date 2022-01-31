#ifndef PROVE_CONTEXT_HPP
#define PROVE_CONTEXT_HPP

#include <string>
#include "config.hpp"
#include "input.hpp"
#include "ffiasm/fr.hpp"
#include "database.hpp"
#include "proof.hpp"
#include "counters.hpp"

using namespace std;

class ProveContext
{
public:
    /* IDs */
    string uuid;
    string timestamp;

    /* Files */
    string inputFile;
    string inputFileEx;
    string publicFile;
    string proofFile;

    /* Executor */
    Input input;
    Database db;
    Counters counters;

    Proof proof;

    ProveContext (RawFr &fr): input(fr), db(fr) {};

    void init (const Config &config);
};

#endif