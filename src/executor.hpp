#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <string>
#include <nlohmann/json.hpp>
#include "ffiasm/fr.hpp"

using namespace std;
using json = nlohmann::json;

void execute (RawFr &fr, json &input, json &rom, json &pil, string &outputFile);

#endif