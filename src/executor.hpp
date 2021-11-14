#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <string>
#include "ffiasm/fr.hpp"
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

void execute(RawFr &fr, json &input, json &rom, json &pil, string &outputFile);

#endif