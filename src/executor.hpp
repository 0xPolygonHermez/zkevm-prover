#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "ffiasm/fr.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void execute(RawFr &fr, json &input, json &rom, json &pil);

#endif