#ifndef STATEDB_TEST_LOAD_HPP
#define STATEDB_TEST_LOAD_HPP

#include <pqxx/pqxx>
#include "config.hpp"
#include "goldilocks_base_field.hpp"

void runStateDBTestLoad (const Config& config);
void* stateDBTestLoadThread (const Config& config, uint8_t idBranch);

bool loadRoot (Goldilocks &fr, pqxx::connection *pConnection, int id, Goldilocks::Element (&root)[4]);
bool saveRoot (Goldilocks &fr, pqxx::connection *pConnection, int id, Goldilocks::Element (&root)[4]);

#endif