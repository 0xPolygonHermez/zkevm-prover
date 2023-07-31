#ifndef HASHDB_TEST_LOAD_HPP
#define HASHDB_TEST_LOAD_HPP

#include <pqxx/pqxx>
#include "config.hpp"
#include "goldilocks_base_field.hpp"

void runHashDBTestLoad (const Config& config);
void* hashDBTestLoadThread (const Config& config, uint8_t idBranch);

bool loadRoot (Goldilocks &fr, pqxx::connection *pConnection, int id, Goldilocks::Element (&root)[4]);
bool saveRoot (Goldilocks &fr, pqxx::connection *pConnection, int id, Goldilocks::Element (&root)[4]);

#endif