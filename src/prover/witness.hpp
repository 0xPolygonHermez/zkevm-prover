#ifndef WITNESS_HPP
#define WITNESS_HPP

#include <string>
#include "zkresult.hpp"
#include "database_map.hpp"

using namespace std;

zkresult witness2db (const string &witness, DatabaseMap::MTMap &db, DatabaseMap::ProgramMap &programs, mpz_class &stateRoot);

#endif