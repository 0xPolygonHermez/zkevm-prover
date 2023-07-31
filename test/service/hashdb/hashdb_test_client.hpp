#ifndef HASHDB_TEST_CLIENT_HPP
#define HASHDB_TEST_CLIENT_HPP

#include "config.hpp"

void runHashDBTestClient (const Config& config);
void* hashDBTestClientThread (const Config& config);

#endif