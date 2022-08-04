#ifndef STARK_TEST_HPP
#define STARK_TEST_HPP


#include "utils.hpp"
#include "stark_info.hpp"

#include <vector>

// Test vectors files
#define starkInfo_File "basic.starkinfo.json"
#define commited_file "basic.commit"
#define constant_file "basic.const"
#define constant_tree_file "basic.consttree"

#define NUM_CHALLENGES_TEST 8

using namespace std;

void StarkTest(void);

#endif