#ifndef STARK_TEST_HPP
#define STARK_TEST_HPP

#include "utils.hpp"
#include "stark_info.hpp"
#include "execFile.hpp"
#include "starkpil/test/zkevm_verifier_cpp/main.hpp"
#include "starkpil/test/zkevm_c12a_verifier_cpp/main.c12a.hpp"
#include "starkpil/test/zkevm_c12b_verifier_cpp/main.c12b.hpp"
#include "starkpil/test/starkC12aMock.hpp"
#include "starkpil/test/starkC12bMock.hpp"
#include "friProofC12.hpp"
#include <vector>
#include "alt_bn128.hpp"
#include "groth16.hpp"
#include "zkey_utils.hpp"

// Test vectors files
#define starkInfo_File "basic.starkinfo.json"
#define commited_file "basic.commit"
#define constant_file "basic.const"
#define constant_tree_file "basic.consttree"

#define NUM_CHALLENGES_TEST 8

using namespace std;

void StarkTest(void);

#endif