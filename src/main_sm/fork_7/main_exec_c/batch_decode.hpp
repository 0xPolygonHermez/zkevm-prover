#ifndef BATCH_DECODE_HPP_fork_7
#define BATCH_DECODE_HPP_fork_7

#include <string>
#include <vector>
#include "zklog.hpp"
#include "zkresult.hpp"
#include "utils.hpp"
#include "main_sm/fork_7/main_exec_c/batch_data.hpp"

using namespace std;

namespace fork_7
{

// Decode a batch L2 data buffer
zkresult BatchDecode(const string &input, BatchData (&output));

}

#endif