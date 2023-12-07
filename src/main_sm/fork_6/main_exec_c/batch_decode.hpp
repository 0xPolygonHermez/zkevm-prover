#ifndef BATCH_DECODE_HPP_fork_6
#define BATCH_DECODE_HPP_fork_6

#include <string>
#include <vector>
#include "zklog.hpp"
#include "zkresult.hpp"
#include "utils.hpp"
#include "main_sm/fork_6/main_exec_c/batch_data.hpp"

using namespace std;

namespace fork_6
{

// Decode a batch L2 data buffer
zkresult BatchDecode(const string &input, BatchData (&output));

}

#endif