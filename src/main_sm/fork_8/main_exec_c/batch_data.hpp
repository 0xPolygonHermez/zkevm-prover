#ifndef BATCH_DATA_HPP_fork_8
#define BATCH_DATA_HPP_fork_8

#include <vector>
#include "main_sm/fork_8/main_exec_c/tx_data.hpp"

using namespace std;

namespace fork_8
{

class BatchData
{
public:
    vector<TXData> tx;
};

}

#endif