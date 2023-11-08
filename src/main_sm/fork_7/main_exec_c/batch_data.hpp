#ifndef BATCH_DATA_HPP_fork_7
#define BATCH_DATA_HPP_fork_7

#include <vector>
#include "main_sm/fork_7/main_exec_c/tx_data.hpp"

using namespace std;

namespace fork_7
{

class BatchData
{
public:
    vector<TXData> tx;
};

}

#endif