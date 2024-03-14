#ifndef BATCH_DATA_HPP_fork_5
#define BATCH_DATA_HPP_fork_5

#include <vector>
#include "main_sm/fork_5/main_exec_c/tx_data.hpp"

using namespace std;

namespace fork_5
{

class BatchData
{
public:
    vector<TXData> tx;
};

}

#endif