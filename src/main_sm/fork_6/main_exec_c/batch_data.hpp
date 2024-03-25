#ifndef BATCH_DATA_HPP_fork_6
#define BATCH_DATA_HPP_fork_6

#include <vector>
#include "main_sm/fork_6/main_exec_c/tx_data.hpp"

using namespace std;

namespace fork_6
{

class BatchData
{
public:
    vector<TXData> tx;
};

}

#endif