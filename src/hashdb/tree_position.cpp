#include "tree_position.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

using namespace std;

bool TreePosition::operator==(const TreePosition &other)
{
    if (level < 0)
    {
        return false;
    }
    if (level != other.level)
    {
        return false;
    }
    if (other.level < 0)
    {
        zklog.error("TreePosition::operator==() found other.level=" + to_string(other.level) + " < 0");
        exitProcess();
    }
    if (other.keys.size() < (uint64_t)other.level)
    {
        zklog.error("TreePosition::operator==() found other.keys.size=" + to_string(other.keys.size()) + " < other.level=" + to_string(other.level));
        exitProcess();
    }
    for (uint64_t i = 0; i< (uint64_t)other.level; i++)
    {
        if (keys[i] != other.keys[i])
        {
            return false;
        }
    }
    return true;
}

TreePosition & TreePosition::operator=(const TreePosition &other)
{
    if (other.level < 0)
    {
        //zklog.error("TreePosition::operator=() found other.level=" + to_string(other.level) + " < 0");
        //exitProcess();
        return *this;
    }
    if (other.keys.size() < (uint64_t)other.level)
    {
        zklog.error("TreePosition::operator=() found other.keys.size=" + to_string(other.keys.size()) + " < other.level=" + to_string(other.level));
        exitProcess();
    }
    
    level = other.level;
    keys.clear();
    for (uint64_t i=0; i<(uint64_t)other.level; i++)
    {
        keys.push_back(other.keys[i]);
    }

    return *this;
}