#ifndef SMT_ACTION_CONTEXT_HPP
#define SMT_ACTION_CONTEXT_HPP

#include <map>
#include <vector>
#include "goldilocks_base_field.hpp"
#include "smt_action.hpp"

class SmtActionContext
{
public:
    // Deepest level and current level
    uint64_t level; // Level at which the proof starts
    int64_t currentLevel; // Current level, from level to zero, as we climb up the tree

    // Remaining key and preceding bits
    Goldilocks::Element rKey[4];
    Goldilocks::Element siblingRKey[4];
    vector<uint64_t> bits; // Key bits consumed in the tree nodes, i.e. preceding remaining key rKey
    vector<uint64_t> siblingBits; // Sibling key bits consumed in the tree nodes, i.e. preceding sibling remaining key siblingRKey

    void init (Goldilocks &fr, const SmtAction &action);
};

#endif