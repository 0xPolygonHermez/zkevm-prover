#ifndef CHILD_HPP
#define CHILD_HPP

#include "leaf_node.hpp"
#include "intermediate_node.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

enum ChildType
{
    UNSPECIFIED  = 0,
    ZERO         = 1,
    LEAF         = 2,
    INTERMEDIATE = 3,
    TREE_CHUNK   = 4
};

class TreeChunk;

class Child
{
public:
    ChildType        type;
    LeafNode         leaf;
    IntermediateNode intermediate;
    uint64_t         treeChunkId;

    Child() : type(UNSPECIFIED) {};

    inline Child & operator=(const Child &other)
    {
        // Copy type
        type = other.type;

        // Copy only rellevant attributes according to the type
        switch (type)
        {
            case ZERO:
            {
                return *this;
            }
            case LEAF:
            {
                leaf = other.leaf;
                return *this;
            }
            case INTERMEDIATE:
            {
                intermediate = other.intermediate;
                return *this;
            }
            case TREE_CHUNK:
            {
                treeChunkId = other.treeChunkId;
                return *this;
            }
            default:
            {
                zklog.error("Child::operator=() found invalid other.type=" + to_string(other.type));
                exitProcess();
                return *this;
            }
        }
    }

    string print (Goldilocks &fr) const;

    string getTypeLetter (void) const
    {
        switch (type)
        {
            case UNSPECIFIED: return "U";
            case ZERO: return "z";
            case LEAF: return "L";
            case INTERMEDIATE: return "I";
            case TREE_CHUNK: return "T";
            default:
            {
                zklog.error("Child::getTypeLetter() found invalid other.type=" + to_string(type));
                exitProcess();
                return "";
            }
        }
    }
};

#endif