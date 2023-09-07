#ifndef STATE_ROOT_HPP
#define STATE_ROOT_HPP

#include <string>

using namespace std;

class StateRoot
{
public:
    string realStateRoot;
    string virtualStateRoot;

    void set (const string stateRoot, const bool bVirtual)
    {
        if (bVirtual)
        {
            virtualStateRoot = stateRoot;
        }
        else
        {
            realStateRoot = stateRoot;
        }
    }

    bool empty (void)
    {
        return (realStateRoot.size() == 0) && (virtualStateRoot.size() == 0);
    }

    bool equals (const string stateRoot, const bool bVirtual)
    {
        if (bVirtual)
        {
            return virtualStateRoot == stateRoot;
        }
        else
        {
            return realStateRoot == stateRoot;
        }
    }

    bool operator== (const StateRoot &other)
    {
        return (realStateRoot == other.realStateRoot) && (virtualStateRoot == other.virtualStateRoot);
    }

    bool operator!= (const StateRoot &other)
    {
        return (realStateRoot != other.realStateRoot) || (virtualStateRoot != other.virtualStateRoot);
    }

    const string toString (void) const
    {
        string result;
        if (virtualStateRoot.size() > 0)
        {
            result = "(virtual)" + virtualStateRoot;
        }
        if (realStateRoot.size() > 0)
        {
            if (result.size() > 0)
            {
                result += "/";
            }
            result += "(real)" + realStateRoot;
        }
        return result;
    }
};

#endif