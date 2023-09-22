#include "key_value_tree.hpp"
#include "scalar.hpp"
#include "exit_process.hpp"

Goldilocks kvtfr;

zkresult KeyValueTree::write (const Goldilocks::Element (&key)[4], const mpz_class &value, uint64_t &level)
{
    string keyString = fea2string(kvtfr, key);

    return write(keyString, value, level);
}

zkresult KeyValueTree::write (const string &keyString, const mpz_class &value, uint64_t &level)
{
    keys[keyString].emplace_back(value);

    // Hardcode the level (temporary) (TODO)
    level = 128;

    return ZKR_SUCCESS;
}

zkresult KeyValueTree::read (const Goldilocks::Element (&key)[4], mpz_class &value, uint64_t &level)
{
    string keyString = fea2string(kvtfr, key);
    return read(keyString, value, level);
}

zkresult KeyValueTree::read (const string &keyString, mpz_class &value, uint64_t &level)
{
    // Find the map entry
    unordered_map< string, vector<mpz_class> >::const_iterator it;
    it = keys.find(keyString);
    if (it == keys.end())
    {
        return ZKR_DB_KEY_NOT_FOUND;
    }

    // There there is at least one value
    if (it->second.size() == 0)
    {
        zklog.error("KeyValueTree::read() found values size = 0");
        exitProcess();
    }

    // Return the last value
    value = it->second[it->second.size() - 1];

    // Hardcode the level (temporary) (TODO)
    level = 128;

    return ZKR_SUCCESS;
}

zkresult KeyValueTree::extract (const Goldilocks::Element (&key)[4], const mpz_class &value)
{
    string keyString = fea2string(kvtfr, key);
    return extract(keyString, value);
}

zkresult KeyValueTree::extract (const string &keyString, const mpz_class &value)
{
    // Find the map entry
    unordered_map< string, vector<mpz_class> >::iterator it;
    it = keys.find(keyString);
    if (it == keys.end())
    {
        return ZKR_DB_KEY_NOT_FOUND;
    }

    // There there is at least one value
    if (it->second.size() == 0)
    {
        zklog.error("KeyValueTree::extract() found values size = 0");
        exitProcess();
    }

    // Check the provided value matches the top of the stack
    if (it->second[it->second.size() - 1] != value)
    {
        zklog.error("KeyValueTree::extract() found stored value=" + it->second[it->second.size() - 1].get_str(10) + " != provided value=" + value.get_str(10));
        exitProcess();
    }

    // Delete the last value
    it->second.pop_back();

    // If this was the last value, delete the entry of the keys map
    if (it->second.size() == 0)
    {
        keys.erase(it);
    }

    return ZKR_SUCCESS;
}

uint64_t KeyValueTree::level (const Goldilocks::Element (&key)[4])
{
    return 128;
}