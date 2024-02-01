#include "rlp.hpp"
#include "zklog.hpp"

using namespace rlp;
using namespace std;

bool rlp::decodeLength (const string &input, uint64_t &p, uint64_t &length, bool &list)
{
    // Read the first byte, prefix
    if (p > input.size())
    {
        zklog.error("rlp::decodeLength() run out of data");
        return false;
    }
    uint8_t prefix = input[p];

    // The first byte is the data itself
    if (prefix <= 0x7f)
    {
        length = 1;
        list = false;
        return true;
    }

    // Take length from first byte
    if (prefix <= 0xb7)
    {
        length = prefix - 0x80;
        p++;
        if ((p + length) > input.size())
        {
            zklog.error("rlp::decodeLength() run out of data");
            return false;
        }
        list = false;
        return true;
    }

    // Take the length of the length from first byte
    if (prefix <= 0xbf)
    {
        uint64_t lengthOfLength = prefix - 0xb7;
        p++;
        if (lengthOfLength > 8)
        {
            zklog.error("rlp::decodeLength() invalid length of length=" + to_string(lengthOfLength));
            return false;
        }
        if (p + lengthOfLength > input.size())
        {
            zklog.error("rlp::decodeLength() run out of data");
            return false;
        }
        length = 0;
        for (uint64_t i = 0; i < lengthOfLength; i++)
        {
            uint8_t d = input[p];
            p++;
            length <<= 8;
            length += d;
        }
        list = false;
        return true;
    }

    // Take list length from first byte
    if (prefix <= 0xf7)
    {
        length = prefix - 0xc0;
        p++;
        if ((p + length) > input.size())
        {
            zklog.error("rlp::decodeLength() run out of data");
            return false;
        }
        list = true;
        return true;
    }

    // Take the length of the length from first byte (list)
    {
        uint64_t lengthOfLength = prefix - 0xf7;
        p++;
        if (lengthOfLength > 8)
        {
            zklog.error("rlp::decodeLength() invalid length of length=" + to_string(lengthOfLength));
            return false;
        }
        if (p + lengthOfLength > input.size())
        {
            zklog.error("rlp::decodeLength() run out of data");
            return false;
        }
        length = 0;
        for (uint64_t i = 0; i < lengthOfLength; i++)
        {
            uint8_t d = input[p];
            p++;
            length <<= 8;
            length += d;
        }
        list = true;
        return true;
    }
}

bool rlp::decodeBa (const string &input, uint64_t &p, string &output, bool &list)
{
    // Check the counter
    if (p > input.size())
    {
        zklog.error("rlp::decodeBa() run out of data");
        return false;
    }

    // Clear the output
    output.clear();

    // Decode the length
    uint64_t length;
    bool bResult;
    bResult = rlp::decodeLength(input, p, length, list);
    if (!bResult)
    {
        zklog.error("rlp::decodeBa() failed calling rlp::decodeLength() p=" + to_string(p));
        return false;        
    }

    // Copy the byte array
    output = input.substr(p, length);
    p += length;

    return true;
}

bool rlp::decodeList (const string &input, std::vector<string> &output)
{
    // Decode the list length
    bool bResult;
    bool list;
    uint64_t p=0;
    uint64_t length;
    bResult = decodeLength(input, p, length, list);
    if (!bResult)
    {
        zklog.error("rlp::decodeList() failed calling rlp::decodeLength() p=" + to_string(p));
        return false;
    }
    if (!list)
    {
        zklog.error("rlp::decodeList() called rlp::decodeLength() but list=false p=" + to_string(p));
        return false;
    }
    if ((p + length) != input.size())
    {
        zklog.error("rlp::decodeList() called rlp::decodeLength() but length=" + to_string(length) + " input.size=" + to_string(input.size()) + " p=" + to_string(p) + " and p + length != input.size");
        return false;
    }

    // Decode the list raw data content searching for byte arrays
    while (p < input.size())
    {
        string ba;
        bResult = decodeBa(input, p, ba, list);
        if (!bResult)
        {
            zklog.error("rlp::decodeList() failed calling rlp::decodeBa() p=" + to_string(p));
            return false;
        }
        if (list)
        {
            zklog.error("rlp::decodeList() called rlp::decodeBa() but list=false p=" + to_string(p));
            return false;
        }
        output.push_back(ba);
    }

    // Check the counter
    if (p != input.size())
    {
        zklog.error("rlp::decodeList() finished with input.size=" + to_string(input.size()) + " p=" + to_string(p) + " and p != input.size");
        return false;
    }

    return true;
}

bool rlp::encodeLength (uint64_t length, bool list, string &output)
{
    // Set the offset based on list
    uint8_t offset = list ? 0xc0 : 0x80;

    // Encode length in one byte
    if (length < 56)
    {
        uint8_t d = offset + length;
        output.push_back(d);
        return true;
    }

    // Encode length in two bytes
    if (length < 256)
    {
        string lengthString;
        uint8_t * pLength = (uint8_t *)&length;
        lengthString.push_back(pLength[0]);
        uint8_t d = offset + 55 + lengthString.size();
        output.push_back(d);
        output += lengthString;
        return true;
    }

    // Encode length in three bytes
    if (length < 256*256)
    {
        string lengthString;
        uint8_t * pLength = (uint8_t *)&length;
        lengthString.push_back(pLength[1]);
        lengthString.push_back(pLength[0]);
        uint8_t d = offset + 55 + lengthString.size();
        output.push_back(d);
        output += lengthString;
        return true;
    }

    // Encode length in four bytes
    if (length < 256*256*256)
    {
        string lengthString;
        uint8_t * pLength = (uint8_t *)&length;
        lengthString.push_back(pLength[2]);
        lengthString.push_back(pLength[1]);
        lengthString.push_back(pLength[0]);
        uint8_t d = offset + 55 + lengthString.size();
        output.push_back(d);
        output += lengthString;
        return true;
    }

    // Enough; longer lengths will fail
    zklog.error("rlp::encodeLength() found length too long=" + to_string(length));
    return false;
}

bool rlp::encodeBa (const string &input, string &output)
{
    // Clear the output
    output.clear();

    // Simple case: single byte, low value
    if (input.size() == 1)
    {
        uint8_t d = input[0];
        if (d < 0x80)
        {
            output = input;
            return true;
        }
    }

    // Encode the length
    bool bResult;
    bResult = encodeLength(input.size(), false, output);
    if (!bResult)
    {
        zklog.error("rlp::encodeBa() failed calling encodeLength()");
        return false;
    }

    // Concatenate the byte array
    output += input;

    return true;
}

bool rlp::encodeList (const std::vector<string> &input, string &output)
{
    // Build the list raw data by encoding all byte arrays, and concatenate them
    bool bResult;
    string list;
    for (uint64_t i=0; i<input.size(); i++)
    {
        // Encode this byte array
        string ba;
        bResult = encodeBa(input[i], ba);
        if (!bResult)
        {
            zklog.error("rlp::encodeList() failed calling encodeBa()");
            return false;
        }

        // Concatenate to the previous ones
        list += ba;        
    }

    // Encode the length of the resulting list raw data
    bResult = encodeLength(list.size(), true, output);
    if (!bResult)
    {
        zklog.error("rlp::encodeList() failed calling encodeLength()");
        return false;
    }

    // Concatenate the list raw data
    output += list;
    
    return true;
}