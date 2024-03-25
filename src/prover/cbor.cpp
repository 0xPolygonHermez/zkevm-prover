#include "cbor.hpp"

string cborType2string (CborResult::ResultType type)
{
    switch (type)
    {
        case CborResult::UNDEFINED: return "UNDEFINED";
        case CborResult::U64: return "U64";
        case CborResult::BA: return "BA";
        case CborResult::TEXT: return "TEXT";
        case CborResult::ARRAY: return "ARRAY";
        case CborResult::TAG: return "TAG";
        default:
        {
            return "<UNRECOGNIZED TYPE=" + to_string(type) + ">"; 
        }
    }
}

// This function parses CBOR field and stores it in a CborResult
void cbor2result (const string &s, uint64_t &p, CborResult &cborResult)
{
    if (p >= s.size())
    {
        zklog.error("cbor2result() found too high p");
        cborResult.result= ZKR_CBOR_INVALID_DATA;
    }
    uint8_t firstByte = s[p];
    p++;
    if (firstByte < 24)
    {
        cborResult.type = CborResult::U64;
        cborResult.u64 = firstByte;
        cborResult.result = ZKR_SUCCESS;
        return;
    }
    uint8_t majorType = firstByte >> 5;
    uint8_t shortCount = firstByte & 0x1F;

    uint64_t longCount = 0;
    if (shortCount <= 23)
    {
        longCount = shortCount;
    }
    else if (shortCount == 24)
    {
        if (p >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        longCount = secondByte;
    }
    else if (shortCount == 25)
    {
        if (p + 1 >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        uint8_t thirdByte = s[p];
        p++;
        longCount = (uint64_t(secondByte)<<8) + uint64_t(thirdByte);
    }
    else if (shortCount == 26)
    {
        if (p + 3 >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        uint8_t thirdByte = s[p];
        p++;
        uint8_t fourthByte = s[p];
        p++;
        uint8_t fifthByte = s[p];
        p++;
        longCount = (uint64_t(secondByte)<<24) + (uint64_t(thirdByte)<<16) + (uint64_t(fourthByte)<<8) + uint64_t(fifthByte);
    }
    else if (shortCount == 27)
    {
        if (p + 7 >= s.size())
        {
            zklog.error("cbor2result() run out of bytes");
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }
        uint8_t secondByte = s[p];
        p++;
        uint8_t thirdByte = s[p];
        p++;
        uint8_t fourthByte = s[p];
        p++;
        uint8_t fifthByte = s[p];
        p++;
        uint8_t sixthByte = s[p];
        p++;
        uint8_t seventhByte = s[p];
        p++;
        uint8_t eighthByte = s[p];
        p++;
        uint8_t ninethByte = s[p];
        p++;
        longCount = (uint64_t(secondByte)<<56) + (uint64_t(thirdByte)<<48) + (uint64_t(fourthByte)<<40) + (uint64_t(fifthByte)<<32) + (uint64_t(sixthByte)<<24) + (uint64_t(seventhByte)<<16) + (uint64_t(eighthByte)<<8) + uint64_t(ninethByte);
    }

    switch (majorType)
    {
        // Assuming CBOR short field encoding
        // For types 0, 1, and 7, there is no payload; the count is the value
        case 0:
        case 1:
        case 7:
        {
            cborResult.type = CborResult::U64;
            cborResult.u64 = shortCount;
            cborResult.result = ZKR_SUCCESS;
            break;
        }

        // For types 2 (byte string) and 3 (text string), the count is the length of the payload
        case 2: // byte string
        {
            if ((p + longCount) > s.size())
            {
                zklog.error("cbor2result() not enough space left for longCount=" + to_string(longCount));
                cborResult.result = ZKR_CBOR_INVALID_DATA;
                return;
            }
            cborResult.ba = s.substr(p, longCount);
            p += longCount;
            cborResult.type = CborResult::BA;
            cborResult.result = ZKR_SUCCESS;
            break;
        }
        case 3: // text string
        {
            if ((p + longCount) > s.size())
            {
                zklog.error("cbor2result() not enough space left for longCount=" + to_string(longCount));
                cborResult.result = ZKR_CBOR_INVALID_DATA;
                return;
            }
            cborResult.text = s.substr(p, longCount);
            p += longCount;
            cborResult.type = CborResult::TEXT;
            cborResult.result = ZKR_SUCCESS;
            break;
        }

        // For types 4 (array) and 5 (map), the count is the number of items (pairs) in the payload
        case 4: // array
        {
            //zklog.info("cbor2result() starting array of " + to_string(longCount) + " elements");
            //zklog.info(" data=" + ba2string(s.substr(p-1)));
            for (uint64_t a=0; a<longCount; a++)
            {
                CborResult result;
                cbor2result(s, p, result);
                if (result.result != ZKR_SUCCESS)
                {
                    zklog.error("cbor2result() found an array and failed calling itself a=" + to_string(a) + " result=" + zkresult2string(result.result));
                    cborResult.result = result.result;
                    return;
                }
                cborResult.array.emplace_back(result);
            }
            cborResult.type = CborResult::ARRAY;
            cborResult.result = ZKR_SUCCESS;
            //zklog.info("cbor2result() ending array of " + to_string(longCount) + " elements");
            break;
        }
        case 5: // map
        {
            zklog.error("cbor2result() majorType=5 (map) not supported longCount=" + to_string(longCount));
            cborResult.result = ZKR_CBOR_INVALID_DATA;
            return;
        }

        // For type 6 (tag), the payload is a single item and the count is a numeric tag number which describes the enclosed item
        case 6: // tag
        {
            //zklog.info("cbor2result() majorType=6 (tag)");
            CborResult result;
            cbor2result(s, p, result);
            if (result.result != ZKR_SUCCESS)
            {
                zklog.error("cbor2result() TAG failed calling itself result=" + zkresult2string(result.result));
                cborResult.result = result.result;
                return;
            }
            cborResult.tagCount = longCount;
            cborResult.tag.emplace_back(result);
            cborResult.type = CborResult::TAG;
            cborResult.result = ZKR_SUCCESS;
            break;
        }
    }
    
    //zklog.info("cbor2result() got result=" + zkresult2string(cborResult.result) + " type=" + cborType2string(cborResult.type));
}

// This CBOR function expects a simple integer < 24; otherwise it fails
zkresult cbor2u64 (const string &s, uint64_t &p, uint64_t &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2u64() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::U64:
        {
            value = cborResult.u64;
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2u64() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}

// This CBOR function expects a byte array; otherwise it fails
zkresult cbor2ba (const string &s, uint64_t &p, string &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2ba() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::BA:
        {
            value = cborResult.ba;
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2ba() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}

// This CBOR function expects a text string; otherwise it fails
zkresult cbor2text (const string &s, uint64_t &p, string &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2text() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::TEXT:
        {
            value = cborResult.text;
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2text() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}

// This function expects an integer, which can be long, and returns a scalar
zkresult cbor2scalar (const string &s, uint64_t &p, mpz_class &value)
{
    CborResult cborResult;
    cbor2result(s, p, cborResult);
    if (cborResult.result != ZKR_SUCCESS)
    {
        zklog.error("cbor2scalar() failed calling cbor2result() result=" + zkresult2string(cborResult.result));
        return cborResult.result;
    }
    switch (cborResult.type)
    {
        case CborResult::U64:
        {
            value = cborResult.u64;
            return ZKR_SUCCESS;
        }
        case CborResult::BA:
        {
            if (cborResult.ba.size() > 32)
            {
                zklog.error("cbor2scalar() got size too long size=" + to_string(cborResult.ba.size()));
                return ZKR_CBOR_INVALID_DATA;
            }
            ba2scalar(value, cborResult.ba);
            return ZKR_SUCCESS;
        }
        default:
        {
            zklog.error("cbor2scalar() called cbor2result() and got invalid result type=" + to_string(cborResult.type));
            return ZKR_CBOR_INVALID_DATA;
        }
    }
}