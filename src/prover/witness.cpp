#include "witness.hpp"
#include "zklog.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "zkglobals.hpp"
#include "key_utils.hpp"

zkresult cborU64 (const string &s, uint64_t &p, uint64_t &value)
{
    if (p >= s.size())
    {
        zklog.error("cborU64() found too high p");
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
    uint8_t firstByte = s[p];
    p++;
    if (firstByte < 24)
    {
        value = firstByte;
        return ZKR_SUCCESS;
    }
    zklog.error("cborU64() found unexpected firstByte=" + to_string(firstByte));
    return ZKR_SM_MAIN_INVALID_WITNESS;
}

zkresult cborScalar (const string &s, uint64_t &p, mpz_class &value)
{
    if (p >= s.size())
    {
        zklog.error("cborScalar() found too high p");
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
    uint8_t firstByte = s[p];
    p++;
    if (firstByte < 24)
    {
        value = firstByte;
        return ZKR_SUCCESS;
    }
    uint8_t majorType = firstByte >> 5;
    uint8_t shortCount = firstByte & 0x1F;

    uint64_t longCount;
    if (shortCount <= 23)
    {
        longCount = shortCount;
    }
    else if (shortCount == 24)
    {
        if (p >= s.size())
        {
            zklog.error("cborScalar() run out of bytes");
            return ZKR_SM_MAIN_INVALID_WITNESS;
        }
        uint8_t secondByte = s[p];
        p++;
        longCount = secondByte;
    }
    else if (shortCount == 25)
    {
        if (p + 1 >= s.size())
        {
            zklog.error("cborScalar() run out of bytes");
            return ZKR_SM_MAIN_INVALID_WITNESS;
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
            zklog.error("cborScalar() run out of bytes");
            return ZKR_SM_MAIN_INVALID_WITNESS;
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
            zklog.error("cborScalar() run out of bytes");
            return ZKR_SM_MAIN_INVALID_WITNESS;
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
            value = shortCount;
            break;
        }

        // For types 2 (byte string) and 3 (text string), the count is the length of the payload
        case 2: // byte string
        {
            if ((p + longCount) > s.size())
            {
                zklog.error("cborScalar() not enough space left for longCount=" + to_string(longCount));
                return ZKR_SM_MAIN_INVALID_WITNESS;
            }
            ba2scalar((const uint8_t *)s.c_str() + p, longCount, value);
            p += longCount;
            break;
        }
        case 3: // text string
        {
            zklog.error("cborScalar() majorType=3 (text string) not supported");
            return ZKR_SM_MAIN_INVALID_WITNESS;
        }

        // For types 4 (array) and 5 (map), the count is the number of items (pairs) in the payload
        case 4: // array
        {
            zkassert(false);
        }
        case 5: // map
        {
            zkassert(false);
        }

        // For type 6 (tag), the payload is a single item and the count is a numeric tag number which describes the enclosed item
        case 6: // tag
        {
            zkassert(false);
        }
    }
    
    //zklog.info("cborScalar() got value=" + value.get_str(16));
    return ZKR_SUCCESS;
}

zkresult calculateWitnessHash (const string &witness, uint64_t &p, uint64_t level, DatabaseMap::MTMap &db, Goldilocks::Element (&hash)[4])
{
    zkresult zkr;

    // Check level range
    if (level > 255)
    {
        zklog.error("calculateWitnessHash() reached an invalid level=" + to_string(level));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
    
    // Get instruction opcode from witness
    if (p >= witness.size())
    {
        zklog.error("calculateWitnessHash() run out of witness data");
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
    uint8_t opcode = witness[p];
    p++;

    switch (opcode)
    {
        case 0x02: // BRANCH -> ( 0x02 CBOR(mask)... ); `mask` defines which children are present (e.g. `0000000000001011` means that children 0, 1 and 3 are present and the other ones are not)
        {
            // Get the mask
            uint64_t mask;
            zkr = cborU64(witness, p, mask);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("calculateWitnessHash() failed calling cborU64() result=" + zkresult2string(zkr));
                return zkr;
            }
            zklog.info("BRANCH level=" + to_string(level) + " mask=" + to_string(mask));

            // Get if there are children at the left and/or at the right, from the mask
            bool hasLeft;
            bool hasRight;
            switch (mask)
            {
                case 2:
                {
                    hasLeft = true;
                    hasRight = false;
                    break;
                }
                case 1:
                {
                    hasLeft = false;
                    hasRight = true;
                    break;
                }
                case 3:
                {
                    hasLeft = true;
                    hasRight = true;
                    break;
                }
                default:
                {
                    zklog.error("calculateWitnessHash() found invalid mask=" + to_string(mask));
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }
            }

            // Calculate the left hash
            Goldilocks::Element leftHash[4];
            if (hasLeft)
            {
                zkr = calculateWitnessHash(witness, p, level + 1, db, leftHash);
                if (zkr != ZKR_SUCCESS)
                {
                    return zkr;
                }
            }
            else
            {
                leftHash[0] = fr.zero();
                leftHash[1] = fr.zero();
                leftHash[2] = fr.zero();
                leftHash[3] = fr.zero();
            }

            // Calculate the right hash
            Goldilocks::Element rightHash[4];
            if (hasRight)
            {
                zkr = calculateWitnessHash(witness, p, level + 1, db, rightHash);
                if (zkr != ZKR_SUCCESS)
                {
                    return zkr;
                }
            }
            else
            {
                rightHash[0] = fr.zero();
                rightHash[1] = fr.zero();
                rightHash[2] = fr.zero();
                rightHash[3] = fr.zero();
            }

            // Calculate this intermediate node hash = poseidonHash(leftHash, rightHash, 0000)

            // Prepare input = [leftHash, rightHash, 0000]
            Goldilocks::Element input[12];
            input[0] = leftHash[0];
            input[1] = leftHash[1];
            input[2] = leftHash[2];
            input[3] = leftHash[3];
            input[4] = rightHash[0];
            input[5] = rightHash[1];
            input[6] = rightHash[2];
            input[7] = rightHash[3];
            input[8] = fr.zero();
            input[9] = fr.zero();
            input[10] = fr.zero();
            input[11] = fr.zero();

            // Calculate the poseidon hash
            poseidon.hash(hash, input);

            // Store the hash-value pair into db
            vector<Goldilocks::Element> valueData(12);
            for (uint64_t i=0; i<12; i++)
            {
                valueData.emplace_back(input[i]);
            }
            db[fea2string(fr, hash)] = valueData;

            zklog.info("BANCH level=" + to_string(level) + " leftHash=" + fea2string(fr, leftHash) + " rightHash=" + fea2string(fr, rightHash) + " hash=" + fea2string(fr, hash));

            break;
        }
        case 0x07: // SMT_LEAF -> ( 0x07 nodeType CBOR(address) /CBOR(storageKey).../ CBOR(value)...)
            // * if `nodeType` == `0x03`, then an extra field `storageKey` is read; otherwise it is skipped
        {
            // Read nodeType
            // 0 = BALANCE
            // 1 = NONCE
            // 2 = SC CODE
            // 3 = SC STORAGE
            // 4 = SC LENGTH
            // 5, 6 = touched addresses
            // < 11 (0xb) = info block tree of Etrog
            if (p >= witness.size())
            {
                zklog.error("calculateWitnessHash() unexpected end of witness");
                return ZKR_SM_MAIN_INVALID_WITNESS;
            }
            uint8_t nodeType = witness[p];
            p++;
            //zklog.info("SMT_LEAF nodeType=" + to_string(nodeType));

            // Read address
            mpz_class address;
            zkr = cborScalar(witness, p, address);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("calculateWitnessHash() failed calling cborScalar(address) result=" + zkresult2string(zkr));
                return zkr;
            }
            //zklog.info("SMT_LEAF address=" + address.get_str(16));

            // Read storage key
            mpz_class storageKey;
            if (nodeType == 0x03) // an extra field storageKey is read
            {
                zkr = cborScalar(witness, p, storageKey);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cborScalar(storageKey) result=" + zkresult2string(zkr));
                    return zkr;
                }
                //zklog.info("SMT_LEAF storageKey=" + storageKey.get_str(16));
            }

            // Read value
            mpz_class value;
            zkr = cborScalar(witness, p, value);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("calculateWitnessHash() failed calling cborScalar(value) result=" + zkresult2string(zkr));
                return zkr;
            }
            //zklog.info("SMT_LEAF value=" + value.get_str(16));

            // Calculate poseidonHash(storageKey)
            // TODO: skip if storageKey==0, use pre-calculated poseidon hash of zero
            Goldilocks::Element Kin0[12];
            scalar2fea(fr, storageKey, Kin0[0], Kin0[1], Kin0[2], Kin0[3], Kin0[4], Kin0[5], Kin0[6], Kin0[7]);
            Kin0[8] = fr.zero();
            Kin0[9] = fr.zero();
            Kin0[10] = fr.zero();
            Kin0[11] = fr.zero();
            Goldilocks::Element Kin0Hash[4];
            poseidon.hash(Kin0Hash, Kin0);

            // Calculate the key = poseidonHash(account, type, poseidonHash(storageKey))
            Goldilocks::Element Kin1[12];
            scalar2fea(fr, address, Kin1[0], Kin1[1], Kin1[2], Kin1[3], Kin1[4], Kin1[5], Kin1[6], Kin1[7]);
            if (!fr.isZero(Kin1[5]) || !fr.isZero(Kin1[6]) || !fr.isZero(Kin1[7]))
            {
                zklog.error("calculateWitnessHash() found non-zero address field elements 5, 6 or 7");
                return ZKR_SM_MAIN_INVALID_WITNESS;
            }

            switch(nodeType)
            {
                case 0: // BALANCE
                {
                    break;
                }
                case 1: // NONCE
                {
                    Kin1[6] = fr.one();
                    break;
                }
                case 2: // SC CODE
                {
                    Kin1[6] = fr.fromU64(2);
                    break;
                }
                case 3: // SC STORAGE
                {
                    Kin1[6] = fr.fromU64(3);
                    break;
                }
                case 4: // SC LENGTH
                {
                    Kin1[6] = fr.fromU64(4);
                    break;
                }
                default:
                {
                    zklog.error("calculateWitnessHash() found invalid nodeType=" + to_string(nodeType));
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }
            }

            // Reinject the first resulting hash as the capacity for the next poseidon hash
            Kin1[8] = Kin0Hash[0];
            Kin1[9] = Kin0Hash[1];
            Kin1[10] = Kin0Hash[2];
            Kin1[11] = Kin0Hash[3];

            // Call poseidon hash
            Goldilocks::Element key[4];
            poseidon.hash(key, Kin1);

            // Calculate this leaf node hash = poseidonHash(remainingKey, valueHash, 1000),
            // where valueHash = poseidonHash(value, 0000)

            // Prepare input = [value8, 0000]
            Goldilocks::Element input[12];
            scalar2fea(fr, value, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
            input[8] = fr.zero();
            input[9] = fr.zero();
            input[10] = fr.zero();
            input[11] = fr.zero();

            // Calculate the value hash
            Goldilocks::Element valueHash[4];
            poseidon.hash(valueHash, input);

            // Store the hash-value pair into db
            vector<Goldilocks::Element> valueData(12);
            for (uint64_t i=0; i<12; i++)
            {
                valueData.emplace_back(input[i]);
            }
            db[fea2string(fr, valueHash)] = valueData;

            // Calculate the remaining key
            Goldilocks::Element rkey[4];
            removeKeyBits(fr, key, level, rkey);

            // Prepare input = [rkey, valueHash, 1000]
            input[0] = rkey[0];
            input[1] = rkey[1];
            input[2] = rkey[2];
            input[3] = rkey[3];
            input[4] = valueHash[0];
            input[5] = valueHash[1];
            input[6] = valueHash[2];
            input[7] = valueHash[3];
            input[8] = fr.one();
            input[9] = fr.zero();
            input[10] = fr.zero();
            input[11] = fr.zero();

            // Calculate the leaf node hash
            poseidon.hash(hash, input);

            // Store the hash-value pair into db
            for (uint64_t i=0; i<12; i++)
            {
                valueData[i] = input[i];
            }
            db[fea2string(fr, hash)] = valueData;

            zklog.info("LEAF level=" + to_string(level) + " address=" + address.get_str(16) + " type=" + to_string(nodeType) + " storageKey=" + storageKey.get_str(16) + " value=" + value.get_str(16) + " key=" + fea2string(fr, key) + " rkey=" + fea2string(fr, rkey) + " value=" + value.get_str(16) + " valueHash=" + fea2string(fr, valueHash) + " hash=" + fea2string(fr, hash));

            break;
        }
        case 0x03: // HASH -> ( 0x03 hash_byte_1 ... hash_byte_32 )
        {
            // Read node hash
            mpz_class hashScalar;
            zkr = cborScalar(witness, p, hashScalar);
            if (zkr != ZKR_SUCCESS)
            {
                zklog.error("calculateWitnessHash() failed calling cborScalar(hashScalar) result=" + zkresult2string(zkr));
                return zkr;
            }
            zklog.info("HASH hash=" + hashScalar.get_str(16));

            // Convert to field elements
            scalar2fea(fr, hashScalar, hash); // TODO: return error if hashScalar is invalid, instead of killing the process

            break;
        }
        case 0x00: // LEAF -> ( 0x00 CBOR(ENCODE_KEY(key))... CBOR(value)... )
        case 0x01: // EXTENSION -> ( 0x01 CBOR(ENCODE_KEY(key))... )
        case 0x04: // CODE -> ( 0x04 CBOR(code)... )
        case 0x05: // ACCOUNT_LEAF -> ( 0x05 CBOR(ENCODE_KEY(key))... flags /CBOR(nonce).../ /CBOR(balance).../ )
            // `flags` is a bitset encoded in a single byte (bit endian):
            // * bit 0 defines if **code** is present; if set to 1, then `has_code=true`;
            // * bit 1 defines if **storage** is present; if set to 1, then `has_storage=true`;
            // * bit 2 defines if **nonce** is not 0; if set to 0, *nonce* field is not encoded;
            // * bit 3 defines if **balance** is not 0; if set to 0, *balance* field is not encoded;
        case 0xBB: // NEW_TRIE -> ( 0xBB )
        default:
        {
            zklog.error("calculateWitnessHash() got unsupported opcode=" + to_string(opcode));
            return ZKR_SM_MAIN_INVALID_WITNESS;
        }
    }

    return ZKR_SUCCESS;
}

zkresult witness2db (const string &witness, DatabaseMap::MTMap &db, string &stateRoot)
{
    zkresult zkr;

    // Check witness is not empty
    if (witness.empty())
    {
        zklog.error("witness2db() got an empty witness");
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }

    // Witness position counter
    uint64_t p = 0;

    // Parse header version
    uint8_t headerVersion = witness[p];
    if (headerVersion != 1)
    {
        zklog.error("witness2db() expected headerVersion=1 but got value=" + to_string(headerVersion));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
    p++;

    // Calculate witness hash
    Goldilocks::Element hash[4];
    zkr = calculateWitnessHash(witness, p, 0, db, hash);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("witness2db() failed calling calculateWitnessHash() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Convert state root
    stateRoot = fea2string(fr, hash);

    zklog.info("witness2db() calculated stateRoot=" + stateRoot);

    return ZKR_SUCCESS;
}
