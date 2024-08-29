#include "witness.hpp"
#include "zklog.hpp"
#include "zkassert.hpp"
#include "scalar.hpp"
#include "zkglobals.hpp"
#include "key_utils.hpp"
#include "hashdb_factory.hpp"
#include "cbor.hpp"
#include "utils.hpp"
#include "keccak.hpp"
#include "timer.hpp"

#define WITNESS_CHECK_BITS
//#define WITNESS_CHECK_SMT
//#define LOG_WITNESS

class WitnessContext
{
public:
    const string &witness;
    uint64_t p; // pointer to the first witness byte pending to be parsed
    uint64_t level; // SMT level, being level=0 the root, level>0 higher levels
    DatabaseMap::MTMap &db; // database to store all the hash-value
    DatabaseMap::ProgramMap &programs; // database to store all the programs (smart contracts)
#ifdef WITNESS_CHECK_BITS
    vector<uint8_t> bits; // key bits consumed while climbing the tree; used only for debugging
#endif
#ifdef WITNESS_CHECK_SMT
    Goldilocks::Element root[4]; // the root of the witness data SMT tree; used only for debugging
#endif
    WitnessContext(const string &witness, DatabaseMap::MTMap &db, DatabaseMap::ProgramMap &programs) : witness(witness), p(0), level(0), db(db), programs(programs)
    {
#ifdef WITNESS_CHECK_SMT
        root[0] = fr.zero();
        root[1] = fr.zero();
        root[2] = fr.zero();
        root[3] = fr.zero();
#endif
    }

};

zkresult calculateWitnessHash (WitnessContext &ctx, Goldilocks::Element (&hash)[4])
{
    zkresult zkr;

    // Check level range
    if (ctx.level > 255)
    {
        zklog.error("calculateWitnessHash() reached an invalid level=" + to_string(ctx.level));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }  

#ifdef WITNESS_CHECK_BITS
    // Check level-bits consistency
    if (ctx.level != ctx.bits.size())
    {
        zklog.error("calculateWitnessHash() got level=" + to_string(ctx.level) + "different from bits.size()=" + to_string(ctx.bits.size()));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
#endif

    // Opcode counters to control that we parse CODE at most once, and if so parse another opcode
    uint64_t numberOfOpcodes = 0;
    uint64_t numberOfCodeOpcodes = 0;

    do // while (numberOfOpcodes==1 && numberOfCodeOpcodes==1), i.e. repeat to parse SMT_LEAF after CODE
    {
        // Get instruction opcode from witness
        if (ctx.p >= ctx.witness.size())
        {
            zklog.error("calculateWitnessHash() run out of witness data");
            return ZKR_SM_MAIN_INVALID_WITNESS;
        }
        uint8_t opcode = ctx.witness[ctx.p];
        ctx.p++;

        switch (opcode)
        {
            case 0x02: // BRANCH -> ( 0x02 CBOR(mask)... ); `mask` defines which children are present (e.g. `0000000000001011` means that children 0, 1 and 3 are present and the other ones are not)
            {
                // Get the mask
                uint64_t mask;
                zkr = cbor2u64(ctx.witness, ctx.p, mask);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2u64() result=" + zkresult2string(zkr));
                    return zkr;
                }
#ifdef LOG_WITNESS
                zklog.info("BRANCH level=" + to_string(ctx.level) + " mask=" + to_string(mask));
#endif

                // Get if there are children at the left and/or at the right, from the mask
                bool hasLeft;
                bool hasRight;
                switch (mask)
                {
                    case 1:
                    {
                        hasLeft = true;
                        hasRight = false;
                        break;
                    }
                    case 2:
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
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.emplace_back(0);
#endif
                    ctx.level++;
                    zkr = calculateWitnessHash(ctx, leftHash);
                    ctx.level--;
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.pop_back();
#endif
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
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.emplace_back(1);
#endif
                    ctx.level++;
                    zkr = calculateWitnessHash(ctx, rightHash);
                    ctx.level--;
#ifdef WITNESS_CHECK_BITS
                    ctx.bits.pop_back();
#endif
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
                vector<Goldilocks::Element> valueData;
                valueData.reserve(12);
                for (uint64_t i=0; i<12; i++)
                {
                    valueData.emplace_back(input[i]);
                }
                ctx.db[fea2string(fr, hash)] = valueData;

#ifdef LOG_WITNESS
                zklog.info("BANCH level=" + to_string(ctx.level) + " leftHash=" + fea2string(fr, leftHash) + " rightHash=" + fea2string(fr, rightHash) + " hash=" + fea2string(fr, hash));
#endif

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
                if (ctx.p >= ctx.witness.size())
                {
                    zklog.error("calculateWitnessHash() unexpected end of witness");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }
                uint8_t nodeType = ctx.witness[ctx.p];
                ctx.p++;
                //zklog.info("SMT_LEAF nodeType=" + to_string(nodeType));

                // Read address
                mpz_class address;
                zkr = cbor2scalar(ctx.witness, ctx.p, address);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2scalar(address) result=" + zkresult2string(zkr));
                    return zkr;
                }
                //zklog.info("SMT_LEAF address=" + address.get_str(16));

                // Read storage key
                mpz_class storageKey;
                if (nodeType == 0x03) // SC STORAGE: an extra field storageKey is read
                {
                    zkr = cbor2scalar(ctx.witness, ctx.p, storageKey);
                    if (zkr != ZKR_SUCCESS)
                    {
                        zklog.error("calculateWitnessHash() failed calling cbor2scalar(storageKey) result=" + zkresult2string(zkr));
                        return zkr;
                    }
                    //zklog.info("SMT_LEAF storageKey=" + storageKey.get_str(16));
                }

                // Read value
                mpz_class value;
                zkr = cbor2scalar(ctx.witness, ctx.p, value);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2scalar(value) result=" + zkresult2string(zkr));
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

#ifdef WITNESS_CHECK_SMT
                HashDBInterface * pHashDB = HashDBClientFactory::createHashDBClient(fr,config);
                pHashDB->set("", 0, 0, ctx.root, key, value, PERSISTENCE_TEMPORARY, ctx.root, NULL, NULL);
#endif

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
                vector<Goldilocks::Element> valueData;
                valueData.reserve(12);
                for (uint64_t i=0; i<12; i++)
                {
                    valueData.emplace_back(input[i]);
                }
                ctx.db[fea2string(fr, valueHash)] = valueData;

#ifdef WITNESS_CHECK_BITS
                // Check key
                bool keyBits[256];
                splitKey(fr, key, keyBits);
                for (uint64_t i=0; i<ctx.level; i++)
                {
                    if (keyBits[i] != ctx.bits[i])
                    {
                        zklog.error("calculateWitnessHash() found different keyBits[i]=" + to_string(keyBits[i]) + " bits[i]=" + to_string(ctx.bits[i]) + " i=" + to_string(i));
                        zklog.error("bits=");
                        for (uint64_t b=0; b<ctx.bits.size(); b++)
                        {
                            zklog.error(" b=" + to_string(b) + " keyBits=" + to_string(keyBits[b]) + " bits=" + to_string(ctx.bits[b]));
                        }

                        return ZKR_SM_MAIN_INVALID_WITNESS;
                    }
                }
#endif

                // Calculate the remaining key
                Goldilocks::Element rkey[4];
                removeKeyBits(fr, key, ctx.level, rkey);

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
                ctx.db[fea2string(fr, hash)] = valueData;

#ifdef LOG_WITNESS
                zklog.info("LEAF level=" + to_string(ctx.level) + " address=" + address.get_str(16) + " type=" + to_string(nodeType) + " storageKey=" + storageKey.get_str(16) + " value=" + value.get_str(16) + " key=" + fea2string(fr, key) + " rkey=" + fea2string(fr, rkey) + " valueHash=" + fea2string(fr, valueHash) + " hash=" + fea2string(fr, hash));
#endif

                break;
            }
            case 0x03: // HASH -> ( 0x03 hash_byte_1 ... hash_byte_32 )
            {
                // Read node hash
                mpz_class hashScalar;
                if (ctx.p + 32 > ctx.witness.size())
                {
                    zklog.error("calculateWitnessHash() run out of witness data");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }
                ba2scalar((const uint8_t *)ctx.witness.c_str() + ctx.p, 32, hashScalar);
                ctx.p += 32;

#ifdef LOG_WITNESS
                zklog.info("HASH hash=" + hashScalar.get_str(16));
#endif

                // Convert to field elements
                scalar2fea(fr, hashScalar, hash); // TODO: return error if hashScalar is invalid, instead of killing the process

                break;
            }
            case 0x04: // CODE -> ( 0x04 CBOR(code)... )
            {
                // Check we parse CODE once, at most
                if (numberOfCodeOpcodes >= 1)
                {
                    zklog.error("calculateWitnessHash() found 2 consecutive CODE opcodes");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }

                string program;

                // Parse CBOR data
                zkr = cbor2ba(ctx.witness, ctx.p, program);
                if (zkr != ZKR_SUCCESS)
                {
                    zklog.error("calculateWitnessHash() failed calling cbor2ba(program) result=" + zkresult2string(zkr));
                    return zkr;
                }
                if (program.empty())
                {
                    zklog.error("calculateWitnessHash() called cbor2ba(program) and got an empty byte array");
                    return ZKR_SM_MAIN_INVALID_WITNESS;
                }

                // Convert to vector
                vector<uint8_t> programVector;
                programVector.reserve(program.size());
                for (uint64_t i=0; i<program.size(); i++)
                {
                    programVector.emplace_back(program[i]);
                }

                // Calculate hash
                Goldilocks::Element linearHash[4];
                poseidonLinearHash(programVector, linearHash);

                // Save into programs
                string linearHashString = fea2string(fr, linearHash);
                ctx.programs[linearHashString] = programVector;

#ifdef LOG_WITNESS
                zklog.info("CODE size=" + to_string(program.size()) + " hash=" + linearHashString + " code=" + ba2string(program));
#endif

                numberOfCodeOpcodes++;

                break;
            }
            case 0x00: // LEAF -> ( 0x00 CBOR(ENCODE_KEY(key))... CBOR(value)... )
            case 0x01: // EXTENSION -> ( 0x01 CBOR(ENCODE_KEY(key))... )
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

        // Increment number of parsed opcodes
        numberOfOpcodes++;

    } while ((numberOfOpcodes == 1) && (numberOfCodeOpcodes == 1));

#ifdef LOG_WITNESS
    zklog.info("calculateWitnessHash() returns hash=" + fea2string(fr, hash));
#endif

    return ZKR_SUCCESS;
}

zkresult witness2db (const string &witness, DatabaseMap::MTMap &db, DatabaseMap::ProgramMap &programs, mpz_class &stateRoot)
{
    struct timeval t;
    gettimeofday(&t, NULL);

    db.clear();
    programs.clear();
    
    zkresult zkr;

    // Check witness is not empty
    if (witness.empty())
    {
        zklog.error("witness2db() got an empty witness");
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }

    // Create witness context
    WitnessContext ctx(witness, db, programs);

    // Parse header version
    uint8_t headerVersion = ctx.witness[ctx.p];
    if (headerVersion != 1)
    {
        zklog.error("witness2db() expected headerVersion=1 but got value=" + to_string(headerVersion));
        return ZKR_SM_MAIN_INVALID_WITNESS;
    }
    ctx.p++;

    // Calculate witness hash    
    Goldilocks::Element hash[4];
    zkr = calculateWitnessHash(ctx, hash);
    if (zkr != ZKR_SUCCESS)
    {
        zklog.error("witness2db() failed calling calculateWitnessHash() result=" + zkresult2string(zkr));
        return zkr;
    }

    // Convert state root
    fea2scalar(fr, stateRoot, hash);

    zklog.info("witness2db() calculated stateRoot=" + stateRoot.get_str(16) + " from size=" + to_string(witness.size()) + "B generating db.size=" + to_string(db.size()) + " and programs.size=" + to_string(programs.size()) + " in " + to_string(TimeDiff(t)) + "us");

#ifdef WITNESS_CHECK_SMT
    zklog.info("witness2db() calculated SMT root=" + fea2string(fr, ctx.root));
#endif

    return ZKR_SUCCESS;
}
