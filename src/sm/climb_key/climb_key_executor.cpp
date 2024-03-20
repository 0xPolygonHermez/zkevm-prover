#include "climb_key_executor.hpp"
#include "utils.hpp"
#include "scalar.hpp"
#include "zklog.hpp"

#define CLIMB_KEY_CLOCKS       4
#define CLIMB_KEY_LAST_CLOCK   (CLIMB_KEY_CLOCKS - 1)
#define CLIMB_KEY_RESULT_CLOCK (CLIMB_KEY_CLOCKS - 2)

#define LIMIT_BEFORE_CLIMB_UP (GOLDILOCKS_PRIME >> 1)

void ClimbKeyExecutor::execute (vector<ClimbKeyAction> &input, ClimbKeyCommitPols &pols)
{
    static const uint64_t GL_CHUNKS[CLIMB_KEY_CLOCKS] = {0x00001, 0x3C000, 0x3FFFF, 0x003FF};
    static const uint64_t CHUNK_MASKS[CLIMB_KEY_CLOCKS] = {0x3FFFF, 0x3FFFF, 0x3FFFF, 0x003FF};
    static const uint64_t CHUNK_FACTORS[CLIMB_KEY_CLOCKS] = {1ULL, 1ULL << 18, 1ULL << 36, 1ULL << 54};

    // Check input size
    if (input.size()* CLIMB_KEY_CLOCKS > N)
    {
        zklog.error("ClimbKeyExecutor::execute() Too many entries input.size()=" + to_string(input.size()) + " > N/" + to_string(CLIMB_KEY_CLOCKS) + "4=" + to_string(N/CLIMB_KEY_CLOCKS));
        exitProcess();
    }

    const Goldilocks::Element two = fr.fromU64(2);

    for (uint64_t i=0; i<input.size(); i++)
    {
        uint64_t level = input[i].level;
        uint64_t zlevel = level % 4;
        uint8_t bit = input[i].bit;

        uint64_t value = fr.toU64(input[i].key[zlevel]);

        uint8_t carry = bit;
        uint8_t lt = 0;

        const Goldilocks::Element glevel = fr.fromU64(level);
        const Goldilocks::Element gbit = fr.fromU64(bit);

        for (uint8_t clock = 0; clock < CLIMB_KEY_CLOCKS; ++clock) {
            const uint64_t row = i * CLIMB_KEY_CLOCKS + clock;
            const uint64_t chunkValue = value & 0x3FFFF;
            const uint64_t chunkValueClimbed = chunkValue * 2 + carry;

            value = value >> 18;

            if (clock == CLIMB_KEY_LAST_CLOCK) {
                pols.key0[row] = (zlevel == 0) ? fr.add(fr.mul(input[i].key[0], two), fr.fromU64(bit)) : input[i].key[0];
                pols.key1[row] = (zlevel == 1) ? fr.add(fr.mul(input[i].key[1], two), fr.fromU64(bit)) : input[i].key[1];
                pols.key2[row] = (zlevel == 2) ? fr.add(fr.mul(input[i].key[2], two), fr.fromU64(bit)) : input[i].key[2];
                pols.key3[row] = (zlevel == 3) ? fr.add(fr.mul(input[i].key[3], two), fr.fromU64(bit)) : input[i].key[3];
            } else {
                pols.key0[row] = input[i].key[0];
                pols.key1[row] = input[i].key[1];
                pols.key2[row] = input[i].key[2];
                pols.key3[row] = input[i].key[3];
            }
            pols.level[row] = glevel;
            pols.keyInChunk[row] = fr.fromU64(chunkValue);

            const Goldilocks::Element keyInChunkShifted = fr.fromU64(chunkValue * CHUNK_FACTORS[clock]);
            pols.keyIn[row] = (clock == 0) ? keyInChunkShifted : fr.add(pols.keyIn[row - 1], keyInChunkShifted);

            pols.bit[row] = gbit;
            pols.carryLt[row] = fr.fromU64(carry + 2 * lt);
            carry = chunkValueClimbed > CHUNK_MASKS[clock] ? 1 : 0;

            // to compare with GL only use bits of CHUNK
            const uint64_t croppedChunkValueClimbed = chunkValueClimbed & CHUNK_MASKS[clock];
            lt = croppedChunkValueClimbed < GL_CHUNKS[clock] ? 1 : (croppedChunkValueClimbed == GL_CHUNKS[clock] ? lt : 0);

            const uint8_t keySelLevel = clock == CLIMB_KEY_LAST_CLOCK ? zlevel : 0xFFFF;
            pols.keySel0[row] = (keySelLevel == 0) ? fr.one() : fr.zero();
            pols.keySel1[row] = (keySelLevel == 1) ? fr.one() : fr.zero();
            pols.keySel2[row] = (keySelLevel == 2) ? fr.one() : fr.zero();
            pols.keySel3[row] = (keySelLevel == 3) ? fr.one() : fr.zero();
            pols.result[row] = (clock == CLIMB_KEY_RESULT_CLOCK) ? fr.one() : fr.zero();
        }
    }

    // filling the rest of trace to pass the constraints

    const uint64_t usedRows = input.size() * CLIMB_KEY_CLOCKS;
    uint64_t row = usedRows;
    while (row < N) {
        pols.keySel0[row+3] = fr.one();
        pols.carryLt[row+1] = two;
        pols.carryLt[row+2] = two;
        pols.carryLt[row+3] = two;
        row = row + 4;
    }

    zklog.info("ClimbKeyExecutor successfully processed " + to_string(input.size()) + " climbkey actions (" + to_string((double(input.size())*CLIMB_KEY_CLOCKS*100)/N) + "%)");
}

void ClimbKeyExecutor::execute (vector<ClimbKeyAction> &action)
{
    void * pAddress = malloc(CommitPols::pilSize());
    if (pAddress == NULL)
    {
        zklog.error("ClimbKeyExecutor::execute() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
        exitProcess();
        return;
    }
    CommitPols cmPols(pAddress, CommitPols::pilDegree());
    execute(action, cmPols.ClimbKey);
    free(pAddress);
}

bool ClimbKeyHelper::calculate(Goldilocks &fr, const Goldilocks::Element &current, bool bit, Goldilocks::Element &result)
{
    uint64_t currentU64 = fr.toU64(current);
    if (currentU64 > (LIMIT_BEFORE_CLIMB_UP - (uint64_t) bit)) {
        return false;
    }
    result = fr.fromU64(currentU64 * 2ULL + (uint64_t) bit);
    return true;
}
