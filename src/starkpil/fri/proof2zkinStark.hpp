#ifndef PROOF2ZKIN__STARK_HPP
#define PROOF2ZKIN__STARK_HPP

#include <nlohmann/json.hpp>
#include "friProof.hpp"

using ordered_json = nlohmann::ordered_json;

struct BatchPublics {
    uint32_t oldStateRootPos = 0;
    uint32_t oldBatchAccInputHashPos = 8;
    uint32_t previousL1InfoTreeRootPos = 16;
    uint32_t previousL1InfoTreeIndexPos = 24;
    uint32_t chainIdPos = 25;
    uint32_t forkIdPos = 26;
    uint32_t newStateRootPos = 27;
    uint32_t newBatchAccInputHashPos = 35;
    uint32_t currentL1InfoTreeRootPos = 43;
    uint32_t currentL1InfoTreeIndexPos = 51;
    uint32_t newLocalExitRootPos = 52;
    uint32_t newLastTimestampPos = 59;
};

struct BlobInnerPublics {
    uint32_t oldBlobStateRootPos = 0;
    uint32_t oldBlobAccInputHashPos = 8;
    uint32_t oldBlobNumPos = 16;
    uint32_t oldStateRootPos = 17;
    uint32_t forkIdPos = 25;
    uint32_t newBlobStateRootPos = 26;
    uint32_t newBlobAccInputHashPos = 34;
    uint32_t newBlobNumPos = 42;
    uint32_t finalAccBatchHashDataPos = 43;
    uint32_t localExitRootFromBlobPos = 51;
    uint32_t isInvalidPos = 59;
    uint32_t timestampLimitPos = 60;
    uint32_t lastL1InfoTreeRootPos = 61;
    uint32_t lastL1InfoTreeIndexPos = 69;
};

struct BlobOuterPublics {
    uint32_t oldStateRootPos = 0;
    uint32_t oldBlobStateRootPos = 8;
    uint32_t oldBlobAccInputHashPos = 16;
    uint32_t oldBlobNumPos = 24;
    uint32_t chainIdPos = 25;
    uint32_t forkIdPos = 26;
    uint32_t newStateRootPos = 27;
    uint32_t newBlobStateRootPos = 35;
    uint32_t newBlobAccInputHashPos = 43;
    uint32_t newBlobNumPos = 51;
    uint32_t newLocalExitRootPos = 52;
};


ordered_json proof2zkinStark(ordered_json &fproof);
ordered_json joinzkinBatchRecursive2(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey, uint64_t steps);
ordered_json joinzkinBlobOuterRecursive2(ordered_json &zkin1, ordered_json &zkin2, ordered_json &verKey, uint64_t steps);
ordered_json joinzkinBlobOuter(ordered_json &zkinBatch, ordered_json &zkinBlobInnerRecursive1, ordered_json &verKey, std::string chainId, uint64_t steps);

#endif