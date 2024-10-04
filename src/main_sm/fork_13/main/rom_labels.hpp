#ifndef ROM_LABELS_HPP_fork_13
#define ROM_LABELS_HPP_fork_13

#include <unistd.h>
#include <gmpxx.h>
#include "constants.hpp"

namespace fork_13
{

class RomLabels
{
public:
    uint64_t finalizeExecutionLabel;
    uint64_t checkAndSaveFromLabel;
    uint64_t ecrecoverStoreArgsLabel;
    uint64_t ecrecoverEndLabel;
    uint64_t checkFirstTxTypeLabel;
    uint64_t writeBlockInfoRootLabel;
    uint64_t outOfCountersStepLabel;
    uint64_t outOfCountersArithLabel;
    uint64_t outOfCountersBinaryLabel;
    uint64_t outOfCountersKeccakLabel;
    uint64_t outOfCountersSha256Label;
    uint64_t outOfCountersMemalignLabel;
    uint64_t outOfCountersPoseidonLabel;
    uint64_t outOfCountersPaddingLabel;
    uint64_t invalidIntrinsicTxSenderCodeLabel;

    RomLabels() :
        finalizeExecutionLabel(U64Mask64),
        checkAndSaveFromLabel(U64Mask64),
        ecrecoverStoreArgsLabel(U64Mask64),
        ecrecoverEndLabel(U64Mask64),
        checkFirstTxTypeLabel(U64Mask64),
        writeBlockInfoRootLabel(U64Mask64),
        outOfCountersStepLabel(U64Mask64),
        outOfCountersArithLabel(U64Mask64),
        outOfCountersBinaryLabel(U64Mask64),
        outOfCountersKeccakLabel(U64Mask64),
        outOfCountersSha256Label(U64Mask64),
        outOfCountersMemalignLabel(U64Mask64),
        outOfCountersPoseidonLabel(U64Mask64),
        outOfCountersPaddingLabel(U64Mask64),
        invalidIntrinsicTxSenderCodeLabel(U64Mask64)
        {};
};

}

#endif