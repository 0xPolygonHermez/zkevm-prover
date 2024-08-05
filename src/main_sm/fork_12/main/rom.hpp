#ifndef ROM_HPP_fork_12
#define ROM_HPP_fork_12

#include <nlohmann/json.hpp>
#include <string>
#include "main_sm/fork_12/main/rom_line.hpp"
#include "main_sm/fork_12/main/rom_constants.hpp"
#include "main_sm/fork_12/main/rom_labels.hpp"
#include "config.hpp"
#include "zkassert.hpp"

using json = nlohmann::json;
using namespace std;

namespace fork_12
{

enum RomType
{
    UNSPECIFIED = 0,
    BATCH = 1,
    BLOB = 2,
    DIAGNOSTIC = 3
};

class Rom
{
public:

    const Config &config;

    uint64_t size; // Size of the ROM program, i.e. number of ROM lines found in rom.json
    RomLine *line; // ROM program lines, parsed and stored in memory
    unordered_map<string, uint64_t> memoryMap; // Map of memory variables offsets
    unordered_map<string, uint64_t> labelsMap; // ROM lines labels, i.e. names of the ROM lines

    /* Offsets of memory variables */
    uint64_t memLengthOffset;
    uint64_t txDestAddrOffset;
    uint64_t txCalldataLenOffset;
    uint64_t txGasLimitOffset;
    uint64_t txValueOffset;
    uint64_t txNonceOffset;
    uint64_t txGasPriceOffset;
    uint64_t txGasPriceRLPOffset;
    uint64_t txChainIdOffset;
    uint64_t txROffset;
    uint64_t txSOffset;
    uint64_t txVOffset;
    uint64_t txSrcOriginAddrOffset;
    uint64_t retDataCTXOffset;
    uint64_t retDataOffsetOffset;
    uint64_t retDataLengthOffset;
    uint64_t newAccInputHashOffset;
    uint64_t oldNumBatchOffset;
    uint64_t newNumBatchOffset;
    uint64_t newLocalExitRootOffset;
    uint64_t gasRefundOffset;
    uint64_t txSrcAddrOffset;
    uint64_t gasCallOffset;
    uint64_t isPreEIP155Offset;
    uint64_t isCreateContractOffset;
    uint64_t storageAddrOffset;
    uint64_t bytecodeLengthOffset;
    uint64_t originCTXOffset;
    uint64_t currentCTXOffset;
    uint64_t gasCTXOffset;
    uint64_t lastCtxUsedOffset;
    uint64_t isCreateOffset;
    uint64_t effectivePercentageRLPOffset;
    uint64_t calldataCTXOffset;
    uint64_t calldataOffsetOffset;
    uint64_t blockNumOffset;
    uint64_t cumulativeGasUsedOffset;
    uint64_t isForcedOffset;
    uint64_t sequencerAddrOffset;
    uint64_t blockInfoSROffset;
    uint64_t timestampOffset;
    uint64_t gerL1InfoTreeOffset;
    uint64_t previousBlockHashOffset;
    uint64_t blockHashL1InfoTreeOffset;
    uint64_t isChangeL2BlockTxOffset;
    uint64_t txIndexOffset;
    uint64_t l2TxHashOffset;
    uint64_t currentTxOffset;
    uint64_t txStatusOffset;
    uint64_t currentLogIndexOffset;

    /* Constants */
    RomConstants constants;

    /* Labels */
    RomLabels labels;

    RomType type;

    /* Constructor */
    Rom (const Config &config, RomType type) :
            config(config),
            size(0),
            line(NULL),
            memLengthOffset(0),
            txDestAddrOffset(0),
            txCalldataLenOffset(0),
            txGasLimitOffset(0),
            txValueOffset(0),
            txNonceOffset(0),
            txGasPriceOffset(0),
            txGasPriceRLPOffset(0),
            txChainIdOffset(0),
            txROffset(0),
            txSOffset(0),
            txVOffset(0),
            txSrcOriginAddrOffset(0),
            retDataCTXOffset(0),
            retDataOffsetOffset(0),
            retDataLengthOffset(0),
            newAccInputHashOffset(0),
            oldNumBatchOffset(0),
            newNumBatchOffset(0),
            newLocalExitRootOffset(0),
            gasRefundOffset(0),
            txSrcAddrOffset(0),
            gasCallOffset(0),
            isPreEIP155Offset(0),
            isCreateContractOffset(0),
            storageAddrOffset(0),
            bytecodeLengthOffset(0),
            originCTXOffset(0),
            currentCTXOffset(0),
            gasCTXOffset(0),
            lastCtxUsedOffset(0),
            isCreateOffset(0),
            effectivePercentageRLPOffset(0),
            blockNumOffset(0),
            cumulativeGasUsedOffset(0),
            isForcedOffset(0),
            sequencerAddrOffset(0),
            blockInfoSROffset(0),
            timestampOffset(0),
            gerL1InfoTreeOffset(0),
            previousBlockHashOffset(0),
            blockHashL1InfoTreeOffset(0),
            isChangeL2BlockTxOffset(0),
            txIndexOffset(0),
            l2TxHashOffset(0),
            currentTxOffset(0),
            txStatusOffset(0),
            currentLogIndexOffset(0),
            type(type)
            {
                zkassertpermanent((type == BATCH) || (type == BLOB) || (type == DIAGNOSTIC));
            };

    /* Destructor */
    ~Rom() { if (line!=NULL) unload(); }

    /* Parses the ROM JSON data and stores them in memory, in ctx.rom[i] */
    void load(Goldilocks &fr, json &romJson);

    /* Frees any memory allocated in load() */
    void unload(void);
    
    uint64_t getLabel(const string &label) const;
    uint64_t getMemoryOffset(const string &label) const;
    uint64_t getConstant(json &romJson, const string &constantName);
    mpz_class getConstantL(json &romJson, const string &constantName);

private:
    void loadProgram(Goldilocks &fr, json &romJson);
    void loadLabels(Goldilocks &fr, json &romJson);
};

} // namespace

#endif