#ifndef ROM_HPP_fork_5
#define ROM_HPP_fork_5

#include <nlohmann/json.hpp>
#include <string>
#include "main_sm/fork_5/main/rom_line.hpp"
#include "main_sm/fork_5/main/rom_constants.hpp"
#include "config.hpp"

using json = nlohmann::json;
using namespace std;

namespace fork_5
{

class Rom
{
public:

    const Config &config;

    uint64_t size; // Size of the ROM program, i.e. number of ROM lines found in rom.json
    RomLine *line; // ROM program lines, parsed and stored in memory
    unordered_map<string, uint64_t> memoryMap; // Map of memory variables offsets
    unordered_map<string, uint64_t> labels; // ROM lines labels, i.e. names of the ROM lines

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
    uint64_t depthOffset;
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

    /* Constants */
    RomConstants constants;

    /* Constructor */
    Rom (const Config &config) :
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
            depthOffset(0),
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
            effectivePercentageRLPOffset(0)
            { };

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