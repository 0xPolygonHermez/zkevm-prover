#ifndef VARIABLES_C_HPP_fork_6
#define VARIABLES_C_HPP_fork_6

namespace fork_6
{

class GlobalVariablesC
{
public:
    // Input variables
    mpz_class oldStateRoot; // Previous state-tree root
    mpz_class oldAccInputHash; // Previous accumulated input hash
    mpz_class globalExitRoot; // Global exit-tree root
    mpz_class oldNumBatch; // Previous batch processed
    mpz_class sequencerAddr; // Coinbase address which will receive the fees
    mpz_class batchHashData; // batchHashData = H_keccak( transactions )
    mpz_class timestamp; // Current batch timestamp
    mpz_class chainID; // Current batch chain id
    mpz_class forkID; // Fork identifier

    // Output variables
    mpz_class newAccInputHash; // Final accumulated input hash. newAccInputHash = H_keccak( oldAccInputHash | batchHashData | globalExitRoot | timestamp | sequencerAddr )
    mpz_class newLocalExitRoot; // Updated local exit tree root
    mpz_class newNumBatch; // Current batch processed

    mpz_class batchL2DataParsed; // Number of bytes read when decoding RLP transactions. Computed during RLP loop
    mpz_class pendingTxs; // Number of transactions decoded in RLP block
    mpz_class lastCtxUsed; // Last context that has been used
    mpz_class ctxTxToUse; // First context to be used when processing transactions
    mpz_class lastHashKIdUsed; // Last hash address used
    mpz_class nextHashPId; // Next poseidon hash address available

    mpz_class batchL2DataLength; // Transactions bytes read from the input
    mpz_class batchHashDataId; // hash address used when adding bytes to batchHashData
    mpz_class batchHashPos; // hash batchHashData position
    mpz_class currentCTX; // keeps track of the context used
    mpz_class originAuxCTX; // keeps track of the previous context when a new one is created
    mpz_class gasCalldata; // gas spent by the calldata

    mpz_class gasCall; // total gas forwarded when creating a new context
    mpz_class addrCall; // address parameter when creating a new context
    mpz_class valueCall; // value parameter when creating a new context
    mpz_class argsLengthCall; // size of the calldata creating a new context
    mpz_class txSrcOriginAddr; // origin address of a tx
    mpz_class txGasPrice; // transaction parameter: 'gasPrice' global var
    mpz_class depth; // Current depth execution
    mpz_class cntKeccakPreProcess; // Number of keccak counters needed to finish the batch

    mpz_class nextFreeLogIndex; // pointer to the next free slot to store information in LOG opcode
    mpz_class originSR; // State root before processing each transaction
    mpz_class batchSR; // State root before processing any transaction
    mpz_class txCount; // Current transaction count

    mpz_class touchedSR; // touched tree root
    mpz_class numTopics; // number of topics depending on LOG opcode call
    mpz_class SPw; // aux variable to store Stack pointer 'SP'
    mpz_class auxSR; // auxiliary variable. Temporary state root
    mpz_class txRLPLength; // transaction RLP list length
    mpz_class txDataRead; // aux variable to check transaction 'data' left that needs to be read
    mpz_class isLoadingRLP; // flag to determine if the function is called from RLP loop
};


class ContextVariablesC
{
public:
    mpz_class txGasLimit; // transaction parameter: 'gas limit'
    mpz_class txDestAddr; // transaction parameter: 'to'
    mpz_class storageAddr; // address which the storage will be modified
    mpz_class txValue; // transaction parameter: 'value'
    mpz_class txNonce; // transaction parameter: nonce
    mpz_class txGasPriceRLP; // transaction parameter: 'gasPrice' decoded from the RLP
    mpz_class effectivePercentageRLP; // transaction parameter: 'effectivePercentage' decoded from the RLP
    mpz_class txChainId; // transaction parameter: 'chainId'
    mpz_class txS; // transaction parameter: ecdsa signature S
    mpz_class txR; // transaction parameter: ecdsa signature R
    mpz_class txV; // transaction parameter: ecdsa signature V
    mpz_class txSrcAddr; // address that sends a transaction 'message.sender'
    mpz_class txHash; // signed tx hash
    mpz_class txCalldataLen; // calldata length
    mpz_class isCreateContract; // flag to determine if a transaction will create a new contract
    mpz_class createContractAddress; // address computed of a new contract
    mpz_class lengthNonce; // 'nonce' length used when computing a new contract address
    mpz_class gasRefund; // keeps track of the transaction gas refund
    mpz_class initSR; // state-tree once the initial upfront cost is substracted and nonce is increased
    mpz_class memLength; // current memory size
    mpz_class lastMemLength; // length of bytes to copy to memory
    mpz_class lastMemoryExpansionCost; // cost of the last memory expansion
    mpz_class lastMemOffset; // offset to copy to memory
    mpz_class retCallOffset; // initial pointer to begin store the return data
    mpz_class retCallLength; // size of the return data
    mpz_class retDataOffset; // pointer to previous context return data offset
    mpz_class retDataLength; // pointer to previous context return data length
    mpz_class retDataCTX; // pointer to context where the return data is stored
    mpz_class argsOffsetCall; // pointer to the init slot where the calldata begins
    mpz_class bytecodeLength; // state-tree length bytecode leaf value of the 'to' address
    mpz_class contractHashId; // hashP address used to store contract bytecode
    mpz_class originCTX; // The source context of the current context
    mpz_class lastSP; // Last stack pointer used of the previous context
    mpz_class lastPC; // Last program counter used of the previous context
    mpz_class isStaticCall; // flag to determine if a new context comes from a STATICCALL opcode
    mpz_class isCreate; // flag to determine if a new context comes from a CREATE opcode
    mpz_class isDelegateCall; // flag to determine if a new context comes from a DELEGATECALL opcode
    mpz_class isCreate2; // flag to determine if a new context comes from a CREATE2 opcode
    mpz_class salt; // CREATE2 parameter 'salt' used to compute new contract address
    mpz_class gasCTX; // remaining gas in the origin CTX when a new context is created
    mpz_class dataStarts; // hash position where de transaction 'data' starts in the batchHashData
    mpz_class isPreEIP155; // flag to check if the current tx is legacy, previous to Spurious Dragon (EIP-155)
    mpz_class initTouchedSR; // touched root once a new context begins
};

}
#endif