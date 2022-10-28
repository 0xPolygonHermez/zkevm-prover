const fs = require("fs");
const path = require("path");
const ethers = require("ethers");
const {
    MemDB, ZkEVMDB, processorUtils, smtUtils, getPoseidon,
} = require('@0xpolygonhermez/zkevm-commonjs');

// paths files
const pathInput = path.join(__dirname, "./input_gen_recursive.json");
const pathOutput = path.join(__dirname, "./aggregate-batches.json");

async function main(){
    // build poseidon
    const poseidon = await getPoseidon();
    const F = poseidon.F;

    // read generate input
    const generateData = require(pathInput);

    // mapping wallets
    const walletMap = {};

    for (let i = 0; i < generateData.genesis.length; i++) {
        const {
            address, pvtKey
        } = generateData.genesis[i];

        const newWallet = new ethers.Wallet(pvtKey);
        walletMap[address] = newWallet;
    }

    // create a zkEVMDB and build a batch
    const db = new MemDB(F);
    const zkEVMDB = await ZkEVMDB.newZkEVM(
        db,
        poseidon,
        [F.zero, F.zero, F.zero, F.zero], // empty smt
        typeof generateData.oldAccInputHash === 'undefined' ? [F.zero, F.zero, F.zero, F.zero] : smtUtils.stringToH4(generateData.oldAccInputHash),
        generateData.genesis,
        null,
        null,
        generateData.chainID
    );

    // Build batches
    let updatedAccounts = {};

    for (let i = 0; i < generateData.batches.length; i++){
        const genBatchData = generateData.batches[i];

        // start batch
        const batch = await zkEVMDB.buildBatch(
            genBatchData.timestamp,
            genBatchData.sequencerAddr,
            smtUtils.stringToH4(genBatchData.globalExitRoot)
        );

        for (let j = 0; j < genBatchData.txs.length; j++){
            const genTx = genBatchData.txs[j];

            // build tx
            const tx = {
                to: genTx.to,
                nonce: genTx.nonce,
                value: ethers.utils.parseUnits(genTx.value, 'wei'),
                gasLimit: genTx.gasLimit,
                gasPrice: ethers.utils.parseUnits(genTx.gasPrice, 'wei'),
                chainId: genTx.chainId,
                data: genTx.data || '0x',
            };

            const rawTxEthers = await walletMap[genTx.from].signTransaction(tx);
            const customRawTx = processorUtils.rawTxToCustomRawTx(rawTxEthers);

            // add tx to batch
            batch.addRawTx(customRawTx);
        }

        // build batch
        await batch.executeTxs();
        updatedAccounts = { ...updatedAccounts, ...batch.getUpdatedAccountsBatch()};
        // consolidate state
        await zkEVMDB.consolidate(batch);
        // get stark input for each batch
        const starkInput = await batch.getStarkInput();
        // write input executor for each batch
        fs.writeFileSync(path.join(__dirname, `./input_executor_${i}.json`), JSON.stringify(starkInput, null, 2));
    }

    // print new states
    const newLeafs = {};
    for (const item in updatedAccounts) {
        const address = item;
        const account = updatedAccounts[address];
        newLeafs[address] = {};

        newLeafs[address].balance = account.balance.toString();
        newLeafs[address].nonce = account.nonce.toString();

        const storage = await zkEVMDB.dumpStorage(address);
        const hashBytecode = await zkEVMDB.getHashBytecode(address);
        newLeafs[address].storage = storage;
        newLeafs[address].hashBytecode = hashBytecode;
    }
    generateData.expectedLeafs = newLeafs;

    // write new leafs
    fs.writeFileSync(pathInput, JSON.stringify(generateData, null, 2));

    // write aggregate batches
    const initialNumBatch = 1;
    const finalNumBatch = zkEVMDB.lastBatch;
    const aggregatorAddress = generateData.aggregatorAddress;

    const outVerifyRecursive = await zkEVMDB.verifyMultipleBatches(initialNumBatch, finalNumBatch, aggregatorAddress);
    fs.writeFileSync(pathOutput, JSON.stringify(outVerifyRecursive, null, 2));
}


main();