const fs = require("fs");
const path = require("path");
const ethers = require("ethers");
const {
    MemDB, ZkEVMDB, processorUtils, smtUtils, getPoseidon,
} = require('@0xpolygonhermez/zkevm-commonjs');

// paths files
const pathInput = path.join(__dirname, "./input_gen.json");
const pathOutput = path.join(__dirname, "./input_executor.json");

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

    // build tx
    const tx = {
        to: generateData.tx.to,
        nonce: generateData.tx.nonce,
        value: ethers.utils.parseUnits(generateData.tx.value, 'wei'),
        gasLimit: generateData.tx.gasLimit,
        gasPrice: ethers.utils.parseUnits(generateData.tx.gasPrice, 'wei'),
        chainId: generateData.tx.chainId,
        data: generateData.tx.data || '0x',
    };

    const rawTxEthers = await walletMap[generateData.tx.from].signTransaction(tx);
    const customRawTx = processorUtils.rawTxToCustomRawTx(rawTxEthers);

    // create a zkEVMDB and build a batch
    const db = new MemDB(F);
    const zkEVMDB = await ZkEVMDB.newZkEVM(
        db,
        poseidon,
        [F.zero, F.zero, F.zero, F.zero], // empty smt
        smtUtils.stringToH4(generateData.oldAccInputHash),
        generateData.genesis,
        null,
        null,
        generateData.chainID
    );

    // start batch
    const batch = await zkEVMDB.buildBatch(
        generateData.timestamp,
        generateData.sequencerAddr,
        smtUtils.stringToH4(generateData.globalExitRoot)
    );

    // add tx to batch
    batch.addRawTx(customRawTx);

    // build batch
    await batch.executeTxs();
    // consolidate state
    await zkEVMDB.consolidate(batch);

    // get stark input
    const starkInput = await batch.getStarkInput();

    // print new states
    const updatedAccounts = batch.getUpdatedAccountsBatch();
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

    fs.writeFileSync(pathOutput, JSON.stringify(starkInput, null, 2));
    fs.writeFileSync(pathInput, JSON.stringify(generateData, null, 2));
}

main();