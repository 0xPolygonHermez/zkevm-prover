
### Verify recursive1 proof with pil-stark
#### Config file
```
    "saveOutputToFile": true,
    "saveProofToFile": true,
```
```bash
cd pil-stark/
ZKEVM_PROVER_DIR=~/zkevm-prover-stable
node src/main_verifier.js -p ${ZKEVM_PROVER_DIR}/config/recursive1/recursive1.pil -s ${ZKEVM_PROVER_DIR}/config/recursive1/recursive1.starkstruct.json -o ${ZKEVM_PROVER_DIR}/runtime/recursive1.proof.json -b $(ls -t ${ZKEVM_PROVER_DIR}/runtime//output/*public* | head -n1) -v ${ZKEVM_PROVER_DIR}/config/recursive1/recursive1.verkey.json
```
### Verify recursive2 proof with pil-stark
```bash
cd pil-stark/
ZKEVM_PROVER_DIR=~/zkevm-prover-stable
node src/main_verifier.js -p ${ZKEVM_PROVER_DIR}/config/recursive2/recursive2.pil -s ${ZKEVM_PROVER_DIR}/config/recursive2/recursive2.starkstruct.json -o $(ls -t ${ZKEVM_PROVER_DIR}/runtime//output/*.aggregated_proof.proof.json | head -n1) -b $(ls -t ${ZKEVM_PROVER_DIR}/runtime//output/*.aggregated_proof.public.json | head -n1) -v ${ZKEVM_PROVER_DIR}/config/recursive2/recursive2.verkey.json
```

#### Verify final snark
```bash
snarkjs g16v ${ZKEVM_PROVER_DIR}/config/final/final.g16.verkey.json $(ls -t ${ZKEVM_PROVER_DIR}/runtime//output/*.final_proof.public.json | head -n1) $(ls -t ${ZKEVM_PROVER_DIR}/runtime//output/*.final_proof.proof.json | head -n1)
```
