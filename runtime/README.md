
#### Config file
```
    "saveOutputToFile": true,
    "saveProofToFile": true,
```
### Verify recursive1 proof with pil-stark
```bash

verkey=$(tr -d '[:blank:]' < config/recursive2/recursive2.verkey.json | awk -F "," '{if(NR==3) print "[\""$1"\"," ; else if (NR == 6) print "\""$1"\"]"; else if(NR>=3 && NR<7) print "\""$1"\","}')

jq --argjson groupInfo "$(echo $verkey)" '. + $groupInfo' $(ls -t runtime/output/*.gen_batch_proof_public.json | head -n1) > $(ls -t runtime/output/*.gen_batch_proof_public.json | head -n1).tmp && mv $(ls -t runtime/output/*.gen_batch_proof_public.json | head -n1).tmp $(ls -t runtime/output/*.gen_batch_proof_public.json | head -n1)

node ../pil-stark/src/main_verifier.js -p config/recursive1/recursive1.pil -s config/recursive1/recursive1.starkinfo.json -o $(ls -t runtime/output/*.batch_proof.proof.json | head -n1) -b $(ls -t runtime//output/*.gen_batch_proof_public.json | head -n1) -v config/recursive1/recursive1.verkey.json
```
### Verify recursive2 proof with pil-stark
```bash
node ../pil-stark/src/main_verifier.js -p config/recursive2/recursive2.pil -s config/recursive2/recursive2.starkinfo.json -o $(ls -t runtime/output/*.aggregated_proof.proof.json | head -n1) -b $(ls -t runtime/output/*.gen_aggregated_proof_public.json | head -n1) -v config/recursive2/recursive2.verkey.json
```

#### Verify final snark
```bash
snarkjs ffv config/final/final.fflonk.verkey.json $(ls -t runtime//output/*.gen_final_proof_public.json | head -n1) $(ls -t runtime//output/*.final_proof.proof.json | head -n1)
```
