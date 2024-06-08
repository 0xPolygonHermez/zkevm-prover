# Verify pil-stark ALL test

```bash
node ../pil-stark/src/main_verifier.js -v test/examples/all/all.verkey.json -s test/examples/all/all.starkinfo.json -o runtime/output/jProof.json -b runtime/output/publics.json
```