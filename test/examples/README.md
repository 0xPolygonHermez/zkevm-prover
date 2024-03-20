# Verify pil-stark ALL test

```bash
node ../pil-stark/src/main_verifier.js -v test/examples/all/all.verkey.json -s test/examples/all/all.starkinfo.json -o runtime/output/jProof.json -b runtime/output/publics.json
```

# Verify pil2-stark ALL test

```bash
node ../pil2-stark-js/src/main_verifier.js -v test/examples/pil2/all/all.verkey.json -s test/examples/pil2/all/all.starkinfo.json -o runtime/output/jProof.json -b runtime/output/publics.json
```

# Verify pil2-stark C18 test

```bash
node ../pil2-stark-js/src/main_verifier.js -v test/examples/pil2/compressor/all.c18.verkey.json -s test/examples/pil2/compressor/all.c18.starkinfo.json -o runtime/output/jProof.json -b runtime/output/publics.json
```