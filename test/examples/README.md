# Verify pil-stark ALL test

```bash
node ../pil-stark/src/main_verifier.js -v test/examples/all/all.verkey.json -s test/examples/all/all.starkinfo.json -o runtime/output/jProof.json -b runtime/output/publics.json
```

# Constraint checker pil2-stark ALL test

```bash
./build/constraintChecker -c test/examples/pil2/all/all.const -t test/examples/pil2/all/all.consttree -s test/examples/pil2/all/all.starkinfo.json -v test/examples/pil2/all/all.verkey.json -h test/examples/pil2/all/all.chelpers/all.chelpers_generic.bin -w test/examples/pil2/all/all.commit -p test/examples/pil2/all/all.publics.json
```

# Verify pil2-stark ALL test

```bash
node ../pil2-stark-js/src/main_verifier.js -v test/examples/pil2/all/all.verkey.json -s test/examples/pil2/all/all.starkinfo.json -o runtime/output/jProof.json -b runtime/output/publics.json
```

# Constraint checker pil2-stark C18 test

```bash
./build/constraintChecker -c test/examples/pil2/compressor/all.c18.const -t test/examples/pil2/compressor/all.c18.consttree -s test/examples/pil2/compressor/all.c18.starkinfo.json -v test/examples/pil2/compressor/all.c18.verkey.json -h test/examples/pil2/compressor/all.c18.chelpers/all.c18.chelpers_generic.bin -w test/examples/pil2/compressor/all.c18.commit -p test/examples/pil2/compressor/all.c18.publics.json
```

# Verify pil2-stark C18 test

```bash
node ../pil2-stark-js/src/main_verifier.js -v test/examples/pil2/compressor/all.c18.verkey.json -s test/examples/pil2/compressor/all.c18.starkinfo.json -o runtime/output/jProof.json -b runtime/output/publics.json
```