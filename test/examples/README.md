## Constraint checker 

# Constraint checker ALL test

```bash
./build/constraintChecker -c test/examples/all/all.const -s test/examples/all/all.starkinfo.json -h test/examples/all/all.chelpers/all.chelpers_generic.bin -t test/examples/all/all.commit -p test/examples/all/all.publics.json
```

# Constraint checker ALL C18 test

```bash
./build/constraintChecker -c test/examples/compressor/all.c18.const -s test/examples/compressor/all.c18.starkinfo.json -h test/examples/compressor/all.c18.chelpers/all.c18.chelpers_generic.bin -t test/examples/compressor/all.c18.commit -p test/examples/compressor/all.c18.publics.json
```

# Constraint checker Fibonacci Pil2 test

```bash
./build/constraintChecker -c test/examples/fibonacci.pil2/fibonacci.pil2.const -s test/examples/fibonacci.pil2/fibonacci.pil2.starkinfo.json -b test/examples/fibonacci.pil2/fibonacci.pil2.chelpers/fibonacci.pil2.chelpers_generic.bin -t test/examples/fibonacci.pil2/fibonacci.pil2.commit -p test/examples/fibonacci.pil2/fibonacci.pil2.publics.json
```

## Verify

# Verify ALL test

```bash
node ../pil2-stark-js/src/main_verifier.js -v test/examples/all/all.verkey.json -s test/examples/all/all.starkinfo.json -i test/examples/all/all.verifierinfo.json -o runtime/output/all_proof.json -b test/examples/all/all.publics.json
```

# Verify ALL C18 test

```bash
node ../pil2-stark-js/src/main_verifier.js -v test/examples/compressor/all.c18.verkey.json -s test/examples/compressor/all.c18.starkinfo.json -i test/examples/compressor/all.c18.verifierinfo.json -o runtime/output/compressor_proof.json -b test/examples/compressor/all.c18.publics.json
```

# Verify Fibonaci Pil2 test

```bash
node ../pil2-stark-js/src/main_verifier.js -v test/examples/fibonacci.pil2/fibonacci.pil2.verkey.json -s test/examples/fibonacci.pil2/fibonacci.pil2.starkinfo.json -i test/examples/fibonacci.pil2/fibonacci.pil2.verifierinfo.json -o runtime/output/fibonacci_pil2_proof.json -b test/examples/fibonacci.pil2/fibonacci.pil2.publics.json
```
