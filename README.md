# Prover
zkEVM proof generator
## General info
The zkEVM Prover process can provide up to 3 RPC services:

### Prover service
- It calls the Prover component that executes the input data (a batch of EVM transactions), calculates the resulting state, and generates the proof of the calculation
- When called by the Prover service, the Executor component combines 14 state machines that process the input data to generate the evaluations of the committed polynomials, required to generate the proof.  Every state machine generates their computation evidence data, and the more complex calculus demonstrations are delegated to the next state machine.
- The interface of this service is defined by the file zk-prover.proto.

### Executor service
- It calls the Executor component that executes the input data (a batch of EVM transactions) and calculates the resulting state.  The proof is not generated.
- It provides a fast way to check if the proposed batch of transactions is properly built and it fits the amount of work that can be proven in one single batch.
- When called by the Executor service, the Executor component only uses the Main state machine, since the committed polynomials are not required as the proof will not be generated.
- The interface of this service is defined by the file executor.proto.

### StateDB service
- It provides an interface to access the state of the system (a Merkle tree) and the database where the state is stored
- It is used by the executor and the prover, as the single source of state.  It can be used to get state details, e.g. account balances.
- The interface of this service is defined by the file statedb.proto.

## Setup

### Compile
The following packages must be installed.
```sh
$ sudo apt install build-essential
$ sudo apt install libbenchmark-dev
$ sudo apt install libomp-dev
$ sudo apt install libgmp-dev
$ sudo apt install nlohmann-json3-dev
$ sudo apt install postgresql
$ sudo apt install libpqxx-dev libpqxx-doc
$ sudo apt install nasm
$ sudo apt install libsecp256k1-dev
$ sudo apt install grpc-proto
$ sudo apt install libsodium-dev
$ sudo apt install libprotobuf-dev
$ sudo apt install libssl-dev
$ sudo apt install cmake
$ sudo apt install libgrpc++-dev
$ sudo apt install protobuf-compiler
$ sudo apt install protobuf-compiler-grpc
$ sudo apt install uuid-dev
```
The following files must be added manually.  Please check size and md5 checksum.
```sh
$ ll testvectors/constantstree.bin
-rw-rw-r-- 1 fractasy fractasy 268715104 ene 11 16:41 testvectors/constantstree.bin
$ ll testvectors/verifier.dat
-rw-rw-r-- 1 fractasy fractasy 297485608 ene 12 10:54 testvectors/verifier.dat
$ ll testvectors/starkverifier_0001.zkey
-rw-r--r-- 1 fractasy fractasy 16816778703 ene 12 18:23 testvectors/starkverifier_0001.zkey

$ md5sum testvectors/constantstree.bin
02dc0dfe47a7aaacca6a34486ad5f314  testvectors/constantstree.bin
$ md5sum testvectors/verifier.dat
771a7a09f419f5e6f28dd0cc5a94c621  testvectors/verifier.dat
$ md5sum testvectors/starkverifier_0001.zkey
e460d81646a3a0ce81a561bbbb871363  testvectors/starkverifier_0001.zkey
```
Run `make` to compile the project.
```sh
$ make clean
$ make -j
```

### StateDB service database
To use persistence in the StateDB (Merkle-tree) service you must create the database objects needed by the service. To do this run the shell script: 
```sh
$ ./tools/statedb/create_db.sh <database> <user> <password>
```
For example:
```sh
$ ./tools/statedb/create_db.sh testdb statedb statedb
```

### Build & run docker
```sh
$ sudo docker build -t zkprover .
$ sudo docker run --rm --network host -ti -p 50051:50051 -p 50061:50061 -p 50071:50071 -v $PWD/testvectors:/usr/src/app zkprover input_executor.json
```

## Usage
The `config.json` file contains the parameters that allow us to configure the different Prover options. The most relevant parameters are commented below.

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| runProverServer | true | Enable Prover GRPC service |
| runExecutorServer | true | Enable Executor server |
| runStateDBServer | true | Enable StateDB (Merkle-tree) GRPC service |
| runFile | false | Execute the Prover using as input a test file defined in `"inputFile"` parameter |
| inputFile | input_executor.json | Test input file. It must be located in the `testvectors` folder |
| outputPath | output | Output path folder to store the result files. It must be located in the `testvectors` folder |
| databaseURL | local | Connection string for the PostgreSQL database used by the StateDB service. If the value is `"local"` then the service will not use a database and the data will be stored only in memory (no persistence). The PostgreSQL database connection string has the following format: `"postgresql://<user>:<password>@<ip>:<port>/<database>"`. For example: `"postgresql://statedb:statedb@127.0.0.1:5432/testdb"` |
| stateDBURL | local | Connection string for the StateDB service. If the value is `"local"` then the GRPC StateDB service will not be used and local StateDB client will be used instead. The StateDB service connection string has the following format: `"<ip>:<port>"`. For example: `"127.0.0.1:50061"` |

To run a proof test you must perform the following steps:
- Edit the config.json file and set the parameter `"runFile"` to `"true"`. The rest of the parameters must be `"false"`
- Indicate in the `"inputFile"` parameter the file with the input test data. You can find a test file `input_executor.json` in the `testvectors` folder
- Run the Prover from the `testvectors` folder using the command `$ ../build/zkProver`
- The result files of the proof will be stored in the folder specified in the `"outputPath"` config parameter

## License

### Copyright
Polygon zkevm-proverjs was developed by Polygon. While we plan to adopt an open source license, we haven’t selected one yet, so all rights are reserved for the time being. Please reach out to us if you have thoughts on licensing.  
  
### Disclaimer
This code has not yet been audited, and should not be used in any production systems.
