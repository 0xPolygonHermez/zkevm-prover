# zkEVM Prover
zkEVM proof generator
## General info
The zkEVM Prover process can provide up to 3 RPC services:

### Prover service
- It calls the Prover component that executes the input data (a batch of EVM transactions), calculates the resulting state, and generates the proof of the calculation based on the PIL polynomials definition and their constrains.
- When called by the Prover service, the Executor component combines 14 state machines that process the input data to generate the evaluations of the committed polynomials, required to generate the proof.  Every state machine generates their computation evidence data, and the more complex calculus demonstrations are delegated to the next state machine.
- The Prover component calls the Stark component to generate a proof of the Executor state machines committed polynomials.
- The interface of this service is defined by the file zk-prover.proto.

### Executor service
- It calls the Executor component that executes the input data (a batch of EVM transactions) and calculates the resulting state.  The proof is not generated.
- It provides a fast way to check if the proposed batch of transactions is properly built and it fits the amount of work that can be proven in one single batch.
- When called by the Executor service, the Executor component only uses the Main state machine, since the committed polynomials are not required as the proof will not be generated.
- The interface of this service is defined by the file executor.proto.

### StateDB service
- It provides an interface to access the state of the system (a Merkle tree) and the database where the state is stored.
- It is used by the executor and the prover, as the single source of state.  It can be used to get state details, e.g. account balances.
- The interface of this service is defined by the file statedb.proto.

## Setup

### Clone repository
```sh
$ git clone git@github.com:0xPolygonHermez/zkevm-prover.git
$ cd zkevm-prover
$ git submodule init
$ git submodule update --remote
```

### Compile
The following packages must be installed.
```sh
$ sudo apt install build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev postgresql libpqxx-dev libpqxx-doc nasm libsecp256k1-dev grpc-proto libsodium-dev libprotobuf-dev libssl-dev cmake libgrpc++-dev protobuf-compiler protobuf-compiler-grpc uuid-dev
```
The following files must be downloaded and added manually to the `testvectors` folder.
| File | Link |
| ---- | ---- |
| zkevm.const | https://de012a78750e59b808d922b39535e862.s3.eu-west-1.amazonaws.com/orange_release/zkevm.const |
| zkevm.consttree | https://de012a78750e59b808d922b39535e862.s3.eu-west-1.amazonaws.com/orange_release/zkevm.consttree |
| zkevm.g16.0001.zkey | https://de012a78750e59b808d922b39535e862.s3.eu-west-1.amazonaws.com/orange_release/zkevm.g16.0001.zkey |

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
To execute the Prover you need to provide a `config.json` file that contains the parameters that allow us to configure the different Prover options. By default, the Prover loads the `config.json`file located in the `testvectors`folder. The most relevant parameters are commented below with the default value for the provided `config.json` file:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| runProverServer | true | Enable Prover GRPC service |
| runExecutorServer | true | Enable Executor server |
| runStateDBServer | true | Enable StateDB (Merkle-tree) GRPC service |
| runFile | false | Execute the Prover using as input a test file defined in `"inputFile"` parameter |
| inputFile | input_executor.json | Test input file. It must be located in the `testvectors` folder |
| outputPath | output | Output path folder to store the result files. It must be located in the `testvectors` folder |
| databaseURL | postgresql://statedb:statedb@127.0.0.1:5432/testdb | Connection string for the PostgreSQL database used by the StateDB service. If the value is `"local"` then the service will not use a database and the data will be stored only in memory (no persistence). The PostgreSQL database connection string has the following format: `"postgresql://<user>:<password>@<ip>:<port>/<database>"`. For example: `"postgresql://statedb:statedb@127.0.0.1:5432/testdb"` |
| stateDBURL | local | Connection string for the StateDB service. If the value is `"local"` then the GRPC StateDB service will not be used and local StateDB client will be used instead. The StateDB service connection string has the following format: `"<ip>:<port>"`. For example: `"127.0.0.1:50061"` |

To run a proof test you must perform the following steps:
- Edit the `config.json` file and set the parameter `"runFile"` to `"true"`. The rest of the parameters must be set to `"false"`. Also set the parameter `"databaseURL` to `"local"` if you don't want to use a postgreSQL database to run the test
- Indicate in the `"inputFile"` parameter the file with the input test data. You can find a test file `input_executor.json` in the `testvectors` folder
- Run the Prover from the `testvectors` folder using the command `$ ../build/zkProver`
- The result files of the proof will be stored in the folder specified in the `"outputPath"` config parameter

## License

### Copyright
Polygon `zkevm-prover` was developed by Polygon. While we plan to adopt an open source license, we haven’t selected one yet, so all rights are reserved for the time being. Please reach out to us if you have thoughts on licensing.  
  
### Disclaimer
This code has not yet been audited, and should not be used in any production systems.

