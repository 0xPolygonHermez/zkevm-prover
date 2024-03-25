# zkEVM Prover

Built to interface with Ethereum Virtual Machines (EVM), the prover provides critical services through three primary RPC clients: the Aggregator client, Executor service, and StateDB service. The Aggregator client connects to an Aggregator server and harnesses multiple zkEVM Provers simultaneously, thereby maximizing proof generation efficiency. This involves a process where the Prover component calculates a resulting state by processing EVM transaction batches and subsequently generates a proof based on the PIL polynomials definition and their constraints. The Executor service offers a mechanism to validate the integrity of proposed EVM transaction batches, ensuring they adhere to specific workload requirements. The StateDB service interfaces with a system's state (represented as a Merkle tree) and the corresponding database, thus serving as a centralized state information repository.

## Components

### Aggregator client

- It establishes a connection to an Aggregator server.
- Multiple zkEVM Provers can simultaneously connect to the Aggregator server, thereby enhancing the proof generation capability.
- Upon being invoked by the Aggregator service for batch proof generation:
  - The Prover component processes the input data (a set of EVM transactions), computes the resulting state, and creates a proof based on the PIL polynomial definitions and their constraints.
  - The Executor component integrates 14 state machines to process the input data and produce evaluations of the committed polynomials, essential for proof generation. Each state machine generates its computational evidence, and intricate calculations are passed on to the subsequent state machine.
- The Prover component then invokes the Stark component to produce a proof for the committed polynomials from the Executor's state machines.
- When tasked by the Aggregator service to produce an aggregated proof:
  - The Prover component amalgamates the results of two previously computed batch or aggregated proofs, supplied by the Aggregator, to create an aggregated proof.
- When tasked by the Aggregator service to produce a final proof:
  - The Prover component uses the outcome of a prior aggregated proof, supplied by the Aggregator, to formulate a conclusive proof that can be validated.
- The server interface for this service is delineated in the file named `aggregator.proto`.

### Executor service

- The Executor component processes the input data, which comprises a batch of EVM transactions, and computes the resulting state. Notably, no proof is produced.
- This service offers a swift method to verify whether a proposed batch of transactions is correctly constructed and if it aligns with the workload that can be proven in a single batch.
- When the Executor service invokes the Executor component, only the Main state machine is utilized. This is because the committed polynomials aren't needed, given that a proof isn't generated.
- The service's interface is outlined in the `executor.proto` file.

### StateDB service

- This service provides an interface to access the system's state (represented as a Merkle tree) and the database where this state is stored.
- Both the executor and the prover rely on it as the unified source of state. It can be utilized to retrieve specific state details, such as account balances.
- The interface for this service is described in the `statedb.proto` file.

## Compiling locally

Steps to compile `zkevm-prover` locally:
### Clone repository

```sh
git clone --recursive https://github.com/0xPolygonHermez/zkevm-prover.git
cd zkevm-prover
```

### Download necessary files

Download this **very large archive (~75GB)**. It's a good idea to start this download now and have it running in the background:

```sh
./tools/download_archive.sh
```

The archive will take up an additional 115GB of space once extracted.

### Install dependencies

The following packages must be installed.

**Important dependency note**: you must install [`libpqxx` version 6.4.5](https://github.com/jtv/libpqxx/releases/tag/6.4.5). If your distribution installs a newer version, please [compile `libpqxx` 6.4.5](https://github.com/jtv/libpqxx/releases/tag/6.4.5) and install it manually instead.

#### Ubuntu/Debian

```sh
apt update
apt install build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev postgresql libpqxx-dev libpqxx-doc nasm libsecp256k1-dev grpc-proto libsodium-dev libprotobuf-dev libssl-dev cmake libgrpc++-dev protobuf-compiler protobuf-compiler-grpc uuid-dev
```

#### openSUSE
```sh
zypper addrepo https://download.opensuse.org/repositories/network:cryptocurrencies/openSUSE_Tumbleweed/network:cryptocurrencies.repo
zypper refresh
zypper install -t pattern devel_basis
zypper install libbenchmark1 libomp16-devel libgmp10 nlohmann_json-devel postgresql libpqxx-devel ghc-postgresql-libpq-devel nasm libsecp256k1-devel grpc-devel libsodium-devel libprotobuf-c-devel libssl53 cmake libgrpc++1_57 protobuf-devel uuid-devel llvm llvm-devel libopenssl-devel
```

#### Fedora
```
dnf group install "C Development Tools and Libraries" "Development Tools"
dnf config-manager --add-repo https://terra.fyralabs.com/terra.repo
dnf install google-benchmark-devel libomp-devel gmp gmp-devel gmp-c++ nlohmann-json-devel postgresql libpqxx-devel nasm libsecp256k1-devel grpc-devel libsodium-devel cmake grpc grpc-devel grpc-cpp protobuf-devel protobuf-c-devel uuid-devel libuuid-devel uuid-c++ llvm llvm-devel openssl-devel 
```

#### Arch
```sh
pacman -S base-devel extra/protobuf community/grpc-cli community/nlohmann-json extra/libpqxx nasm extra/libsodium community/libsecp256k1
```

### Compilation

You may first need to recompile the protobufs:
```sh
cd src/grpc
make
cd ../..
```

Run `make` to compile the main project:

```sh
make clean
make -j
```

To compile in debug mode, run `make -j dbg=1`.

### Test vectors

```sh
./build/zkProver -c testvectors/config_runFile_BatchProof.json
```

## StateDB service database

To use persistence in the StateDB (Merkle-tree) service you must create the database objects needed by the service. To do this run the shell script:

```sh
./tools/statedb/create_db.sh <database> <user> <password>
```

For example:

```sh
./tools/statedb/create_db.sh testdb statedb statedb
```

## Docker

```sh
sudo docker build -t zkprover .
sudo docker run --rm --network host -ti -p 50051:50051 -p 50061:50061 -p 50071:50071 -v $PWD/testvectors:/usr/src/app zkprover input_executor.json
```

## Usage

To run the Prover, supply a `config.json` file containing the parameters that help customize various Prover settings. By default, the Prover accesses the `config.json` file from the `testvectors` directory. Below are some of the key parameters, accompanied by their default values from the given `config.json`:

| Parameter              | Description |
| ---------------------- | ----------- |
| `runStateDBServer`     | Enables StateDB GRPC service, provides SMT (Sparse Merkle Tree) and Database access |
| `runExecutorServer`    | Enables Executor GRPC service, provides a service to process transaction batches    |
| `runAggregatorClient`  | Enables Aggregator GRPC client, connects to the Aggregator and process its requests |
| `aggregatorClientHost` | IP address of the Aggregator server to which the Aggregator client must connect to  |
| `runProverServer`      | Enables Prover GRPC service                                                         |
| `runFileProcessBatch`  | Processes a batch using as input a JSON file defined in the `"inputFile"` parameter |
| `runFileGenProof`      | Generates a proof using as input a JSON file defined in the `"inputFile"` parameter |
| `inputFile`            | Input JSON file with path relative to the `testvectors` folder                      |
| `outputPath`           | Output path to store the result files, relative to the `testvectors` folder         |
| `saveRequestToFile`    | Saves service received requests to a text file                                      |
| `saveResponseToFile`   | Saves service returned responses to a text file                                     |
| `saveInputToFile`      | Saves service received input data to a JSON file                                    |
| `saveOutputToFile`     | Saves service returned output data to a JSON file                                   |
| `databaseURL`          | For the StateDB service, if the value is `"local"`, data is stored in memory; otherwise, use the PostgreSQL format: `"postgresql://<user>:<password>@<ip>:<port>/<database>"`, e.g., `"postgresql://statedb:statedb@127.0.0.1:5432/testdb"`. |
| `stateDBURL`           | For the StateDB service, if the value is "`local"`, a local client replaces the GRPC service. Use the format: `"<ip>:<port>", e.g., "127.0.0.1:50061"`. |

To execute a proof test:

1. Modify the `config.json` file, setting the `"runFileGenProof"` parameter to `"true"`. Ensure all other parameters are set to `"false"`. If you prefer not to use a PostgreSQL database for the test, adjust the `"databaseURL"` to `"local"`.
2. For the `"inputFile"` parameter, specify the desired input test data file. As an example, the `testvectors` directory contains the `input_executor.json` file.
3. Launch the Prover from the `testvectors` directory using the command: `../build/zkProver`.
4. The proof's result files will be saved in the directory defined by the `"outputPath"` configuration parameter.
