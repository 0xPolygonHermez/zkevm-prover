FROM ubuntu:22.04 as build

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev libbenchmark-dev nasm nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev git libssl-dev cmake libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY ./src ./src
COPY ./test ./test
COPY ./tools ./tools
COPY ./config ./config
COPY Makefile .
RUN make generate
RUN make -j

FROM ubuntu:22.04

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev libssl-dev libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY --from=build /usr/src/app/build/zkProver /usr/local/bin

COPY ./testvectors ./testvectors
COPY ./config ./config
COPY ./setup ./setup
COPY ./src/main_sm/fork_1/scripts/rom.json ./src/main_sm/fork_1/scripts/rom.json
COPY ./src/main_sm/fork_2/scripts/rom.json ./src/main_sm/fork_2/scripts/rom.json
COPY ./src/main_sm/fork_3/scripts/rom.json ./src/main_sm/fork_3/scripts/rom.json
COPY ./src/main_sm/fork_4/scripts/rom.json ./src/main_sm/fork_4/scripts/rom.json
COPY ./src/main_sm/fork_5/scripts/rom.json ./src/main_sm/fork_5/scripts/rom.json
COPY ./src/main_sm/fork_6/scripts/rom.json ./src/main_sm/fork_6/scripts/rom.json
COPY ./src/main_sm/fork_7/scripts/rom.json ./src/main_sm/fork_7/scripts/rom.json
COPY ./src/main_sm/fork_8/scripts/rom.json ./src/main_sm/fork_8/scripts/rom.json
COPY ./src/main_sm/fork_9/scripts/rom.json ./src/main_sm/fork_9/scripts/rom.json
COPY ./src/main_sm/fork_10/scripts/rom_10.json ./src/main_sm/fork_10/scripts/rom_10.json
COPY ./src/main_sm/fork_10/scripts/rom_11.json ./src/main_sm/fork_10/scripts/rom_11.json
COPY ./src/main_sm/fork_12/scripts/rom_12.json ./src/main_sm/fork_12/scripts/rom_12.json
COPY ./src/main_sm/fork_13/scripts/rom_13.json ./src/main_sm/fork_13/scripts/rom_13.json

ENTRYPOINT [ "zkProver" ]

