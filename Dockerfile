FROM ubuntu:22.04 as build

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev libbenchmark-dev nasm nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev git libssl-dev cmake libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY ./src ./src
COPY ./test ./test
COPY ./tools ./tools
COPY ./config ./config
COPY Makefile .
RUN make -j

FROM ubuntu:22.04

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev libssl-dev libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY --from=build /usr/src/app/build/zkProver /usr/local/bin

COPY ./testvectors ./testvectors
COPY ./config ./config
COPY ./src/main_sm/fork_1/scripts/rom.json ./src/main_sm/fork_1/scripts/rom.json
COPY ./src/main_sm/fork_2/scripts/rom.json ./src/main_sm/fork_2/scripts/rom.json
COPY ./src/main_sm/fork_3/scripts/rom.json ./src/main_sm/fork_3/scripts/rom.json
COPY ./src/main_sm/fork_4/scripts/rom.json ./src/main_sm/fork_4/scripts/rom.json
COPY ./src/main_sm/fork_5/scripts/rom.json ./src/main_sm/fork_5/scripts/rom.json

ENTRYPOINT [ "zkProver" ]

