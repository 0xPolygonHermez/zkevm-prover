FROM ubuntu:22.04 as build

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev postgresql libpqxx-dev libpqxx-doc nasm libsecp256k1-dev grpc-proto libsodium-dev libprotobuf-dev libssl-dev cmake libgrpc++-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY ./src ./src
COPY ./test ./test
COPY ./tools ./tools
COPY Makefile .
RUN make -j

FROM ubuntu:22.04

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev postgresql libpqxx-dev libpqxx-doc nasm libsecp256k1-dev grpc-proto libsodium-dev libprotobuf-dev libssl-dev cmake libgrpc++-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY --from=build /usr/src/app/build/zkProver /usr/local/bin

ENTRYPOINT [ "zkProver" ]

