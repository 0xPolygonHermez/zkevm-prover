FROM ubuntu:impish as build
 
WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev libbenchmark-dev nasm nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev git libssl-dev cmake libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY ./src ./src
COPY Makefile .
RUN make -j

FROM ubuntu:impish

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev libssl-dev libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY --from=build /usr/src/app/build/zkProver /usr/local/bin

ENTRYPOINT [ "zkProver" ]

