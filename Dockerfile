FROM ubuntu:impish as build
 
WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev libbenchmark-dev nasm nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev git libssl-dev cmake libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY ./src ./src
COPY ./tools ./tools
COPY ./test ./test
COPY Makefile .
RUN make -j

FROM ubuntu:impish

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev libssl-dev libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

COPY --from=build /usr/src/app/build/zkProver /usr/local/bin

COPY ./testvectors/zkevm.starkinfo.json /usr/src/app/zkevm.starkinfo.json
COPY ./testvectors/binary.pil.json /usr/src/app/binary.pil.json
COPY ./testvectors/mem.pil.json /usr/src/app/mem.pil.json
COPY ./testvectors/storage.pil.json /usr/src/app/storage.pil.json
COPY ./testvectors/storage_sm_rom.json /usr/src/app/storage_sm_rom.json
COPY ./testvectors/keccak_connections.json /usr/src/app/keccak_connections.json
COPY ./testvectors/keccak_script.json /usr/src/app/keccak_script.json
COPY ./testvectors/proof.json /usr/src/app/proof.json
COPY ./testvectors/public.json /usr/src/app/public.json
COPY ./testvectors/zkevm.pil.json /usr/src/app/zkevm.pil.json

ENTRYPOINT [ "zkProver" ]

