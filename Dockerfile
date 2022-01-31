FROM debian:bullseye-slim 

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev libbenchmark-dev nasm nlohmann-json3-dev libsecp256k1-dev libomp-dev libpqxx-dev git libssl-dev cmake libgrpc++-dev libprotobuf-dev grpc-proto libsodium-dev protobuf-compiler protobuf-compiler-grpc libuuid-dev

#COPY ./src ./src
#COPY Makefile .
#RUN make 

#ENTRYPOINT [ "/usr/src/app/build/zkProver" ]

