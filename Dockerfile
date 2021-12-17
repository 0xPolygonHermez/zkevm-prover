FROM debian:bullseye-slim 

WORKDIR /usr/src/app
COPY . .

RUN apt update && apt install -y build-essential clang libgmp-dev libbenchmark-dev nasm nlohmann-json3-dev libsecp256k1-dev

RUN make 



