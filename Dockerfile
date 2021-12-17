FROM debian:bullseye-slim 

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential libgmp-dev libbenchmark-dev nasm nlohmann-json3-dev libsecp256k1-dev libomp-dev

COPY . .

RUN make 

ENTRYPOINT [ "/usr/src/app/build/zkProver" ]


