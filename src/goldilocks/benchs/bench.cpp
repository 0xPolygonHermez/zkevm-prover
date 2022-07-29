#include <benchmark/benchmark.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "omp.h"

#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

#define NUM_HASHES 10000

#define FFT_SIZE (1 << 23)
#define NUM_COLUMNS 100
#define BLOWUP_FACTOR 1
#define NPHASES_NTT 2
#define NPHASES_LDE 2
#define NBLOCKS 1

static void POSEIDON_BENCH_FULL(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)SPONGE_WIDTH;

    Goldilocks::Element fibonacci[input_size];
    Goldilocks::Element result[input_size];

    // Test vector: Fibonacci series
    // 0 1 1 2 3 5 8 13 ... NUM_HASHES * SPONGE_WIDTH ...
    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();
    for (uint64_t i = 2; i < input_size; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }
    // Benchmark
    for (auto _ : state)
    {
        // Every thread process chunks of SPONGE_WIDTH elements
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            PoseidonGoldilocks::hash_full_result((Goldilocks::Element(&)[SPONGE_WIDTH])result[i * SPONGE_WIDTH], (Goldilocks::Element(&)[SPONGE_WIDTH])fibonacci[i * SPONGE_WIDTH]);
        }
    }
    // Check poseidon results poseidon ( 0 1 1 2 3 5 8 13 21 34 55 89 )
    assert(Goldilocks::toU64(result[0]) == 0X3095570037F4605D);
    assert(Goldilocks::toU64(result[1]) == 0X3D561B5EF1BC8B58);
    assert(Goldilocks::toU64(result[2]) == 0X8129DB5EC75C3226);
    assert(Goldilocks::toU64(result[3]) == 0X8EC2B67AFB6B87ED);

    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void POSEIDON_BENCH(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)SPONGE_WIDTH;
    uint64_t result_size = (uint64_t)NUM_HASHES * (uint64_t)CAPACITY;

    Goldilocks::Element fibonacci[input_size];
    Goldilocks::Element result[result_size];

    // Test vector: Fibonacci series
    // 0 1 1 2 3 5 8 13 ... NUM_HASHES * SPONGE_WIDTH ...
    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();
    for (uint64_t i = 2; i < input_size; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }
    // Benchmark
    for (auto _ : state)
    {
        // Every thread process chunks of SPONGE_WIDTH elements
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            PoseidonGoldilocks::hash((Goldilocks::Element(&)[CAPACITY])result[i * CAPACITY], (Goldilocks::Element(&)[SPONGE_WIDTH])fibonacci[i * SPONGE_WIDTH]);
        }
    }
    // Check poseidon results poseidon ( 0 1 1 2 3 5 8 13 21 34 55 89 )
    assert(Goldilocks::toU64(result[0]) == 0X3095570037F4605D);
    assert(Goldilocks::toU64(result[1]) == 0X3D561B5EF1BC8B58);
    assert(Goldilocks::toU64(result[2]) == 0X8129DB5EC75C3226);
    assert(Goldilocks::toU64(result[3]) == 0X8EC2B67AFB6B87ED);

    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void NTT_BENCH(benchmark::State &state)
{
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));

    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));

#pragma omp parallel for
    for (uint64_t k = 0; k < NUM_COLUMNS; k++)
    {
        uint64_t offset = k * FFT_SIZE;
        a[offset] = Goldilocks::one();
        a[offset + 1] = Goldilocks::one();
        for (uint64_t i = 2; i < FFT_SIZE; i++)
        {
            a[i] = a[i - 1] + a[i - 2];
        }
    }
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0))
        for (u_int64_t i = 0; i < NUM_COLUMNS; i++)
        {
            u_int64_t offset = i * FFT_SIZE;
            gntt.NTT(a + offset, a + offset, FFT_SIZE);
        }
    }
    free(a);
}

static void NTT_Block_BENCH(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }
    for (auto _ : state)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, NPHASES_NTT, NBLOCKS);
    }
    free(a);
}

static void LDE_BENCH(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

    a[0] = Goldilocks::one();
    a[1] = Goldilocks::one();
    for (uint64_t i = 2; i < (uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS; i++)
    {
        a[i] = a[i - 1] + a[i - 2];
    }

    Goldilocks::Element shift = Goldilocks::fromU64(49); // TODO: ask for this number, where to put it how to calculate it
    gntt.INTT(a, a, FFT_SIZE);

    // TODO: This can be pre-generated
    Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();
    for (int i = 1; i < FFT_SIZE; i++)
    {
        r[i] = r[i - 1] * shift;
    }

    Goldilocks::Element *zero_array = (Goldilocks::Element *)malloc((uint64_t)((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(Goldilocks::Element));
#pragma omp parallel for num_threads(state.range(0))
    for (uint i = 0; i < ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE); i++)
    {
        zero_array[i] = Goldilocks::zero();
    }

    for (auto _ : state)
    {
        Goldilocks::Element *res = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));

#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t k = 0; k < NUM_COLUMNS; k++)
        {
            for (int i = 0; i < FFT_SIZE; i++)
            {
                a[k * FFT_SIZE + i] = a[k * FFT_SIZE + i] * r[i];
            }
            std::memcpy(res, &a[k * FFT_SIZE], FFT_SIZE);
            std::memcpy(&res[FFT_SIZE], zero_array, (FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE);

            gntt_extension.NTT(res, res, (FFT_SIZE << BLOWUP_FACTOR));
        }
        free(res);
    }
    free(zero_array);
    free(a);
    free(r);
}

static void LDE_BENCH_Block(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    Goldilocks::Element shift = Goldilocks::fromU64(49); // TODO: ask for this number, where to put it how to calculate it

    gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NPHASES_NTT);

    // TODO: This can be pre-generated
    Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();
    for (int i = 1; i < FFT_SIZE; i++)
    {
        r[i] = r[i - 1] * shift;
    }

#pragma omp parallel for
    for (uint64_t i = 0; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * i + j] * r[i];
        }
    }
#pragma omp parallel for schedule(static)
    for (uint64_t i = (uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS; i < (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * (uint64_t)NUM_COLUMNS; i++)
    {
        a[i] = Goldilocks::zero();
    }
    for (auto _ : state)
    {
        gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR), NUM_COLUMNS, NPHASES_LDE, NBLOCKS);
    }
    free(a);
    free(r);
}

BENCHMARK(POSEIDON_BENCH_FULL)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(1, 1, 1)
    ->RangeMultiplier(2)
    ->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->UseRealTime();

BENCHMARK(POSEIDON_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(1, 1, 1)
    ->RangeMultiplier(2)
    ->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->UseRealTime();

BENCHMARK(NTT_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();

BENCHMARK(NTT_Block_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();

BENCHMARK(LDE_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();
BENCHMARK(LDE_BENCH_Block)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();

BENCHMARK_MAIN();
// Build command: g++ benchs/bench.cpp src/* -lbenchmark -lomp -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -L$(find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//') -O3 -o bench && ./bench
