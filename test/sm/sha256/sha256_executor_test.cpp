#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <functional>
#include "sha256_executor.hpp"
#include "sha256_executor_test.hpp"
#include "sha256_gate.hpp"
#include "timer.hpp"

void SHA256SMTest(Goldilocks &fr, Sha256Executor &executor)
{
	void *pAddress = malloc(CommitPols::pilSize());
	if (pAddress == NULL)
	{
		zklog.error("SHA256SMTest() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
		exitProcess();
	}
	CommitPols cmPols(pAddress, CommitPols::pilDegree());

	// How this test works:
	// 1. We generate numberOfSlots test vectors, each is randomByteCount random bytes.
	// 2. We hash these test vectors using the standard SHA256 reference implementation.
	// 3. We also pack them into numberOfSlots "slots" to feed the SHA256 executor and hash the slots too.
	// 4. We compare the hashes.

	const uint64_t numberOfSlots = ((SHA256GateConfig.polLength - 1) / SHA256GateConfig.slotSize);
	const uint64_t randomByteCount = 32;

	string *pHash = new string[numberOfSlots];
	string *pHashGate = new string[numberOfSlots];
	std::vector<std::vector<Goldilocks::Element>> pInput(numberOfSlots);

	cout << "Starting FE " << numberOfSlots << " slots test..." << endl;

	for (uint64_t slot = 0; slot < numberOfSlots; slot++)
	{
		// Generate the randomByteCount random bytes for the test vector.
		uint8_t randomTestVector[randomByteCount];
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, 255);
		for (size_t i = 0; i < randomByteCount; ++i)
		{
			// randomTestVector[i] = static_cast<uint8_t>(dis(gen));
			randomTestVector[i] = 0;
		}

		// Calculate the reference hash through a reference implementation of SHA256.
		std::string hashString = "";
		SHA256(randomTestVector, randomByteCount, hashString);
		pHash[slot] = hashString;

		// Create an input slot for the SHA256 executor of size SHA256GateConfig.sinRefNumber.
		std::vector<Goldilocks::Element> pInputSlot(SHA256GateConfig.sinRefNumber);

		// padded data = randomTestVector[randomByteCount] + 0x80 + paddedZeros*0x00 + randomByteCount[8]
		uint64_t paddedSizeInBitsMin = randomByteCount * 8 + 1 + 64;
		uint64_t paddedSizeInBits = ((paddedSizeInBitsMin / 512) + 1) * 512;
		uint64_t paddedSize = paddedSizeInBits / 8;
		uint64_t paddedZeros = (paddedSizeInBits - paddedSizeInBitsMin) / 8;
		uint8_t padding[64] = {0};
		u642bytes(randomByteCount * 8, &padding[56], true);
		uint64_t onePosition = 64 - 8 - paddedZeros - 1;
		padding[onePosition] = 0x80;
		uint8_t randomTestVectorPadded[paddedSize];
		for (uint64_t i = 0; i < randomByteCount; i++)
		{
			randomTestVectorPadded[i] = randomTestVector[i];
		}
		for (uint64_t i = 0; i < (paddedSize - randomByteCount); i++)
		{
			randomTestVectorPadded[randomByteCount + i] = padding[onePosition + i];
		}

		GateState S(SHA256GateConfig);
		SHA256Gate(S, randomTestVectorPadded);

		// Fill the executor input slot with the generated random byte test vector, bit by bit.
		// Note that we're filling it with randomByteCount.
		for (uint64_t byte = 0; byte < paddedSize; byte++)
		{
			for (uint64_t bit = 0; bit < 8; bit++)
			{
				// Fill the bits normally from the test vector, with each bit being replaced with an equivalent
				// bit initialized as a Goldilocks finite field element.
				uint8_t mask = 1 << bit;
				pInputSlot[(byte * 8) + bit] = ((randomTestVectorPadded[byte] & mask) != 0) ? fr.one() : fr.zero();
			}
		}

		// Push the exector input slot into the vector of numberOfSlots slots.
		pInput[slot] = pInputSlot;
	}

	// Run the executor.
	TimerStart(SHA256_SM_EXECUTOR_FE);
	executor.execute(pInput, cmPols.Sha256);
	TimerStopAndLog(SHA256_SM_EXECUTOR_FE);

	for (uint64_t slot = 0; slot < numberOfSlots; slot++)
	{
		uint8_t aux[256];

		// For each slot, we must extract the output of the executor bit by bit.
		// Each bit is represented by a polynomial. We must get these bits by specifying their position in
		// the executor's output buffer for this slot, which starts at the distance of soutRef0.
		// Each output bit is separated by a multiple of SHA256GateConfig.soutRefDistance.
		// Once we get the polynomial, we AND it with 1 to extract the bit.
		for (uint64_t i = 0; i < 256; i++)
		{
			uint64_t bitIndex = SHA256GateConfig.relRef2AbsRef(SHA256GateConfig.soutRef0 + i * SHA256GateConfig.soutRefDistance, slot);
			uint64_t pol = executor.getPol(cmPols.Sha256.output, bitIndex);
			aux[i] = ((pol & uint64_t(1)) == 0) ? 0 : 1;
		}

		// We have retrieved all the bits from the executor.
		// We now convert them into a byte array,
		// and then convert the byte array into a hex string for comparison.
		uint8_t aux2[32];
		for (uint64_t i = 0; i < 32; i++)
		{
			bits2byte(&aux[i * 8], aux2[i]);
		}
		string aux3;
		ba2string(aux3, aux2, 32);

		// Compare the reference hash to the executor's hash,
		// both represented as hex strings.
		if (aux3 == pHash[slot].substr(0, 64))
		{
			cout << "Pass: slot=" << slot << " Sout=" << aux3 << endl;
		}
		else
		{
			cerr << "Error: slot=" << slot << " Sout=" << aux3 << " does not match hash=" << pHash[slot] << " (" << pHashGate[slot] << ")" << endl;
		}
	}
	free(pAddress);
}

uint64_t SHA256SMExecutorTest(Goldilocks &fr, const Config &config)
{
	cout << "SHA256SMExecutorTest() starting" << endl;

	Sha256Executor executor(fr, config);
	SHA256SMTest(fr, executor);

	cout << "SHA256SMExecutorTest() done" << endl;
	return 0;
}
