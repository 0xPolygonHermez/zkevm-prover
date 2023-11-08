#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <functional>
#include "keccak_f_executor.hpp"
#include "keccak_executor_test.hpp"
#include "timer.hpp"

bool getBit(uint8_t byte, int position)
{
	// Create a bitmask with a 1 at the desired position
	uint8_t mask = 1 << position;

	// Perform bitwise AND and check if the result is zero or not
	return (byte & mask) != 0;
}

void KeccakSMTest(Goldilocks &fr, KeccakFExecutor &executor)
{
	void *pAddress = malloc(CommitPols::pilSize());
	if (pAddress == NULL)
	{
		zklog.error("KeccakSMTest() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
		exitProcess();
	}
	CommitPols cmPols(pAddress, CommitPols::pilDegree());

	// How this test works:
	// 1. We generate 54 test vectors, each is 135 random bytes.
	// 2. We hash these test vectors using the standard Keccak256 reference implementation.
	// 3. We also pack them into 54 "slots" to feed the Keccak executor and hash the slots too.
	// 4. We compare the hashes.

	const uint64_t numberOfSlots = ((KeccakGateConfig.polLength - 1) / KeccakGateConfig.slotSize);
	const uint64_t keccakBitratePlusCapacity = 1600;
	const uint64_t randomByteCount = 135;

	string *pHash = new string[numberOfSlots];
	std::vector<std::vector<Goldilocks::Element>> pInput(numberOfSlots);

	cout << "Starting FE " << numberOfSlots << " slots test..." << endl;

	for (uint64_t slot = 0; slot < numberOfSlots; slot++)
	{
		// Generate the 135 random bytes for the test vector.
		std::vector<uint8_t> randomTestVector(randomByteCount);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, 255);
		for (size_t i = 0; i < randomByteCount; ++i)
		{
			randomTestVector[i] = static_cast<uint8_t>(dis(gen));
		}

		// Calculate the reference hash through a reference implementation of Keccak.
		mpz_class hash = 0;
		keccak256(randomTestVector, hash);

		// Convert the output into a hex string, for comparison later.
		// We might need to add some missing leading zeroes
		// to the resulting hex string so that it's 64 characters.
		char *hashChars = new char[64];
		mpz_get_str(hashChars, 16, hash.get_mpz_t());
		std::string hashString(hashChars, 64);
		if (std::strlen(hashChars) < 64)
		{
			int missingLeadingZeroes = 64 - std::strlen(hashChars);
			hashString.insert(0, missingLeadingZeroes, '0');
		}
		pHash[slot] = hashString;

		// Create an input slot for the Keccak executor of size keccakBitratePlusCapacity.
		std::vector<Goldilocks::Element> pInputSlot(keccakBitratePlusCapacity);

		// Fill the executor input slot with the generated random byte test vector, bit by bit.
		// Note that we're filling it with (randomByteCount + 1), because the last byte is a
		// special "padding byte", which must be passed to the executor input but not to the
		// reference Keccak implementation used above.
		for (uint64_t byte = 0; byte < (randomByteCount + 1); byte++)
		{
			for (uint64_t bit = 0; bit < 8; bit++)
			{
				if (byte < randomByteCount)
				{
					// Fill the bits normally from the test vector, with each bit being replaced with an equivalent
					// bit initialized as a Goldilocks finite field element.
					pInputSlot[(byte * 8) + bit] = getBit(randomTestVector[byte], bit) ? fr.one() : fr.zero();
				}
				else if (byte == randomByteCount)
				{
					// Looks like we're at the final padding byte. This padding byte must have the form
					// 10000001. So we only set the first and last bit.
					pInputSlot[(byte * 8) + bit] = (bit == 0 || bit == 7) ? fr.one() : fr.zero();
				}
			}
		}
		// Push the exector input slot into the vector of 54 slots.
		pInput[slot] = pInputSlot;
	}

	// Run the executor.
	TimerStart(KECCAK_SM_EXECUTOR_FE);
	executor.execute(pInput, cmPols.KeccakF);
	TimerStopAndLog(KECCAK_SM_EXECUTOR_FE);

	for (uint64_t slot = 0; slot < numberOfSlots; slot++)
	{
		uint8_t aux[256];

		// For each slot, we must extract the output of the executor bit by bit.
		// Each bit is represented by a polynomial. We must get these bits by specifying their position in
		// the executor's output buffer for this slot, which starts at the distance of soutRef0.
		// Each output bit is separated by a multiple of 44.
		// Once we get the polynomial, we AND it with 1 to extract the bit.
		for (uint64_t i = 0; i < 256; i++)
		{
			uint64_t bitIndex = KeccakGateConfig.relRef2AbsRef(KeccakGateConfig.soutRef0 + i * 44, slot);
			uint64_t pol = executor.getPol(cmPols.KeccakF.a, bitIndex);
			aux[i] = ((pol & uint64_t(1)) == 0)? 0 : 1;
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
			cerr << "Error: slot=" << slot << " Sout=" << aux3 << " does not match hash=" << pHash[slot] << endl;
		}
	}
	free(pAddress);
}

uint64_t KeccakSMExecutorTest(Goldilocks &fr, const Config &config)
{
	cout << "KeccakSMExecutorTest() starting" << endl;

	KeccakFExecutor executor(fr, config);
	KeccakSMTest(fr, executor);

	cout << "KeccakSMExecutorTest() done" << endl;
	return 0;
}
