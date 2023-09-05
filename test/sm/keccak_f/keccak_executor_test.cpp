#include <vector>
#include "keccak_f_executor.hpp"
#include "keccak_executor_test.hpp"
#include "timer.hpp"

void KeccakSMTest(Goldilocks &fr, const Config &config, KeccakFExecutor &executor)
{
	void *pAddress = malloc(CommitPols::pilSize());
	if (pAddress == NULL)
	{
		zklog.error("KeccakSMTest() failed calling malloc() of size=" + to_string(CommitPols::pilSize()));
		exitProcess();
	}
	CommitPols cmPols(pAddress, CommitPols::pilDegree());

	const uint64_t numberOfSlots = ((KeccakGateConfig.polLength - 1) / KeccakGateConfig.slotSize);
	const uint64_t keccakBitratePlusCapacity = 1600;
	const uint64_t randomByteCount = 136;
	std::vector<std::vector<Goldilocks::Element>> pInput;
	string *pHash = new string[numberOfSlots];

	cout << "Starting FE " << numberOfSlots << " slots test..." << endl;

	for (uint64_t slot = 0; slot < numberOfSlots; slot++)
	{
		std::vector<Goldilocks::Element> pInputSlot;
		std::vector<uint8_t> toHash;
		for (uint64_t i = 0; i < keccakBitratePlusCapacity; i++)
		{
			if (i < (randomByteCount - 1) * 8)
			{
				pInputSlot.push_back(fr.one());
			}
			else if (i == (randomByteCount - 1) * 8)
			{
				pInputSlot.push_back(fr.one());
			}
			else if (i == (randomByteCount - 1) * 8 + 7)
			{
				pInputSlot.push_back(fr.one());
			}
			else
			{
				pInputSlot.push_back(fr.zero());
			}
		}
		pInput.push_back(pInputSlot);
		for (uint64_t i = 0; i < (randomByteCount - 1); i++)
		{
			toHash.push_back(0xFF);
		}
		mpz_class hash = 0;
		keccak256(toHash, hash);
		char *hashChars = new char[64];
		mpz_get_str(hashChars, 16, hash.get_mpz_t());
		std::string hashString(hashChars, 64);
		pHash[slot] = hashString;
	}

	TimerStart(KECCAK_SM_EXECUTOR_FE);
	executor.execute(pInput, cmPols.KeccakF);
	TimerStopAndLog(KECCAK_SM_EXECUTOR_FE);

	for (uint64_t slot = 0; slot < numberOfSlots; slot++)
	{
		uint8_t aux[256];
		for (uint64_t i = 0; i < 256; i++)
		{
			if ((executor.getPol(cmPols.KeccakF.a, KeccakGateConfig.relRef2AbsRef(KeccakGateConfig.soutRef0 + i * 44, slot))) == 0)
			{
				aux[i] = 0;
			}
			else
			{
				aux[i] = 1;
			}
		}
		uint8_t aux2[32];
		for (uint64_t i = 0; i < 32; i++)
		{
			bits2byte(&aux[i * 8], aux2[i]);
		}
		string aux3;
		ba2string(aux3, aux2, 32);
		if (aux3 != pHash[slot])
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
	KeccakSMTest(fr, config, executor);

	cout << "KeccakSMExecutorTest() done" << endl;
	return 0;
}
