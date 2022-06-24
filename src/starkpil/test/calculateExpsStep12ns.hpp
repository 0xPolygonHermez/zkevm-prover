#include "goldilocks/goldilocks_base_field.hpp"
#include "commit_pols_fibonacci.hpp"
#include "pols2ns_fibonacci.hpp"

#include "precalculated_pols2ns_fibonacci_file.hpp"
namespace step12ns
{
	static void calculateExps(Pols2nsFibonacci &pols)
	{
		Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(4 * sizeof(Goldilocks::Element));

		// First code

		Goldilocks::mul(tmp[0], pols.cm.Fibonacci.l1[0], pols.cm.Fibonacci.l1[0]);
		Goldilocks::mul(tmp[1], pols.cm.Fibonacci.l2[0], pols.cm.Fibonacci.l2[0]);
		Goldilocks::add(tmp[2], tmp[0], tmp[1]);
		Goldilocks::sub(tmp[3], tmp[2], pols.exps[2048]);
		Goldilocks::mul(pols.q[0], PreCalculatedPols2ns::Zi(0), tmp[3]);
		Goldilocks::mul(tmp[0], pols.cm.Fibonacci.l1[1], pols.cm.Fibonacci.l1[1]);
		Goldilocks::mul(tmp[1], pols.cm.Fibonacci.l2[1], pols.cm.Fibonacci.l2[1]);
		Goldilocks::add(tmp[2], tmp[0], tmp[1]);
		Goldilocks::sub(tmp[3], tmp[2], pols.exps[2049]);
		Goldilocks::mul(pols.q[1], PreCalculatedPols2ns::Zi(1), tmp[3]);


		// Iteration code

#pragma omp parallel for
		for(uint64_t i = 2; i < 2047; i++)
		{
			Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(4 * sizeof(Goldilocks::Element));

			Goldilocks::mul(tmp[0], pols.cm.Fibonacci.l1[i], pols.cm.Fibonacci.l1[i]);
			Goldilocks::mul(tmp[1], pols.cm.Fibonacci.l2[i], pols.cm.Fibonacci.l2[i]);
			Goldilocks::add(tmp[2], tmp[0], tmp[1]);
			Goldilocks::sub(tmp[3], tmp[2], pols.exps[2048 + i]);
			Goldilocks::mul(pols.q[0 + i], PreCalculatedPols2ns::Zi(i), tmp[3]);
		}

		// Last code

		Goldilocks::mul(tmp[0], pols.cm.Fibonacci.l1[2047], pols.cm.Fibonacci.l1[2047]);
		Goldilocks::mul(tmp[1], pols.cm.Fibonacci.l2[2047], pols.cm.Fibonacci.l2[2047]);
		Goldilocks::add(tmp[2], tmp[0], tmp[1]);
		Goldilocks::sub(tmp[3], tmp[2], pols.exps[4095]);
		Goldilocks::mul(pols.q[2047], PreCalculatedPols2ns::Zi(2047), tmp[3]);
	}
}
