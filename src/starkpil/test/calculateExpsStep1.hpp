#include "goldilocks/goldilocks_base_field.hpp"
#include "commit_pols_fibonacci.hpp"
#include "pols_fibonacci.hpp"

namespace step1
{
	static void calculateExps(PolsFibonacci &pols)
	{
		Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(2 * sizeof(Goldilocks::Element));

		// First code

		Goldilocks::mul(tmp[0], pols.cm.Fibonacci.l1[0], pols.cm.Fibonacci.l1[0]);
		Goldilocks::mul(tmp[1], pols.cm.Fibonacci.l2[0], pols.cm.Fibonacci.l2[0]);
		Goldilocks::add(pols.exps[1024], tmp[0], tmp[1]);

		// Iteration code

#pragma omp parallel for
		for (uint64_t i = 1; i < 1023; i++)
		{
			Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(2 * sizeof(Goldilocks::Element));

			Goldilocks::mul(tmp[0], pols.cm.Fibonacci.l1[i], pols.cm.Fibonacci.l1[i]);
			Goldilocks::mul(tmp[1], pols.cm.Fibonacci.l2[i], pols.cm.Fibonacci.l2[i]);
			Goldilocks::add(pols.exps[1024 + i], tmp[0], tmp[1]);
		}

		// Last code

		Goldilocks::mul(tmp[0], pols.cm.Fibonacci.l1[1023], pols.cm.Fibonacci.l1[1023]);
		Goldilocks::mul(tmp[1], pols.cm.Fibonacci.l2[1023], pols.cm.Fibonacci.l2[1023]);
		Goldilocks::add(pols.exps[2047], tmp[0], tmp[1]);
	}
}
