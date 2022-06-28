#include "goldilocks/goldilocks_base_field.hpp"
#include "commit_pols_fibonacci.hpp"
#include "pols_fibonacci.hpp"

namespace step2prev
{
	static void calculateExps(PolsFibonacci &pols)
	{
		Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(0 * sizeof(Goldilocks::Element));

		// First code


		// Iteration code

#pragma omp parallel for
		for(uint64_t i = 1; i < 1023; i++)
		{
			Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(0 * sizeof(Goldilocks::Element));

		}

		// Last code

	}
}
