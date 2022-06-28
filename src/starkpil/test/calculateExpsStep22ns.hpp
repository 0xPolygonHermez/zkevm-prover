#include "goldilocks/goldilocks_base_field.hpp"
#include "commit_pols_fibonacci.hpp"
#include "pols2ns_fibonacci.hpp"

#include "precalculated_pols2ns_fibonacci_file.hpp"
namespace step22ns
{
	static void calculateExps(Pols2nsFibonacci &pols)
	{
		Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(0 * sizeof(Goldilocks::Element));

		// First code


		// Iteration code

#pragma omp parallel for
		for(uint64_t i = 2; i < 2047; i++)
		{
			Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(0 * sizeof(Goldilocks::Element));

		}

		// Last code

	}
}
