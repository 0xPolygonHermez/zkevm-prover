/* code generated with arith_eq_gen.js
* equation: s*x1-s*x3-y1-y3+p*q2-p*offset
* 
* p=0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
* offset=0x20000000000000000000000000000000000000000000000000000000000000000
*/

#include <stdint.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"

USING_PROVER_FORK_NAMESPACE;

int64_t eq4 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o)
{
	switch(step) {
		case 0: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[0][_o]) - (int64_t)fr.toU64(p.y3[0][_o]));

		case 1: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[1][_o]) - (int64_t)fr.toU64(p.y3[1][_o]));

		case 2: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[2][_o]) - (int64_t)fr.toU64(p.y3[2][_o]));

		case 3: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[3][_o]) - (int64_t)fr.toU64(p.y3[3][_o]));

		case 4: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[4][_o]) - (int64_t)fr.toU64(p.y3[4][_o]));

		case 5: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[5][_o]) - (int64_t)fr.toU64(p.y3[5][_o]));

		case 6: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[6][_o]) - (int64_t)fr.toU64(p.y3[6][_o]));

		case 7: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[7][_o]) - (int64_t)fr.toU64(p.y3[7][_o]));

		case 8: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[8][_o]) - (int64_t)fr.toU64(p.y3[8][_o]));

		case 9: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[9][_o]) - (int64_t)fr.toU64(p.y3[9][_o]));

		case 10: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[10][_o]) - (int64_t)fr.toU64(p.y3[10][_o]));

		case 11: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[11][_o]) - (int64_t)fr.toU64(p.y3[11][_o]));

		case 12: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[12][_o]) - (int64_t)fr.toU64(p.y3[12][_o]));

		case 13: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[13][_o]) - (int64_t)fr.toU64(p.y3[13][_o]));

		case 14: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[14][_o]) - (int64_t)fr.toU64(p.y3[14][_o]));

		case 15: 
		return (
		((int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[0][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xfc2f   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[0][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[0][_o]))
		 - (int64_t)fr.toU64(p.y1[15][_o]) - (int64_t)fr.toU64(p.y3[15][_o]));

		case 16: 
		return (
		((int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[1][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[1][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[1][_o]))
		    - 0x1f85e );

		case 17: 
		return (
		((int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[2][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xfffe   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[2][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[2][_o]))
		    - 0x1fffe );

		case 18: 
		return (
		((int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[3][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[3][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[3][_o]))
		    - 0x1fffc );

		case 19: 
		return (
		((int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[4][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[4][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[4][_o]))
		    - 0x1fffe );

		case 20: 
		return (
		((int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[5][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[5][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[5][_o]))
		    - 0x1fffe );

		case 21: 
		return (
		((int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[6][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[6][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[6][_o]))
		    - 0x1fffe );

		case 22: 
		return (
		((int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[7][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[7][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[7][_o]))
		    - 0x1fffe );

		case 23: 
		return (
		((int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[8][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[8][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[8][_o]))
		    - 0x1fffe );

		case 24: 
		return (
		((int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[9][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[9][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[9][_o]))
		    - 0x1fffe );

		case 25: 
		return (
		((int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[10][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[10][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[10][_o]))
		    - 0x1fffe );

		case 26: 
		return (
		((int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[11][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[11][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[11][_o]))
		    - 0x1fffe );

		case 27: 
		return (
		((int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[12][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[12][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[12][_o]))
		    - 0x1fffe );

		case 28: 
		return (
		((int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[13][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[13][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[13][_o]))
		    - 0x1fffe );

		case 29: 
		return (
		((int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[14][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o])) +
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[14][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[14][_o]))
		    - 0x1fffe );

		case 30: 
		return (
		((int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.s[15][_o]) * (int64_t)fr.toU64(p.x3[15][_o]) + 0xffff   * (int64_t)fr.toU64(p.q2[15][_o]))
		    - 0x1fffe );

		case 31: 
		return ( - 0x1fffe);
	}
	return 0;
}
