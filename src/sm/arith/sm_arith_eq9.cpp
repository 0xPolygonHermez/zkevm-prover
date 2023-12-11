/* code generated with arith_eq_gen.js
* equation: x1-x2-x3+p*q1-p*offset
* 
* p=0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
* offset=0x8
*/

#include <stdint.h>
#include "definitions.hpp"
#include "sm/pols_generated/commit_pols.hpp"
#include "goldilocks_base_field.hpp"

USING_PROVER_FORK_NAMESPACE;

int64_t eq9 (Goldilocks &fr, ArithCommitPols &p, uint64_t step, uint64_t _o)
{
	switch(step) {
		case 0: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x7ea38  + (int64_t)fr.toU64(p.x1[0][_o]) - (int64_t)fr.toU64(p.x2[0][_o]) - (int64_t)fr.toU64(p.x3[0][_o]));

		case 1: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x6c3e0  + (int64_t)fr.toU64(p.x1[1][_o]) - (int64_t)fr.toU64(p.x2[1][_o]) - (int64_t)fr.toU64(p.x3[1][_o]));

		case 2: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x460b0  + (int64_t)fr.toU64(p.x1[2][_o]) - (int64_t)fr.toU64(p.x2[2][_o]) - (int64_t)fr.toU64(p.x3[2][_o]));

		case 3: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x1e100  + (int64_t)fr.toU64(p.x1[3][_o]) - (int64_t)fr.toU64(p.x2[3][_o]) - (int64_t)fr.toU64(p.x3[3][_o]));

		case 4: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x65468  + (int64_t)fr.toU64(p.x1[4][_o]) - (int64_t)fr.toU64(p.x2[4][_o]) - (int64_t)fr.toU64(p.x3[4][_o]));

		case 5: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x34388  + (int64_t)fr.toU64(p.x1[5][_o]) - (int64_t)fr.toU64(p.x2[5][_o]) - (int64_t)fr.toU64(p.x3[5][_o]));

		case 6: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x35488  + (int64_t)fr.toU64(p.x1[6][_o]) - (int64_t)fr.toU64(p.x2[6][_o]) - (int64_t)fr.toU64(p.x3[6][_o]));

		case 7: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x4bc08  + (int64_t)fr.toU64(p.x1[7][_o]) - (int64_t)fr.toU64(p.x2[7][_o]) - (int64_t)fr.toU64(p.x3[7][_o]));

		case 8: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x2c2e8  + (int64_t)fr.toU64(p.x1[8][_o]) - (int64_t)fr.toU64(p.x2[8][_o]) - (int64_t)fr.toU64(p.x3[8][_o]));

		case 9: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x40c08  + (int64_t)fr.toU64(p.x1[9][_o]) - (int64_t)fr.toU64(p.x2[9][_o]) - (int64_t)fr.toU64(p.x3[9][_o]));

		case 10: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x22db0  + (int64_t)fr.toU64(p.x1[10][_o]) - (int64_t)fr.toU64(p.x2[10][_o]) - (int64_t)fr.toU64(p.x3[10][_o]));

		case 11: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x5c280  + (int64_t)fr.toU64(p.x1[11][_o]) - (int64_t)fr.toU64(p.x2[11][_o]) - (int64_t)fr.toU64(p.x3[11][_o]));

		case 12: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x50148  + (int64_t)fr.toU64(p.x1[12][_o]) - (int64_t)fr.toU64(p.x2[12][_o]) - (int64_t)fr.toU64(p.x3[12][_o]));

		case 13: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x70988  + (int64_t)fr.toU64(p.x1[13][_o]) - (int64_t)fr.toU64(p.x2[13][_o]) - (int64_t)fr.toU64(p.x3[13][_o]));

		case 14: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x27390  + (int64_t)fr.toU64(p.x1[14][_o]) - (int64_t)fr.toU64(p.x2[14][_o]) - (int64_t)fr.toU64(p.x3[14][_o]));

		case 15: 
		return (
		(0xfd47   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xd87c   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[1][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[0][_o]))
		    - 0x18320  + (int64_t)fr.toU64(p.x1[15][_o]) - (int64_t)fr.toU64(p.x2[15][_o]) - (int64_t)fr.toU64(p.x3[15][_o]));

		case 16: 
		return (
		(0xd87c   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x8c16   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[2][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[1][_o])));

		case 17: 
		return (
		(0x8c16   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x3c20   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[3][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[2][_o])));

		case 18: 
		return (
		(0x3c20   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xca8d   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[4][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[3][_o])));

		case 19: 
		return (
		(0xca8d   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x6871   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[5][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[4][_o])));

		case 20: 
		return (
		(0x6871   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x6a91   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[6][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[5][_o])));

		case 21: 
		return (
		(0x6a91   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x9781   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[7][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[6][_o])));

		case 22: 
		return (
		(0x9781   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x585d   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[8][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[7][_o])));

		case 23: 
		return (
		(0x585d   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x8181   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[9][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[8][_o])));

		case 24: 
		return (
		(0x8181   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x45b6   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[10][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[9][_o])));

		case 25: 
		return (
		(0x45b6   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xb850   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[11][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[10][_o])));

		case 26: 
		return (
		(0xb850   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xa029   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[12][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[11][_o])));

		case 27: 
		return (
		(0xa029   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0xe131   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[13][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[12][_o])));

		case 28: 
		return (
		(0xe131   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x4e72   * (int64_t)fr.toU64(p.q1[14][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[13][_o])));

		case 29: 
		return (
		(0x4e72   * (int64_t)fr.toU64(p.q1[15][_o])) +
		(0x3064   * (int64_t)fr.toU64(p.q1[14][_o])));

		case 30: 
		return ( (0x3064   * (int64_t)fr.toU64(p.q1[15][_o])));

		case 31: 
		return (
		0x0     );
	}
	return 0;
}
