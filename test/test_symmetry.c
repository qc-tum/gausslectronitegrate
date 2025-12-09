#include "symmetry.h"
#include "util.h"
#include "aligned_memory.h"


char* test_point_group_representation()
{
	for (int i = 0; i < 48; ++i)
	{
		for (int j = 0; j < 48; ++j)
		{
			struct imat3x3 prod;
			imat3x3_multiply(&octahedral_matrep[i], &octahedral_matrep[j], &prod);

			int k;
			for (k = 0; k < 48; ++k)
			{
				if (imat3x3_equal(&prod, &octahedral_matrep[k])) {
					break;
				}
			}

			if (k == 48) {
				return "product of two octahedral point group representation matrices does not result in another representation matrix";
			}

			if (k != octahedral_multiplication_table[i][j]) {
				return "product of two octahedral point group representation matrices does not agree with matrix index from multiplication table";
			}
		}
	}

	return 0;
}


char* test_grid_permutation()
{
	const struct cartesian_grid_3d grid = {
		.coord_range = {
			{ .istart = -2, .num = 5 },
			{ .istart = -2, .num = 5 },
			{ .istart = -2, .num = 5 },
		}
	};

	const glong num_points = cartesian_grid_3d_num_points(&grid);

	glong* perm = aligned_malloc(num_points * sizeof(glong));

	const int ir = 17;
	compute_cartesian_grid_3d_permutation(&grid, &octahedral_matrep[ir], perm);

	if (!is_permutation(perm, num_points)) {
		return "grid permutation map is not a valid permutation";
	}

	// find the inverse transformation from the multiplication table
	int ir_inv;
	for (ir_inv = 0; ir_inv < 48; ++ir_inv)
	{
		if (octahedral_multiplication_table[ir_inv][ir] == 0) {
			break;
		}
	}
	if (ir_inv == 48) {
		return "cannot find inverse transformation matrix in multiplication table of the octahedral point group";
	}

	glong* perm_inv = aligned_malloc(num_points * sizeof(glong));
	compute_cartesian_grid_3d_permutation(&grid, &octahedral_matrep[ir_inv], perm_inv);
	if (!is_permutation(perm_inv, num_points)) {
		return "grid permutation map is not a valid permutation";
	}

	glong* perm_composed = aligned_malloc(num_points * sizeof(glong));
	compose_permutations(num_points, perm_inv, perm, perm_composed);
	if (!is_permutation(perm_composed, num_points)) {
		return "composed map is not a valid permutation";
	}

	// expecting identity permutation
	if (!is_identity_permutation(perm_composed, num_points)) {
		return "composition of grid permutations effected by a transformation and its inverse should result in the identity permutation";
	}

	aligned_free(perm_composed);
	aligned_free(perm_inv);
	aligned_free(perm);

	return 0;
}
