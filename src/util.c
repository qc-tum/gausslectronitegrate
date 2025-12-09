#include "util.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Test whether an integer map is a permutation of the list [0, ..., n - 1].
///
bool is_permutation(const glong* map, const glong n)
{
	for (glong i = 0; i < n; i++) {
		if (map[i] < 0 || map[i] >= n) {
			return false;
		}
	}

	bool* indicator = aligned_calloc(n * sizeof(bool));
	for (glong i = 0; i < n; i++) {
		indicator[map[i]] = true;
	}
	for (glong i = 0; i < n; i++) {
		if (!indicator[i])
		{
			aligned_free(indicator);
			return false;
		}
	}
	aligned_free(indicator);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether a permutation is the identity permutation.
///
bool is_identity_permutation(const glong* perm, const glong n)
{
	for (glong i = 0; i < n; i++) {
		if (perm[i] != i) {
			return false;
		}
	}
	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compose two permutations: ret = p ∘ q, i.e., ret(i) = p(q(i)).
///
void compose_permutations(const glong n, const glong* restrict p, const glong* restrict q, glong* restrict ret)
{
	for (glong i = 0; i < n; i++) {
		ret[i] = p[q[i]];
	}
}
