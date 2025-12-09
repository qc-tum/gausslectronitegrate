#include "matrix.h"


//________________________________________________________________________________________________________________________
///
/// \brief Multiply two integer matrices.
///
void imat3x3_multiply(const struct imat3x3* restrict s, const struct imat3x3* restrict t, struct imat3x3* restrict ret)
{
	for (int i = 0; i < 3; ++i)
	{
		for (int k = 0; k < 3; ++k)
		{
			ret->data[i][k] = 0;
			for (int j = 0; j < 3; j++) {
				ret->data[i][k] += s->data[i][j] * t->data[j][k];
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two integer matrices are equal.
///
bool imat3x3_equal(const struct imat3x3* restrict s, const struct imat3x3* restrict t)
{
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (s->data[i][j] != t->data[i][j]) {
				return false;
			}
		}
	}

	return true;
}
