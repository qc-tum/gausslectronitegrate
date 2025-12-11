#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>


/// \brief Universal "long" integer.
typedef int64_t glong;

#define GLONG_MAX INT64_MAX


//________________________________________________________________________________________________________________________
///
/// \brief Square function x -> x^2.
///
static inline double square(const double x)
{
	return x * x;
}


//________________________________________________________________________________________________________________________
///
/// \brief Cubic power x -> x^3.
///
static inline double cubic_power(const double x)
{
	return x * x * x;
}


//________________________________________________________________________________________________________________________
///
/// \brief Minimum of two long integers.
///
static inline glong lmin(const glong a, const glong b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two long integers.
///
static inline glong lmax(const glong a, const glong b)
{
	return (a >= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Norm (Euclidean length) of a vector in three dimensions.
///
static inline double vec3_norm(const double v[3])
{
	const double vx = fabs(v[0]);
	const double vy = fabs(v[1]);
	const double vz = fabs(v[2]);

	if (vy <= vz)
	{
		if (vx <= vz)
		{
			if (vz == 0) {
				return 0;
			}

			assert(0 < vz);
			const double d0 = vx / vz;
			const double d1 = vy / vz;

			return vz * sqrt(1 + d0*d0 + d1*d1);
		}
		else  // vz < vx
		{
			assert(0 < vx);
			const double d0 = vy / vx;
			const double d1 = vz / vx;

			return vx * sqrt(1 + d0*d0 + d1*d1);
		}
	}
	else  // vz < vy
	{
		if (vx <= vy)
		{
			assert(0 < vy);
			const double d0 = vx / vy;
			const double d1 = vz / vy;

			return vy * sqrt(1 + d0*d0 + d1*d1);
		}
		else  // vy < vx
		{
			assert(0 < vx);
			const double d0 = vy / vx;
			const double d1 = vz / vx;

			return vx * sqrt(1 + d0*d0 + d1*d1);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Uniform distance (infinity norm) between 'x' and 'y'.
///
static inline double uniform_distance(const glong n, const double* x, const double* y)
{
	double d = 0;
	for (glong i = 0; i < n; ++i)
	{
		d = fmax(d, fabs(x[i] - y[i]));
	}
	return d;
}


//________________________________________________________________________________________________________________________
//


bool is_permutation(const glong* map, const glong n);

bool is_identity_permutation(const glong* perm, const glong n);


void compose_permutations(const glong n, const glong* p, const glong* q, glong* ret);
