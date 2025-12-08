#pragma once


//________________________________________________________________________________________________________________________
///
/// \brief Linear range of integers.
///
struct range
{
	long istart;  //!< starting integer
	long num;     //!< number of entries (logical length)
};


//________________________________________________________________________________________________________________________
///
/// \brief Integer range containing all pairwise differences between elements of 'r' and 's'.
///
static inline struct range range_minus(const struct range r, const struct range s)
{
	struct range d = {
		.istart = r.istart - (s.istart + s.num - 1),  // "smallest - largest"
		.num    = r.num + s.num - 1
	};
	return d;
}
