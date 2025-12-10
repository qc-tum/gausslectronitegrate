#include <math.h>
#include "util.h"


char* test_lfloor_div()
{
	for (glong a = -13; a <= 13; ++a)
	{
		for (glong b = -20; b <= 20; ++b)
		{
			if (b == 0) {
				continue;
			}

			glong c = lfloor_div(a, b);

			glong c_ref = (glong)floor((double)a / (double)b);
			if (c != c_ref) {
				return "floor integer division does not agree with reference value";
			}
		}
	}

	return 0;
}
