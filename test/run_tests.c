#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_point_group_representation();
char* test_grid_permutation();
char* test_kinetic_integrals();
char* test_nuclear_integrals();
char* test_eri_integrals();
char* test_lfloor_div();


#define TEST_FUNCTION_ENTRY(fname) { .func = fname, .name = #fname }


int main()
{
	#ifdef _OPENMP
	printf("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	printf("OpenMP not available\n");
	#endif

	struct test tests[] = {
		TEST_FUNCTION_ENTRY(test_point_group_representation),
		TEST_FUNCTION_ENTRY(test_grid_permutation),
		TEST_FUNCTION_ENTRY(test_kinetic_integrals),
		TEST_FUNCTION_ENTRY(test_nuclear_integrals),
		TEST_FUNCTION_ENTRY(test_eri_integrals),
		TEST_FUNCTION_ENTRY(test_lfloor_div),
	};
	int num_tests = sizeof(tests) / sizeof(tests[0]);

	int num_pass = 0;
	for (int i = 0; i < num_tests; ++i)
	{
		printf(".");
		char* msg = tests[i].func();
		if (msg == 0) {
			num_pass++;
		}
		else {
			printf("\nTest '%s' failed: %s\n", tests[i].name, msg);
		}
	}
	printf("\nNumber of successful tests: %i / %i\n", num_pass, num_tests);

	if (num_pass < num_tests)
	{
		printf("At least one test failed!\n");
	}
	else
	{
		printf("All tests passed.\n");
	}

	return num_pass != num_tests;
}
