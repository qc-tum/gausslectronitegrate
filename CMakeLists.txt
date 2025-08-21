
cmake_minimum_required(VERSION 3.15)

project(gausslectronitegrate LANGUAGES C)

include(CTest)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP)

set(GAUSSLECTRONITEGRATE_DIRS "src")
set(GAUSSLECTRONITEGRATE_SOURCES "src/gausslet_factors.c" "src/nuclear_integrals.c" "src/eri_integrals.c" "src/hdf5_util.c")
set(TEST_SOURCES "test/test_nuclear_integrals.c" "test/test_eri_integrals.c" "test/run_tests.c")

add_executable(            gausslectronitegrate_test ${GAUSSLECTRONITEGRATE_SOURCES} ${TEST_SOURCES})
target_include_directories(gausslectronitegrate_test PRIVATE ${GAUSSLECTRONITEGRATE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     gausslectronitegrate_test PRIVATE m ${HDF5_LIBRARIES})
if (OpenMP_C_FOUND)
    target_link_libraries( gausslectronitegrate_test PRIVATE OpenMP::OpenMP_C)
endif()

add_test(NAME gausslectronitegrate_test COMMAND gausslectronitegrate_test)
