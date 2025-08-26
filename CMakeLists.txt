
cmake_minimum_required(VERSION 3.15)

project(gausslectronitegrate LANGUAGES C)

include(CTest)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(OpenMP)

set(CMAKE_C_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")

set(GAUSSLECTRONITEGRATE_DIRS "src")
set(GAUSSLECTRONITEGRATE_SOURCES "src/gausslet_factors.c" "src/nuclear_integrals.c" "src/eri_integrals.c" "src/hdf5_util.c" "src/timing.c")
set(TEST_SOURCES "test/test_nuclear_integrals.c" "test/test_eri_integrals.c" "test/run_tests.c")

add_executable(            gausslectronitegrate_test ${GAUSSLECTRONITEGRATE_SOURCES} ${TEST_SOURCES})
target_include_directories(gausslectronitegrate_test PRIVATE ${GAUSSLECTRONITEGRATE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     gausslectronitegrate_test PRIVATE m ${HDF5_LIBRARIES})
if (OpenMP_C_FOUND)
    target_link_libraries( gausslectronitegrate_test PRIVATE OpenMP::OpenMP_C)
endif()

add_library(               gausslectronitegrate_pymodule SHARED ${GAUSSLECTRONITEGRATE_SOURCES} "python/pymodule.c")
target_include_directories(gausslectronitegrate_pymodule PRIVATE ${GAUSSLECTRONITEGRATE_DIRS} ${HDF5_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(     gausslectronitegrate_pymodule PRIVATE ${HDF5_LIBRARIES} ${Python3_LIBRARIES} Python3::NumPy)
if (OpenMP_C_FOUND)
    target_link_libraries( gausslectronitegrate_pymodule PRIVATE OpenMP::OpenMP_C)
endif()
set_target_properties(     gausslectronitegrate_pymodule PROPERTIES PREFIX "" OUTPUT_NAME "gausslectronitegrate_pymodule" LINKER_LANGUAGE C)

add_executable(            perf_eri_integrals ${GAUSSLECTRONITEGRATE_SOURCES} "perf/perf_eri_integrals.c")
target_include_directories(perf_eri_integrals PRIVATE ${GAUSSLECTRONITEGRATE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     perf_eri_integrals PRIVATE m ${HDF5_LIBRARIES})
if (OpenMP_C_FOUND)
    target_link_libraries( perf_eri_integrals PRIVATE OpenMP::OpenMP_C)
endif()

add_test(NAME gausslectronitegrate_test COMMAND gausslectronitegrate_test)

install(TARGETS gausslectronitegrate_pymodule DESTINATION .)
