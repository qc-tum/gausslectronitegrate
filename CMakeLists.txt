
cmake_minimum_required(VERSION 3.15)

project(gausslectronitegrate LANGUAGES C)

option(ENABLE_GAUSSLECTRONITEGRATE_UNIT_TESTS "Build and run the unit tests" ON)
option(ENABLE_GAUSSLECTRONITEGRATE_PYTHON_MODULE "Build the Python module library" ON)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

if (ENABLE_GAUSSLECTRONITEGRATE_UNIT_TESTS)
    include(CTest)
endif()

find_package(HDF5 REQUIRED COMPONENTS C)
if (ENABLE_GAUSSLECTRONITEGRATE_PYTHON_MODULE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
endif()
find_package(OpenMP)

set(CMAKE_C_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")

set(GAUSSLECTRONITEGRATE_INCLUDE_DIRS "include")
set(GAUSSLECTRONITEGRATE_SOURCES "src/gausslet_factors.c" "src/matrix.c" "src/symmetry.c" "src/kinetic_integrals.c" "src/nuclear_integrals.c" "src/eri_integrals.c" "src/util.c" "src/hdf5_util.c" "src/timing.c")
set(TEST_SOURCES "test/test_symmetry.c" "test/test_kinetic_integrals.c" "test/test_nuclear_integrals.c" "test/test_eri_integrals.c" "test/run_tests.c")

add_library(               gausslectronitegrate_lib ${GAUSSLECTRONITEGRATE_SOURCES})
target_include_directories(gausslectronitegrate_lib PUBLIC ${GAUSSLECTRONITEGRATE_INCLUDE_DIRS})
target_include_directories(gausslectronitegrate_lib PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(     gausslectronitegrate_lib PRIVATE ${HDF5_LIBRARIES})
if (OpenMP_C_FOUND)
    target_link_libraries( gausslectronitegrate_lib PRIVATE OpenMP::OpenMP_C)
endif()
set_target_properties(     gausslectronitegrate_lib PROPERTIES OUTPUT_NAME "gausslectronitegrate" LINKER_LANGUAGE C)

if (ENABLE_GAUSSLECTRONITEGRATE_UNIT_TESTS)
    add_executable(            gausslectronitegrate_test ${GAUSSLECTRONITEGRATE_SOURCES} ${TEST_SOURCES})
    target_include_directories(gausslectronitegrate_test PRIVATE ${GAUSSLECTRONITEGRATE_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
    target_link_libraries(     gausslectronitegrate_test PRIVATE ${HDF5_LIBRARIES})
    if (OpenMP_C_FOUND)
        target_link_libraries( gausslectronitegrate_test PRIVATE OpenMP::OpenMP_C)
    endif()
endif()

if (ENABLE_GAUSSLECTRONITEGRATE_PYTHON_MODULE)
    add_library(               gausslectronitegrate_pymodule ${GAUSSLECTRONITEGRATE_SOURCES} "python/pymodule.c")
    target_include_directories(gausslectronitegrate_pymodule PRIVATE ${GAUSSLECTRONITEGRATE_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(     gausslectronitegrate_pymodule PRIVATE ${HDF5_LIBRARIES} ${Python3_LIBRARIES} Python3::NumPy)
    if (OpenMP_C_FOUND)
        target_link_libraries( gausslectronitegrate_pymodule PRIVATE OpenMP::OpenMP_C)
    endif()
    set_target_properties(     gausslectronitegrate_pymodule PROPERTIES PREFIX "" OUTPUT_NAME "gausslectronitegrate_pymodule" LINKER_LANGUAGE C)
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)  # test whether current project is a stand-alone project
    add_executable(            perf_eri_integrals ${GAUSSLECTRONITEGRATE_SOURCES} "perf/perf_eri_integrals.c")
    target_include_directories(perf_eri_integrals PRIVATE ${GAUSSLECTRONITEGRATE_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
    target_link_libraries(     perf_eri_integrals PRIVATE ${HDF5_LIBRARIES})
    if (OpenMP_C_FOUND)
        target_link_libraries( perf_eri_integrals PRIVATE OpenMP::OpenMP_C)
    endif()
endif()

if (ENABLE_GAUSSLECTRONITEGRATE_UNIT_TESTS)
    add_test(NAME gausslectronitegrate_test COMMAND gausslectronitegrate_test)
endif()

if (ENABLE_GAUSSLECTRONITEGRATE_PYTHON_MODULE)
    install(TARGETS gausslectronitegrate_pymodule DESTINATION .)
endif()
