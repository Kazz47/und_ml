include_directories(${GTEST_INCLUDE_DIRS} ${INCLUDE_DIRS})

# Testing Parameters
# set(GTEST_SHUFFLE 1) # Shuffle test execution order
# set(GTEST_RANDOM_SEED 0-9999)

set(test_sources
    matrix_ops_test
    log_sigmoid_test
    tan_sigmoid_test
    file_reader_test
    ann_test
    integration_test
)

add_executable(tests ${test_sources})

# Create dependency of the tests on googletest
add_dependencies(tests googletest)

target_link_libraries(tests
    ${GTEST_LIBS_DIR}/libgtest.a
    ${GTEST_LIBS_DIR}/libgtest_main.a
    pthread
    ann_shared
)

# Run the Tests
add_custom_command(
    OUTPUT testOut.dat
    COMMENT "Running tests"
    COMMAND tests --gtest_shuffle
)

add_custom_target(
    RunTest ALL
    DEPENDS testOut.dat
)

