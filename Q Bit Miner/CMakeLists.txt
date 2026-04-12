cmake_minimum_required(VERSION 3.22)

project(QuantumMiner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QBIT_MINER_BUILD_TESTS "Build Q Bit Miner substrate tests" ON)

add_library(qbit_miner_core
    src/app/application.cpp
    src/cache/substrate_cache.cpp
    src/runtime/runtime_bus.cpp
    src/runtime/substrate_controller.cpp
    src/substrate/calibration_export.cpp
    src/substrate/device_validation_export.cpp
    src/substrate/field_dynamics.cpp
    src/substrate/research_calibration_importer.cpp
    src/substrate/trace_serialization.cpp
)

target_include_directories(qbit_miner_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(qbit_miner_core PUBLIC cxx_std_20)

if(WIN32)
    target_compile_definitions(qbit_miner_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_executable(quantum_miner
    src/main.cpp
)

target_link_libraries(quantum_miner PRIVATE qbit_miner_core)
set_target_properties(quantum_miner PROPERTIES OUTPUT_NAME "Quantum Miner")

if(QBIT_MINER_BUILD_TESTS)
    enable_testing()

    add_executable(qbit_miner_tests
        tests/test_photonic_identity.cpp
    )

    target_link_libraries(qbit_miner_tests PRIVATE qbit_miner_core)
    add_test(NAME qbit_miner_tests COMMAND qbit_miner_tests)
endif()