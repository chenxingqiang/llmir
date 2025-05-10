# Attention Optimization Benchmarks CMakeLists.txt

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Benchmark executables
add_executable(simple_attention_benchmark simple_attention_benchmark.cpp)
add_executable(simple_mqa_benchmark simple_mqa_benchmark.cpp)
add_executable(simple_pruned_attention_benchmark simple_pruned_attention_benchmark.cpp)

# Install targets
install(TARGETS 
  simple_attention_benchmark
  simple_mqa_benchmark
  simple_pruned_attention_benchmark
  RUNTIME DESTINATION bin/benchmarks
)

# Install scripts
install(FILES
  run_benchmarks.sh
  parse_results.py
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION bin/benchmarks
)

# Create directory for benchmark results
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/benchmarks/results)
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/benchmarks/plots) 