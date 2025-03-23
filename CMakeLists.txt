project(gpt2)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CXX_STANDARD 26)

add_executable(gpt2 impl/main.cpp)
