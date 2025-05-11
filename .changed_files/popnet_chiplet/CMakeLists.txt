#新的编译脚本

#https://blog.csdn.net/liitdar/article/details/81144461

cmake_minimum_required(VERSION 3.0)

project(popnet)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_BUILD_TYPE RELEASE)

set(POWER_RELEASE orion_power_beta_mar2003)
set(GRAPH_LIB graph_lib)

# Set build type to Debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

add_subdirectory(${POWER_RELEASE}/power)
add_subdirectory(${GRAPH_LIB})

set(CMAKE_CXX_FLAGS "-DS_OUT_DEBUG -Wno-deprecated -g -DPOWER_TEST")
include_directories(${POWER_RELEASE}/power ${POWER_RELEASE}/library)
include_directories(${GRAPH_LIB})

set(SIM_SRCS SString.cc SStd.cc SRGen.cc configuration.cc flit.cc sim_router.cc sim_routing.cc sim_vcarb.cc  mess_event.cc sim_foundation.cc sim_protoengine.cc mess_queue.cc main.cc)
aux_source_directory(changes NEW_SRCS)

link_directories(${POWER_RELEASE}/power)

find_package(Boost REQUIRED COMPONENTS
   graph
)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(popnet ${SIM_SRCS} ${NEW_SRCS})

target_link_libraries(popnet orion_power mygraph)

target_link_libraries(popnet ${Boost_LIBRARIES})
