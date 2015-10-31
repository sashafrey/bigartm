project(bigartm)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../)

set(SRC_LIST
	srcmain.cc
	${CMAKE_SOURCE_DIR}/src/artm/cpp_interface.cc
	${CMAKE_SOURCE_DIR}/src/artm/messages.pb.cc
)

add_executable(bigartm ${SRC_LIST})

if (WIN32)
  target_link_libraries(bigartm artm ${Boost_LIBRARIES})
else()
  # Set options to enable static linkage
  set(CMAKE_EXE_LINKER_FLAGS "-static")
  # These two options deny dynamic linking either via -Wl,-Bdynamic or another way.
  # https://cmake.org/cmake/help/v3.0/prop_tgt/LINK_SEARCH_START_STATIC.html
  set_target_properties(bigartm PROPERTIES LINK_SEARCH_START_STATIC ON)
  # Link against all necessary libraries
  target_link_libraries(bigartm
      artm-static
	  ${PROTOBUF_LIBRARIES}
      ${Boost_LIBRARIES}
      ${GLOG_LIBRARIES}
      ${LINK_DL})
endif()

install(TARGETS bigartm DESTINATION bin)
