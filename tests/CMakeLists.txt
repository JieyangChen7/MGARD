add_subdirectory(mgard)

find_package(Catch2 3.0.0)

if(Catch2_FOUND)
	configure_file(
		"${_MGARD_TEST_DIR}/include/testing_paths.hpp.in"
		"${CMAKE_CURRENT_BINARY_DIR}/mgard/include/testing_paths.hpp"
	)
	add_executable(mgard-tests ${MGARD_TESTS_CPP})
	target_include_directories(
		mgard-tests PRIVATE
		"${_MGARD_TEST_DIR}/include"
		"${CMAKE_CURRENT_BINARY_DIR}/mgard/include"
	)
	target_link_libraries(mgard-tests PRIVATE mgard-library Catch2::Catch2WithMain)
	set_target_properties(mgard-tests PROPERTIES OUTPUT_NAME tests)
	include(Catch)
	catch_discover_tests(mgard-tests TEST_PREFIX "mgard/")
else()
	message(STATUS "Will not build tests (Catch2 not found).")
endif()

set(Catch2_FOUND "${Catch2_FOUND}" PARENT_SCOPE)
