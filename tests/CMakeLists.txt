find_package(Catch2 3.0.0)

if(Catch2_FOUND)
	include(Catch)
endif()

add_subdirectory(mgard)
add_subdirectory(mgard-x)

set(Catch2_FOUND "${Catch2_FOUND}" PARENT_SCOPE)
