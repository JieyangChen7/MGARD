set(_MGARD_X_TEST_DIR "${CMAKE_CURRENT_LIST_DIR}")

if(NOT Catch2_FOUND)
  message(STATUS "Will not build mgard-x tests (Catch2 not found).")
  return()
endif()

if(NOT DEVICE_TYPE_LIST)
  message(STATUS "Will not build mgard-x tests (no device backend enabled).")
  return()
endif()

# Generate one test source file from a .cpp.in template for the given device type,
# set its compile language, and append it to MGARD_X_TEST_SOURCES.
macro(mgard_x_add_test SRC_NAME DEVICE_TYPE)
  set(_src "${_MGARD_X_TEST_DIR}/src/${SRC_NAME}.cpp.in")
  set(_gen "${CMAKE_CURRENT_BINARY_DIR}/src/${SRC_NAME}_${DEVICE_TYPE}.cpp")
  configure_file(${_src} ${_gen})
  if(${DEVICE_TYPE} STREQUAL "CUDA")
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE CUDA)
  elseif(${DEVICE_TYPE} STREQUAL "HIP")
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE HIP)
  else()
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE CXX)
  endif()
  list(APPEND MGARD_X_TEST_SOURCES ${_gen})
endmacro()

foreach(DEVICE_TYPE IN LISTS DEVICE_TYPE_LIST)
  set(MGARD_X_TEST_SOURCES "")

  mgard_x_add_test(test_Huffman         ${DEVICE_TYPE})
  mgard_x_add_test(test_BlockDelta      ${DEVICE_TYPE})
  mgard_x_add_test(test_DataRefactoring ${DEVICE_TYPE})
  mgard_x_add_test(test_Compressor      ${DEVICE_TYPE})
  mgard_x_add_test(test_Zstd            ${DEVICE_TYPE})
  mgard_x_add_test(test_RLE             ${DEVICE_TYPE})
  mgard_x_add_test(test_ComposedHuffmanZstd ${DEVICE_TYPE})

  # Portable (nvcomp-free) LZ4 backend: available on every device type.
  mgard_x_add_test(test_LZ4               ${DEVICE_TYPE})
  mgard_x_add_test(test_ComposedHuffmanLZ4 ${DEVICE_TYPE})

  set(TEST_TARGET "mgard-x-tests-${DEVICE_TYPE}")
  add_executable(${TEST_TARGET} ${MGARD_X_TEST_SOURCES})
  target_link_libraries(${TEST_TARGET} PRIVATE mgard-library Catch2::Catch2WithMain)

  string(TOLOWER "${DEVICE_TYPE}" DEVICE_TYPE_LOWER)
  catch_discover_tests(${TEST_TARGET} TEST_PREFIX "mgard-x/${DEVICE_TYPE_LOWER}/")
endforeach()
