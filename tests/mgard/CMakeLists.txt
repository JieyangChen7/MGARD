set(_MGARD_TEST_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(
	MGARD_TESTS_CPP
	"${_MGARD_TEST_DIR}/src/testing_utilities.cpp"
	"${_MGARD_TEST_DIR}/src/test_utilities.cpp"
	"${_MGARD_TEST_DIR}/src/test_blas.cpp"
	"${_MGARD_TEST_DIR}/src/test_shuffle.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorMeshHierarchy.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorMeshHierarchyIteration.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorLinearOperator.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorMassMatrix.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorProlongation.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorRestriction.cpp"
	"${_MGARD_TEST_DIR}/src/test_LinearQuantizer.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorMultilevelCoefficientQuantizer.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorNorms.cpp"
	"${_MGARD_TEST_DIR}/src/test_TensorQuantityOfInterest.cpp"
	"${_MGARD_TEST_DIR}/src/test_compress.cpp"
	"${_MGARD_TEST_DIR}/src/test_decompose.cpp"
	"${_MGARD_TEST_DIR}/src/test_format.cpp"
	"${_MGARD_TEST_DIR}/src/test_quantize.cpp"
	# "${_MGARD_TEST_DIR}/src/test_compressors.cpp"
	"${_MGARD_TEST_DIR}/src/test_CompressedDataset.cpp"
)

if(MGARD_ENABLE_UNSTRUCTURED AND MOAB_FOUND)
	list(
		APPEND
		MGARD_TESTS_CPP
		"${_MGARD_TEST_DIR}/src/unstructured/test_estimators.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_indicators.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_norms.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_data.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_pcg.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_measure.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_MeshLevel.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_MeshHierarchy.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_MeshRefiner.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_LinearOperator.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_MassMatrix.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_UniformMeshHierarchy.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_UniformMeshRefiner.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_UniformEdgeFamilies.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_UniformRestriction.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_EnumeratedMeshRange.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_SituatedCoefficientRange.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_IndicatorInput.cpp"
		"${_MGARD_TEST_DIR}/src/unstructured/test_MultilevelCoefficientQuantizer.cpp"
	)
endif()

set(_MGARD_TEST_DIR "${_MGARD_TEST_DIR}" PARENT_SCOPE)
set(MGARD_TESTS_CPP "${MGARD_TESTS_CPP}" PARENT_SCOPE)
