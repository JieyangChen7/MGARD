

// #include "/home/jieyang/dev/MGARD/include/cuda/FlyingEdges.hpp"

#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/cont/Initialize.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/cont/ColorTable.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandlePermutation.h>
#include <vtkm/cont/ArrayHandleUniformPointCoordinates.h>

#include <vtkm/worklet/contour/CommonState.h>
#include <vtkm/worklet/contour/FieldPropagation.h>
#include <vtkm/worklet/contour/FlyingEdges.h>
#include <vtkm/worklet/contour/MarchingCells.h>
#include <vtkm/worklet/SurfaceNormals.h>

#include <vtkm/filter/PolicyBase.h>
#include <vtkm/filter/FieldMetadata.h>
#include <vtkm/filter/FilterDataSetWithField.h>
#include <vtkm/filter/MapFieldPermutation.h>
#include <vtkm/filter/Contour.h>

#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/rendering/MapperWireframer.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/Actor.h>
#include <vtkm/rendering/Scene.h>
#include <vtkm/rendering/View3D.h>

#include "FlyingEdges.hpp"


bool require_arg(int argc, char *argv[], std::string option) {
  for (int i = 0; i < argc; i++) {
    if (option.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  exit(-1);
}

template <typename T> 
size_t readfile(const char *input_file, T *&in_buff) {
  std::cout << mgard_cuda::log::log_info << "Loading file: " << input_file
            << "\n";

  FILE *pFile;
  pFile = fopen(input_file, "rb");
  if (pFile == NULL) {
    std::cout << mgard_cuda::log::log_err << "file open error!\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff = (T *)malloc(lSize);
  lSize = fread(in_buff, 1, lSize, pFile);
  fclose(pFile);
  // min_max(lSize/sizeof(T), in_buff);
  return lSize;
}

std::string get_arg(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    for (int i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        return std::string(argv[i + 1]);
      }
    }
  }
  return std::string("");
}


int get_arg_int(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      int d = std::stoi(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      exit(-1);
    }
  }
  return 0;
}

std::vector<mgard_cuda::SIZE> get_arg_dims(int argc, char *argv[],
                                           std::string option) {
  std::vector<mgard_cuda::SIZE> shape;
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int arg_idx = 0;
    for (int i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
        arg_idx = i + 1;
      }
    }
    try {
      int d = std::stoi(arg);
      for (int i = 0; i < d; i++) {
        shape.push_back(std::stoi(argv[arg_idx + 1 + i]));
      }
      return shape;
    } catch (std::invalid_argument const &e) {
      exit(-1);
    }
  }
  return shape;
}

double get_arg_double(int argc, char *argv[], std::string option) {
  if (require_arg(argc, argv, option)) {
    std::string arg;
    int i;
    for (i = 0; i < argc; i++) {
      if (option.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
      }
    }
    try {
      double d = std::stod(arg);
      return d;
    } catch (std::invalid_argument const &e) {
      exit(-1);
    }
  }
  return 0;
}

template <typename T>
void test_vtkm(int argc, char *argv[], std::vector<mgard_cuda::SIZE> shape, T *original_data, T iso_value) {
  // vtkm::cont::InitializeOptions options = vtkm::cont::InitializeOptions::
  vtkm::cont::Initialize(argc, argv);
  vtkm::cont::RuntimeDeviceTracker& deviceTracker = vtkm::cont::GetRuntimeDeviceTracker();
  deviceTracker.ForceDevice( vtkm::cont::DeviceAdapterTagCuda());
  vtkm::Id3 dims(shape[0], shape[1], shape[2]);
  vtkm::Id3 org(0, 0, 0);
  vtkm::Id3 spc(1, 1, 1);

  size_t original_size = 1;
  for (mgard_cuda::DIM i = 0; i < shape.size(); i++)
    original_size *= shape[i];


  vtkm::cont::DataSet inputDataSet;
  vtkm::cont::DataSetBuilderUniform dataSetBuilderUniform;
  vtkm::cont::DataSetFieldAdd dsf;
  inputDataSet = dataSetBuilderUniform.Create(dims, org, spc);
  std::vector<T> vec_data(original_data, original_data+original_size);
  dsf.AddPointField(inputDataSet, "v", vec_data);

  vtkm::filter::Contour contour;
  contour.SetGenerateNormals(true);
  contour.SetMergeDuplicatePoints(true);
  contour.SetNumberOfIsoValues(1);
  contour.SetIsoValue(0, iso_value);
  contour.SetActiveField("v");

  vtkm::cont::DataSet ds_from_mc = contour.Execute(inputDataSet);
}


template <typename T>
void test_mine(std::vector<mgard_cuda::SIZE> shape, T *original_data, T iso_value) {
  mgard_cuda::Array<3, T, mgard_cuda::CUDA> v({shape[2], shape[1], shape[0]});
  v.loadData(original_data);

  mgard_cuda::Array<1, mgard_cuda::SIZE, mgard_cuda::CUDA> Triangles;
  mgard_cuda::Array<1, T, mgard_cuda::CUDA> Points;

  mgard_cuda::FlyingEdges<T, mgard_cuda::CUDA>().Execute(shape[2], shape[1], shape[0],
                                     mgard_cuda::SubArray<3, T, mgard_cuda::CUDA>(v),
                                     iso_value, Triangles, Points, 0);

  mgard_cuda::PrintSubarray("Triangles", mgard_cuda::SubArray(Triangles));
  mgard_cuda::PrintSubarray("Points", mgard_cuda::SubArray(Points));

  
}

int main(int argc, char *argv[]) {

  std::cout << "start\n";
  std::string input_file = get_arg(argc, argv, "-i");
  mgard_cuda::DIM D = get_arg_int(argc, argv, "-n");
  std::vector<mgard_cuda::SIZE> shape = get_arg_dims(argc, argv, "-n");

  size_t original_size = 1;
  for (mgard_cuda::DIM i = 0; i < D; i++)
    original_size *= shape[i];
  float *original_data;
  size_t in_size = 0;

  if (std::string(input_file).compare("random") == 0) {
    std::cout << "generating data...";
    in_size = original_size * sizeof(float);
    original_data = new float[original_size];
    for (size_t i = 0; i < shape[2]; i++){
      for (size_t j = 0; j < shape[1]; j++){
        for (size_t k = 0; k < shape[0]; k++){
          original_data[i*shape[1]*shape[0]+j*shape[0]+k] = j;
        }
      }
    }
    std::cout << "Done\n";
  } else {
    in_size = readfile(input_file.c_str(), original_data);
  }
  if (in_size != original_size * sizeof(float)) {
    std::cout << mgard_cuda::log::log_err << "input file size mismatch" << in_size << "vs." << original_size * sizeof(float) << "!\n";
  }

  float iso_value = 1.5;
  std::cout << "test_vtkm\n";
  test_vtkm(argc, argv, shape, original_data, iso_value);
  std::cout << "test_mine\n";
  test_mine(shape, original_data, iso_value);


  


}