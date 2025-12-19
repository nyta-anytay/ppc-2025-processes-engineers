#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

#include "lazareva_a_gauss_filter_horizontal/common/include/common.hpp"
#include "lazareva_a_gauss_filter_horizontal/mpi/include/ops_mpi.hpp"
#include "lazareva_a_gauss_filter_horizontal/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lazareva_a_gauss_filter_horizontal {

class LazarevaAGaussFilterHorizontalFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int crop_size = std::get<0>(params);

    LoadBorschImage(crop_size);
  }

  void LoadBorschImage(int crop_size) {
    int width = -1;
    int height = -1;
    int channels = -1;

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_lazareva_a_gauss_filter_horizontal, "borsch.jpg");

    auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_grey);

    if (data == nullptr) {
      throw std::runtime_error("Failed to load borsch.jpg: " + std::string(stbi_failure_reason()));
    }

    int actual_crop = std::min({width, height, crop_size});

    input_data_.resize(2 + (static_cast<size_t>(actual_crop) * static_cast<size_t>(actual_crop)));
    input_data_[0] = actual_crop;
    input_data_[1] = actual_crop;

    for (int i = 0; i < actual_crop; i++) {
      for (int j = 0; j < actual_crop; j++) {
        input_data_[2 + (i * actual_crop) + j] = static_cast<int>(data[(i * width) + j]);
      }
    }

    stbi_image_free(data);

    expected_output_size_ = static_cast<size_t>(actual_crop) * static_cast<size_t>(actual_crop);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank != 0 && output_data.empty()) {
        return true;
      }
    }

    if (output_data.size() != expected_output_size_) {
      return false;
    }

    bool all_valid = std::ranges::all_of(output_data, [](int val) { return val >= 0 && val <= 255; });

    return all_valid;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  size_t expected_output_size_ = 0;
};

namespace {

TEST_P(LazarevaAGaussFilterHorizontalFuncTests, BorschImageTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(32, "borsch_32x32"), std::make_tuple(64, "borsch_64x64"), std::make_tuple(128, "borsch_128x128"),
    std::make_tuple(256, "borsch_256x256"), std::make_tuple(512, "borsch_512x512")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LazarevaAGaussFilterHorizontalMPI, InType>(
                                               kTestParam, PPC_SETTINGS_lazareva_a_gauss_filter_horizontal),
                                           ppc::util::AddFuncTask<LazarevaAGaussFilterHorizontalSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_lazareva_a_gauss_filter_horizontal));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    LazarevaAGaussFilterHorizontalFuncTests::PrintFuncTestName<LazarevaAGaussFilterHorizontalFuncTests>;

INSTANTIATE_TEST_SUITE_P(BorschImageTests, LazarevaAGaussFilterHorizontalFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lazareva_a_gauss_filter_horizontal
