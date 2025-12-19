#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "lazareva_a_gauss_filter_horizontal/common/include/common.hpp"
#include "lazareva_a_gauss_filter_horizontal/mpi/include/ops_mpi.hpp"
#include "lazareva_a_gauss_filter_horizontal/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lazareva_a_gauss_filter_horizontal {

class LazarevaAGaussFilterHorizontalPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kImageSize = 5000;

  InType input_data_;
  size_t expected_output_size_ = 0;

  void SetUp() override {
    int height = kImageSize;
    int width = kImageSize;
    int data_size = 2 + (height * width);

    input_data_.resize(data_size);
    input_data_[0] = height;
    input_data_[1] = width;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < height * width; i++) {
      input_data_[2 + i] = dist(gen);
    }

    expected_output_size_ = static_cast<size_t>(height) * static_cast<size_t>(width);
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
};

TEST_P(LazarevaAGaussFilterHorizontalPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LazarevaAGaussFilterHorizontalMPI, LazarevaAGaussFilterHorizontalSEQ>(
        PPC_SETTINGS_lazareva_a_gauss_filter_horizontal);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LazarevaAGaussFilterHorizontalPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LazarevaAGaussFilterHorizontalPerfTests, kGtestValues, kPerfTestName);

}  // namespace lazareva_a_gauss_filter_horizontal
