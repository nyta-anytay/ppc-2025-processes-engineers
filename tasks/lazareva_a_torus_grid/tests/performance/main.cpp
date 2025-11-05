#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>

#include "lazareva_a_torus_grid/common/include/common.hpp"
#include "lazareva_a_torus_grid/mpi/include/ops_mpi.hpp"
#include "lazareva_a_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lazareva_a_torus_grid {

class LazarevaATorusGridPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kCount = 10000000;
  InType input_data_;
  OutType expected_data_;
  int world_size_ = 1;
  int rank_ = 0;
  int source_ = 0;
  int dest_ = 0;
  bool is_seq_test_ = false;

  void SetUp() override {
    std::string task_name = std::get<1>(GetParam());
    is_seq_test_ = (task_name.find("seq") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    } else {
      world_size_ = 1;
      rank_ = 0;
    }

    // Для реального теста производительности - разные source и dest
    source_ = 0;
    if (is_seq_test_) {
      dest_ = 0;
    } else {
      dest_ = (world_size_ > 1) ? (world_size_ - 1) : 0;
    }

    input_data_.clear();
    input_data_.reserve(kCount + 2);
    input_data_.push_back(source_);
    input_data_.push_back(dest_);

    for (int i = 0; i < kCount; ++i) {
      input_data_.push_back(i);
    }

    expected_data_.clear();
    expected_data_.reserve(kCount);
    for (int i = 0; i < kCount; ++i) {
      expected_data_.push_back(i);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (is_seq_test_) {
      return CheckSeqOutput(output_data);
    }
    return CheckMpiOutput(output_data);
  }

  [[nodiscard]] bool CheckSeqOutput(const OutType &output_data) const {
    std::size_t expected_size = expected_data_.size() + 3;
    if (output_data.size() != expected_size) {
      return false;
    }
    // Проверяем только первый и последний элементы для скорости
    return output_data[0] == expected_data_[0] &&
           output_data[expected_data_.size() - 1] == expected_data_[expected_data_.size() - 1];
  }

  [[nodiscard]] bool CheckMpiOutput(const OutType &output_data) const {
    if (rank_ != dest_) {
      return output_data.empty();
    }
    if (output_data.size() < expected_data_.size()) {
      return false;
    }
    // Проверяем только первый и последний элементы для скорости
    return output_data[0] == expected_data_[0] &&
           output_data[expected_data_.size() - 1] == expected_data_[expected_data_.size() - 1];
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LazarevaATorusGridPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LazarevaATorusGridMPI, LazarevaATorusGridSEQ>(
    PPC_SETTINGS_lazareva_a_torus_grid);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LazarevaATorusGridPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LazarevaATorusGridPerfTest, kGtestValues, kPerfTestName);

}  // namespace lazareva_a_torus_grid
