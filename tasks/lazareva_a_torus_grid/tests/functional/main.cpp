#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "lazareva_a_torus_grid/common/include/common.hpp"
#include "lazareva_a_torus_grid/mpi/include/ops_mpi.hpp"
#include "lazareva_a_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace lazareva_a_torus_grid {

class LazarevaATorusGridFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
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

    TestType params = std::get<2>(GetParam());
    int data_size = std::get<0>(params);
    std::string test_name = std::get<1>(params);

    source_ = 0;
    if (is_seq_test_) {
      dest_ = source_;
    } else if (test_name == "vertical_path" || test_name == "large_vertical") {
      int rows = static_cast<int>(std::sqrt(static_cast<double>(world_size_)));
      while (world_size_ % rows != 0) {
        rows--;
      }
      int cols = world_size_ / rows;

      dest_ = (cols < world_size_) ? cols : 0;
    } else if (test_name == "diagonal_path") {
      dest_ = (world_size_ > 1) ? (world_size_ - 1) : 0;
    } else {
      dest_ = (world_size_ > 1) ? std::min(data_size % world_size_, world_size_ - 1) : 0;
    }

    input_data_.clear();
    input_data_.push_back(source_);
    input_data_.push_back(dest_);

    for (int i = 0; i < data_size; ++i) {
      input_data_.push_back(i + 1);
    }

    expected_data_.clear();
    for (int i = 0; i < data_size; ++i) {
      expected_data_.push_back(i + 1);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (is_seq_test_) {
      return CheckSeqOutput(output_data);
    }
    return CheckMpiOutput(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  [[nodiscard]] bool CheckSeqOutput(const OutType &output_data) const {
    std::size_t expected_size = expected_data_.size() + 3;

    if (output_data.size() != expected_size) {
      return false;
    }

    for (std::size_t i = 0; i < expected_data_.size(); ++i) {
      if (output_data[i] != expected_data_[i]) {
        return false;
      }
    }

    if (output_data[expected_data_.size()] != 2) {
      return false;
    }

    if (output_data[expected_data_.size() + 1] != source_) {
      return false;
    }
    if (output_data[expected_data_.size() + 2] != dest_) {
      return false;
    }

    return true;
  }

  [[nodiscard]] bool CheckMpiOutput(const OutType &output_data) const {
    if (rank_ != dest_) {
      return output_data.empty();
    }

    if (output_data.size() < expected_data_.size()) {
      return false;
    }

    for (std::size_t i = 0; i < expected_data_.size(); ++i) {
      if (output_data[i] != expected_data_[i]) {
        return false;
      }
    }

    if (output_data.size() > expected_data_.size()) {
      int path_size = output_data[expected_data_.size()];
      if (path_size < 1) {
        return false;
      }

      std::size_t path_start_idx = expected_data_.size() + 1;
      if (output_data.size() >= path_start_idx + static_cast<std::size_t>(path_size)) {
        int first_in_path = output_data[path_start_idx];
        int last_in_path = output_data[path_start_idx + path_size - 1];
        if (first_in_path != source_ || last_in_path != dest_) {
          return false;
        }
      }
    }

    return true;
  }

  InType input_data_;
  OutType expected_data_;
  int world_size_ = 1;
  int rank_ = 0;
  int dest_ = 0;
  int source_ = 0;
  bool is_seq_test_ = false;
};

namespace {

TEST_P(LazarevaATorusGridFuncTest, TorusGridDataTransfer) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "single_element"),     std::make_tuple(3, "small_data"),
    std::make_tuple(5, "medium_data"),        std::make_tuple(7, "odd_data"),
    std::make_tuple(10, "ten_elements"),      std::make_tuple(50, "fifty_elements"),
    std::make_tuple(100, "hundred_elements"), std::make_tuple(4, "vertical_path"),
    std::make_tuple(8, "diagonal_path"),      std::make_tuple(16, "large_vertical")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LazarevaATorusGridMPI, InType>(kTestParam, PPC_SETTINGS_lazareva_a_torus_grid),
    ppc::util::AddFuncTask<LazarevaATorusGridSEQ, InType>(kTestParam, PPC_SETTINGS_lazareva_a_torus_grid));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LazarevaATorusGridFuncTest::PrintFuncTestName<LazarevaATorusGridFuncTest>;

INSTANTIATE_TEST_SUITE_P(TorusGridTests, LazarevaATorusGridFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lazareva_a_torus_grid
