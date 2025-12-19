#pragma once

#include <array>

#include "lazareva_a_gauss_filter_horizontal/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lazareva_a_gauss_filter_horizontal {

class LazarevaAGaussFilterHorizontalSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LazarevaAGaussFilterHorizontalSEQ(const InType &in);

 private:
  int height_ = 0;
  int width_ = 0;

  static constexpr std::array<std::array<int, 3>, 3> kKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
  static constexpr int kKernelSum = 16;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lazareva_a_gauss_filter_horizontal
