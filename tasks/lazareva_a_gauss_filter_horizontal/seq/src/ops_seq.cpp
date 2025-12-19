#include "lazareva_a_gauss_filter_horizontal/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "lazareva_a_gauss_filter_horizontal/common/include/common.hpp"

namespace lazareva_a_gauss_filter_horizontal {

namespace {
int GetKernelValue(int ki, int kj) {
  constexpr std::array<std::array<int, 3>, 3> kKernelLocal = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
  return kKernelLocal.at(ki).at(kj);
}
}  // namespace

LazarevaAGaussFilterHorizontalSEQ::LazarevaAGaussFilterHorizontalSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LazarevaAGaussFilterHorizontalSEQ::ValidationImpl() {
  return (GetInput().size() >= 2) && GetOutput().empty() && (GetInput()[0] > 0) && (GetInput()[1] > 0) &&
         (GetInput()[0] <= std::numeric_limits<int>::max() / GetInput()[1]) &&
         (GetInput().size() == (2 + (static_cast<size_t>(GetInput()[0]) * static_cast<size_t>(GetInput()[1]))));
}

bool LazarevaAGaussFilterHorizontalSEQ::PreProcessingImpl() {
  height_ = GetInput()[0];
  width_ = GetInput()[1];

  GetOutput().clear();
  GetOutput().resize(static_cast<size_t>(height_) * static_cast<size_t>(width_));

  return true;
}

bool LazarevaAGaussFilterHorizontalSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  for (int i = 0; i < height_; i++) {
    for (int j = 0; j < width_; j++) {
      int sum = 0;

      for (int ki = 0; ki < 3; ki++) {
        for (int kj = 0; kj < 3; kj++) {
          int row = std::clamp(i + ki - 1, 0, height_ - 1);
          int col = std::clamp(j + kj - 1, 0, width_ - 1);

          int pixel_value = input[2 + (row * width_) + col];
          sum += pixel_value * GetKernelValue(ki, kj);
        }
      }

      output[(i * width_) + j] = sum / kKernelSum;
    }
  }

  return true;
}

bool LazarevaAGaussFilterHorizontalSEQ::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == (static_cast<size_t>(height_) * static_cast<size_t>(width_)));
}

}  // namespace lazareva_a_gauss_filter_horizontal
