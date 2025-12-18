#include "lazareva_a_gauss_filter_horizontal/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "lazareva_a_gauss_filter_horizontal/common/include/common.hpp"
#include "util/include/util.hpp"

namespace lazareva_a_gauss_filter_horizontal {

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

  // Prepare output buffer
  GetOutput().clear();
  GetOutput().resize(height_ * width_);

  return true;
}

bool LazarevaAGaussFilterHorizontalSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  // Extract image data (skip first 2 elements: height and width)
  std::vector<int> image(input.begin() + 2, input.end());

  // Apply Gaussian filter to each pixel
  for (int i = 0; i < height_; i++) {
    for (int j = 0; j < width_; j++) {
      int sum = 0;

      // Convolve with 3x3 kernel
      for (int ki = -1; ki <= 1; ki++) {
        for (int kj = -1; kj <= 1; kj++) {
          int row = i + ki;
          int col = j + kj;

          // Handle boundaries by clamping (replicate edge pixels)
          if (row < 0) {
            row = 0;
          }
          if (row >= height_) {
            row = height_ - 1;
          }
          if (col < 0) {
            col = 0;
          }
          if (col >= width_) {
            col = width_ - 1;
          }

          int pixel_value = image[row * width_ + col];
          int kernel_value = kernel_[ki + 1][kj + 1];

          sum += pixel_value * kernel_value;
        }
      }

      // Normalize by kernel sum and store result
      output[i * width_ + j] = sum / kernel_sum_;
    }
  }

  return true;
}

bool LazarevaAGaussFilterHorizontalSEQ::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(height_ * width_));
}

}  // namespace lazareva_a_gauss_filter_horizontal
