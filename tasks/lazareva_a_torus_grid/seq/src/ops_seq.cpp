#include "lazareva_a_torus_grid/seq/include/ops_seq.hpp"

#include <vector>

#include "lazareva_a_torus_grid/common/include/common.hpp"

namespace lazareva_a_torus_grid {

LazarevaATorusGridSEQ::LazarevaATorusGridSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LazarevaATorusGridSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.size() < 3) {
    return false;
  }

  int source = input[0];
  int dest = input[1];

  return source >= 0 && dest >= 0;
}

bool LazarevaATorusGridSEQ::PreProcessingImpl() {
  return true;
}

bool LazarevaATorusGridSEQ::RunImpl() {
  const auto &input = GetInput();

  int source = input[0];
  int dest = input[1];

  std::vector<int> data(input.begin() + 2, input.end());

  GetOutput() = data;

  GetOutput().push_back(2);
  GetOutput().push_back(source);
  GetOutput().push_back(dest);

  return true;
}

bool LazarevaATorusGridSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lazareva_a_torus_grid
