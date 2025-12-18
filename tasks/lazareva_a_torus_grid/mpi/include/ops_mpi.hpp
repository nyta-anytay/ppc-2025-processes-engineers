#pragma once

#include <vector>

#include "lazareva_a_torus_grid/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lazareva_a_torus_grid {

class LazarevaATorusGridMPI : public BaseTask {
 public:
  explicit LazarevaATorusGridMPI(const InType &in);

  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] int CoordsToRank(int row, int col) const;
  [[nodiscard]] static int ShortestDirection(int from, int to, int size);
  [[nodiscard]] int ComputeNextNode(int current, int dest) const;
  [[nodiscard]] std::vector<int> ComputeFullPath(int source, int dest) const;

  static void SendData(const std::vector<int> &data, int dest_node);
  static std::vector<int> ReceiveData(int source_node);

  int world_size_ = 0;
  int rank_ = 0;
  int rows_ = 0;
  int cols_ = 0;
};

}  // namespace lazareva_a_torus_grid
