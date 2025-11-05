#pragma once

#include "lazareva_a_torus_grid/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lazareva_a_torus_grid {

class LazarevaATorusGridSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LazarevaATorusGridSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lazareva_a_torus_grid
