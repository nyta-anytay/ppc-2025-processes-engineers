#include "lazareva_a_torus_grid/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "lazareva_a_torus_grid/common/include/common.hpp"

namespace lazareva_a_torus_grid {

LazarevaATorusGridMPI::LazarevaATorusGridMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LazarevaATorusGridMPI::ValidationImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    return false;
  }

  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  int is_valid = 0;

  if (rank_ == 0) {
    const auto &input = GetInput();

    if (input.size() >= 3) {
      int source = input[0];
      int dest = input[1];
      is_valid = static_cast<int>(source >= 0 && source < world_size_ && dest >= 0 && dest < world_size_);
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid != 0;
}

bool LazarevaATorusGridMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  rows_ = static_cast<int>(std::sqrt(static_cast<double>(world_size_)));
  while (world_size_ % rows_ != 0) {
    rows_--;
  }
  cols_ = world_size_ / rows_;

  return true;
}

int LazarevaATorusGridMPI::CoordsToRank(int row, int col) const {
  row = ((row % rows_) + rows_) % rows_;
  col = ((col % cols_) + cols_) % cols_;
  return (row * cols_) + col;
}

int LazarevaATorusGridMPI::ShortestDirection(int from, int to, int size) {
  int forward = ((to - from) + size) % size;
  int backward = ((from - to) + size) % size;

  return (forward <= backward) ? 1 : -1;
}

int LazarevaATorusGridMPI::ComputeNextNode(int current, int dest) const {
  if (current == dest) {
    return -1;
  }

  int curr_row = current / cols_;
  int curr_col = current % cols_;
  int dest_row = dest / cols_;
  int dest_col = dest % cols_;

  if (curr_col != dest_col) {
    int dir = ShortestDirection(curr_col, dest_col, cols_);
    return CoordsToRank(curr_row, curr_col + dir);
  }

  if (curr_row != dest_row) {
    int dir = ShortestDirection(curr_row, dest_row, rows_);
    return CoordsToRank(curr_row + dir, curr_col);
  }

  return -1;
}

std::vector<int> LazarevaATorusGridMPI::ComputeFullPath(int source, int dest) const {
  std::vector<int> path;
  path.push_back(source);

  int current = source;
  while (current != dest) {
    int next = ComputeNextNode(current, dest);
    if (next == -1) {
      break;
    }
    path.push_back(next);
    current = next;
  }

  return path;
}

void LazarevaATorusGridMPI::SendData(const std::vector<int> &data, int dest_node) {
  int data_size = static_cast<int>(data.size());
  MPI_Send(&data_size, 1, MPI_INT, dest_node, 0, MPI_COMM_WORLD);
  if (data_size > 0) {
    MPI_Send(data.data(), data_size, MPI_INT, dest_node, 1, MPI_COMM_WORLD);
  }
}

std::vector<int> LazarevaATorusGridMPI::ReceiveData(int source_node) {
  int recv_size = 0;
  MPI_Recv(&recv_size, 1, MPI_INT, source_node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<int> recv_data;
  if (recv_size > 0) {
    recv_data.resize(recv_size);
    MPI_Recv(recv_data.data(), recv_size, MPI_INT, source_node, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return recv_data;
}

bool LazarevaATorusGridMPI::RunImpl() {
  int source = 0;
  int dest = 0;

  if (rank_ == 0) {
    const auto &input = GetInput();
    source = input[0];
    dest = input[1];
  }

  MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dest, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> path = ComputeFullPath(source, dest);
  int path_size = static_cast<int>(path.size());

  auto it = std::ranges::find(path, rank_);
  bool on_path = (it != path.end());
  int my_index = on_path ? static_cast<int>(std::distance(path.begin(), it)) : -1;

  std::vector<int> recv_data;

  if (source == dest) {
    if (rank_ == source) {
      const auto &input = GetInput();
      recv_data.assign(input.begin() + 2, input.end());
    }
  } else if (rank_ == source) {
    const auto &input = GetInput();
    recv_data.assign(input.begin() + 2, input.end());
    SendData(recv_data, path[1]);
  } else if (on_path) {
    int prev_node = path[my_index - 1];
    recv_data = ReceiveData(prev_node);

    if (rank_ != dest && (my_index + 1) < path_size) {
      SendData(recv_data, path[my_index + 1]);
    }
  }

  if (rank_ == dest) {
    GetOutput() = std::move(recv_data);
    GetOutput().push_back(path_size);
    GetOutput().insert(GetOutput().end(), path.begin(), path.end());
  } else {
    GetOutput().clear();
  }

  return true;
}

bool LazarevaATorusGridMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lazareva_a_torus_grid
