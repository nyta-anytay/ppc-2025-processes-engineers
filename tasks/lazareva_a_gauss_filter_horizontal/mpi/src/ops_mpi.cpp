#include "lazareva_a_gauss_filter_horizontal/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace lazareva_a_gauss_filter_horizontal {

LazarevaAGaussFilterHorizontalMPI::LazarevaAGaussFilterHorizontalMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LazarevaAGaussFilterHorizontalMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 0;

  if (rank == 0) {
    is_valid = static_cast<int>(
        (GetInput().size() >= 2) && GetOutput().empty() && (GetInput()[0] > 0) && (GetInput()[1] > 0) &&
        (GetInput()[0] <= std::numeric_limits<int>::max() / GetInput()[1]) &&
        (GetInput().size() == (2 + (static_cast<size_t>(GetInput()[0]) * static_cast<size_t>(GetInput()[1])))));
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return static_cast<bool>(is_valid);
}

bool LazarevaAGaussFilterHorizontalMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput().clear();
    height_ = GetInput()[0];
    width_ = GetInput()[1];
  }

  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().resize(height_ * width_);
  }

  return true;
}

bool LazarevaAGaussFilterHorizontalMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Calculate row distribution
  int rows_per_proc = height_ / size;
  int remainder = height_ % size;

  // Calculate local parameters for each process
  std::vector<int> rows_count(size);
  std::vector<int> rows_offset(size);

  int offset = 0;
  for (int i = 0; i < size; i++) {
    rows_count[i] = rows_per_proc + (i < remainder ? 1 : 0);
    rows_offset[i] = offset;
    offset += rows_count[i];
  }

  int local_rows = rows_count[rank];
  int local_start_row = rows_offset[rank];

  // Calculate extended region (with halo rows for filtering)
  int halo_top = (local_start_row > 0) ? 1 : 0;
  int halo_bottom = (local_start_row + local_rows < height_) ? 1 : 0;
  int extended_rows = local_rows + halo_top + halo_bottom;

  // Prepare sendcounts and displacements for extended regions
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    int start = rows_offset[i];
    int count = rows_count[i];
    int htop = (start > 0) ? 1 : 0;
    int hbot = (start + count < height_) ? 1 : 0;

    sendcounts[i] = (count + htop + hbot) * width_;
    displs[i] = (start - htop) * width_;
  }

  // Scatter extended data to all processes
  std::vector<int> local_data(extended_rows * width_);

  MPI_Scatterv(rank == 0 ? GetInput().data() + 2 : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               local_data.data(), extended_rows * width_, MPI_INT, 0, MPI_COMM_WORLD);

  // Apply Gaussian filter
  std::vector<int> local_result(local_rows * width_);

  for (int i = 0; i < local_rows; i++) {
    int ext_i = i + halo_top;  // Position in extended buffer

    for (int j = 0; j < width_; j++) {
      int sum = 0;

      for (int ki = -1; ki <= 1; ki++) {
        for (int kj = -1; kj <= 1; kj++) {
          int row = ext_i + ki;
          int col = j + kj;

          // Clamp row within extended buffer
          if (row < 0) {
            row = 0;
          }
          if (row >= extended_rows) {
            row = extended_rows - 1;
          }

          // Clamp column
          if (col < 0) {
            col = 0;
          }
          if (col >= width_) {
            col = width_ - 1;
          }

          int pixel_value = local_data[row * width_ + col];
          sum += pixel_value * kernel_[ki + 1][kj + 1];
        }
      }

      local_result[i * width_ + j] = sum / kernel_sum_;
    }
  }

  // Prepare gather parameters (only actual rows, not halos)
  std::vector<int> recvcounts(size);
  std::vector<int> recvdispls(size);

  offset = 0;
  for (int i = 0; i < size; i++) {
    recvcounts[i] = rows_count[i] * width_;
    recvdispls[i] = offset;
    offset += recvcounts[i];
  }

  // Gather results
  MPI_Gatherv(local_result.data(), local_rows * width_, MPI_INT, rank == 0 ? GetOutput().data() : nullptr,
              recvcounts.data(), recvdispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool LazarevaAGaussFilterHorizontalMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(height_ * width_));
  }

  return true;
}

}  // namespace lazareva_a_gauss_filter_horizontal
