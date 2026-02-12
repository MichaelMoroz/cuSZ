#ifndef PSZ_KERNEL_LRZ_GPU_CONFIG_HH
#define PSZ_KERNEL_LRZ_GPU_CONFIG_HH

#include <cuda_runtime.h>

#include <array>

#include "detail/composite.hh"

namespace {
dim3 div3(dim3 l, dim3 subl)
{
  return dim3((l.x - 1) / subl.x + 1, (l.y - 1) / subl.y + 1, (l.z - 1) / subl.z + 1);
};
}  // namespace

namespace psz::config {

struct utils {
  static int ndim(dim3 len3)
  {
    if (len3.z == 1 && len3.y == 1)
      return 1;
    else if (len3.z == 1 && len3.y != 1)
      return 2;
    else
      return 3;
  };

  static int ndim(std::array<size_t, 3> len3)
  {
    if (len3[2] == 1 && len3[1] == 1)
      return 1;
    else if (len3[2] == 1 && len3[1] != 1)
      return 2;
    else
      return 3;
  };
};

template <int dim, int X = 0, int Y = 0>
struct c_lorenzo;

template <int dim, int X = 0, int Y = 0>
struct x_lorenzo;

template <>
struct c_lorenzo<1> {
  static constexpr unsigned TILE_X = 1024, TILE_Y = 1, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 4, SEQ_Y = 1, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{TILE_X / SEQ_X, 1, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct c_lorenzo<2> {
  static constexpr unsigned TILE_X = 16, TILE_Y = 16, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 1, SEQ_Y = 8, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{16, 2, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct c_lorenzo<2, 32, 32> {
  static constexpr unsigned TILE_X = 32, TILE_Y = 32, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 1, SEQ_Y = 8, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{32, 4, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  static_assert(32 * SEQ_X == TILE_X);
  static_assert(4 * SEQ_Y == TILE_Y);

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct c_lorenzo<2, 64, 32> {                           // for uint16_t
  static constexpr unsigned TILE_X = 64, TILE_Y = 32, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 2, SEQ_Y = 8, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};         // 2-unit alignment
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};  // y-sequentiality == 8
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{32, 4, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  static_assert(32 * SEQ_X == TILE_X);
  static_assert(4 * SEQ_Y == TILE_Y);

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct c_lorenzo<2, 128, 32> {                          // for uint8_t
  static constexpr unsigned TILE_X = 128, TILE_Y = 32, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 4, SEQ_Y = 8, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};        // 4-unit x-alignment
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};  // y-sequentiality == 8
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{32, 4, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  static_assert(32 * SEQ_X == TILE_X);
  static_assert(4 * SEQ_Y == TILE_Y);

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct c_lorenzo<3> {
  static constexpr unsigned TILE_X = 32, TILE_Y = 8, TILE_Z = 8;
  static constexpr unsigned SEQ_X = 1, SEQ_Y = 1, SEQ_Z = 8;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};  // z-sequentiality == 8
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{32, 8, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  // TODO consistent tile-dim
  using Perf = psz::PredPerf<8, SEQ_X, 8, SEQ_Y, 8, SEQ_Z>;
};

template <>
struct x_lorenzo<1> {
  static constexpr unsigned TILE_X = 1024, TILE_Y = 1, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 4, SEQ_Y = 1, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};  // x-sequentiality == 8
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{TILE_X / SEQ_X, 1, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct x_lorenzo<2> {
  static constexpr unsigned TILE_X = 16, TILE_Y = 16, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 1, SEQ_Y = 8, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};  // y-sequentiality == 8
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{16, 2, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct x_lorenzo<2, 32> {
  static constexpr unsigned TILE_X = 32, TILE_Y = 32, TILE_Z = 1;
  static constexpr unsigned SEQ_X = 1, SEQ_Y = 8, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};  // y-sequentiality == 8
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{32, 4, 1};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

template <>
struct x_lorenzo<3> {
  static constexpr unsigned TILE_X = 32, TILE_Y = 8, TILE_Z = 8;
  static constexpr unsigned SEQ_X = 1, SEQ_Y = 8, SEQ_Z = 1;
  inline static dim3 tile{TILE_X, TILE_Y, TILE_Z};
  inline static dim3 sequentiality{SEQ_X, SEQ_Y, SEQ_Z};  // y-sequentiality == 8
  inline static dim3 seq{SEQ_X, SEQ_Y, SEQ_Z};
  inline static dim3 thread_block{32, 1, 8};
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = psz::PredPerf<TILE_X, SEQ_X, TILE_Y, SEQ_Y, TILE_Z, SEQ_Z>;
};

};  // namespace psz::config

#endif /* PSZ_KERNEL_LRZ_GPU_CONFIG_HH */
