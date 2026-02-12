#include "../detail/extrema.thrust.inl"
__INSTANTIATE_THRUSTGPU_EXTREMA(float)

namespace psz::module {
template <>
void GPU_extrema<float>(float* d_ptr, size_t len, float res[4])
{
  psz::thrustgpu::GPU_extrema<float>(d_ptr, len, res);
}
}  // namespace psz::module
