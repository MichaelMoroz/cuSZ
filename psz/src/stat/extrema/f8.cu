#include "../detail/extrema.thrust.inl"
__INSTANTIATE_THRUSTGPU_EXTREMA(double)

namespace psz::module {
template <>
void GPU_extrema<double>(double* d_ptr, size_t len, double res[4])
{
  psz::thrustgpu::GPU_extrema<double>(d_ptr, len, res);
}
}  // namespace psz::module
