#ifndef INCLUDE_CINFINITY_MEMORY_WDL_HPP
#define INCLUDE_CINFINITY_MEMORY_WDL_HPP

#include <array>

namespace cinfinity::memory {
class WDL {
 public:
  WDL(float win, float draw, float loss);

  [[nodiscard]] auto winChance() const noexcept -> float;
  [[nodiscard]] auto drawChance() const noexcept -> float;
  [[nodiscard]] auto lossChance() const noexcept -> float;
  [[nodiscard]] auto invert() const noexcept -> WDL;

 private:
  std::array<float, 2> m_data;
};
}  // namespace cinfinity::memory

#endif  // INCLUDE_CINFINITY_MEMORY_WDL_HPP

#ifndef CINFINITY_NO_IMPLEMENTATION
// NOLINTNEXTLINE(misc-include-cleaner)
#include "wdl.inl"
#endif