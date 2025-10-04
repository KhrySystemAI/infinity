#ifndef INCLUDE_CINFINITY_CORE_WDL_HPP
#define INCLUDE_CINFINITY_CORE_WDL_HPP

#include <array>

namespace cinfinity::core {
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
}  // namespace cinfinity::core

#endif  // INCLUDE_CINFINITY_CORE_WDL_HPP

#ifndef CINFINITY_NO_IMPLEMENTATION
#include "wdl.inl"
#endif