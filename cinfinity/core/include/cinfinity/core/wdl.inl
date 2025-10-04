#ifndef INCLUDE_CINFINITY_CORE_WDL_INL
#define INCLUDE_CINFINITY_CORE_WDL_INL

#include "wdl.hpp"

namespace cinfinity::core {
WDL::WDL(float w, float d, float l) {
  float total = w + d + l;
  if (total < 1e-6) {
    m_data = {0.0f, 0.0f};
  } else {
    m_data = {w / total, l / total};
  }
}

float WDL::winChance() const noexcept {
  return m_data[0];
}

float WDL::drawChance() const noexcept {
  return 1 - (winChance() + lossChance());
}

float WDL::lossChance() const noexcept {
  return m_data[1];
}

WDL WDL::invert() const noexcept {
  return {m_data[1], drawChance(), m_data[0]};
}
}  // namespace cinfinity::core

#endif  // INCLUDE_CINFINITY_CORE_WDL_INL