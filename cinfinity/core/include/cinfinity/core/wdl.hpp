#ifndef INCLUDE_CINFINITY_CORE_WDL_HPP
#define INCLUDE_CINFINITY_CORE_WDL_HPP

#include <array>

namespace cinfinity::core {
    class WDL {
        public:
            WDL(float w, float d, float l);

            _NODISCARD float winChance() const noexcept;
            _NODISCARD float drawChance() const noexcept;
            _NODISCARD float lossChance() const noexcept;

        private:
            std::array<float, 2> m_data;
    };
} // namespace cinfinity::core

#endif // INCLUDE_CINFINITY_CORE_WDL_HPP

#ifndef CINFINITY_NO_IMPLEMENTATION
    #include "wdl.inl"
#endif