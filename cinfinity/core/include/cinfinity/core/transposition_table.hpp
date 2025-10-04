#ifndef INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_HPP
#define INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_HPP

#include <array>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>
#include <absl/random/random.h>

namespace cinfinity::core {
    class TranspositionTable {
        public:
            struct Entry {
                absl::flat_hash_map<uint16_t, float> policy;
                std::array<float, 3> value;
                size_t visits;
                uint16_t last_used;

                _NODISCARD size_t size() const noexcept;
            }; // struct Entry

            struct Bucket {
                absl::flat_hash_map<uint64_t, std::unique_ptr<Entry>> m_data;
                absl::Mutex m_lock;
            }; // struct Bucket

            TranspositionTable();

            bool batchCreate(std::vector<std::tuple<uint8_t, uint64_t, std::unique_ptr<Entry>>> entries);
            _NODISCARD Entry* read(std::tuple<uint8_t, uint64_t> key);
            bool batchDelete(std::vector<std::tuple<uint8_t, uint64_t>> keys);
            bool bytesDelete(size_t bytes);

        private:
            std::array<std::unique_ptr<Bucket>, 256> m_buckets;
            uint16_t m_currentGeneration;
            absl::BitGen m_bitgen;

    }; // class TranspositionTable
} // namespace cinfinity::core

#endif // INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_HPP

#ifndef CINFINITY_NO_IMPLEMENTATION
    #include "transposition_table.inl"
#endif // CINFINITY_NO_IMPLEMENTATION