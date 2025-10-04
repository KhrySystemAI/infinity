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

#include "wdl.hpp"

namespace cinfinity::core {
    class TranspositionTable {
        public:
            struct Entry {
                public:
                    friend class TranspositionTable; 
                    
                    Entry(absl::flat_hash_map<uint16_t, float> policy, WDL value, uint16_t lastUsed);

                    [[nodiscard]] float getPolicy(uint16_t move) const noexcept;
                    [[nodiscard]] WDL getValue() const noexcept;
                    [[nodiscard]] size_t getVisits() const noexcept;
                    [[nodiscard]] uint16_t getLastUsed() const noexcept;

                    [[nodiscard]] size_t size() const noexcept;

                private:
                    absl::flat_hash_map<uint16_t, float> m_policy;
                    WDL m_value;
                    size_t m_visits;
                    uint16_t m_lastUsed;
            }; // struct Entry

            struct Bucket {
                friend class TranspositionTable;

                private:
                    absl::flat_hash_map<uint64_t, std::unique_ptr<Entry>> m_data;
                    absl::Mutex m_lock;
            }; // struct Bucket

            TranspositionTable();

            bool batchCreate(std::vector<std::tuple<uint8_t, uint64_t, std::unique_ptr<Entry>>> entries);
            [[nodiscard]] Entry* read(std::tuple<uint8_t, uint64_t> key);
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