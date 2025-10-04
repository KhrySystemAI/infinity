#ifndef INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_HPP
#define INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_HPP

#include <absl/container/flat_hash_map.h>
#include <absl/random/random.h>
#include <absl/synchronization/mutex.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "wdl.hpp"

namespace cinfinity::core {
class TranspositionTable {
 public:
  struct Entry {
   public:
    friend class TranspositionTable;

    Entry(absl::flat_hash_map<uint16_t, float> policy, WDL value,
          uint16_t lastUsed);

    [[nodiscard]] auto getPolicy(uint16_t move) const noexcept -> float;
    [[nodiscard]] auto getValue() const noexcept -> WDL;
    [[nodiscard]] auto getVisits() const noexcept -> size_t;
    [[nodiscard]] auto getLastUsed() const noexcept -> uint16_t;

    [[nodiscard]] auto size() const noexcept -> size_t;

   private:
    absl::flat_hash_map<uint16_t, float> m_policy;
    WDL m_value;
    size_t m_visits;
    uint16_t m_lastUsed;
  };  // struct Entry

  struct Bucket {
    friend class TranspositionTable;

   private:
    absl::flat_hash_map<uint64_t, std::unique_ptr<Entry>> m_data{};
    absl::Mutex m_lock{};
  };  // struct Bucket

  TranspositionTable();

  auto batchCreate(
      std::vector<std::tuple<uint8_t, uint64_t, std::unique_ptr<Entry>>>
          entries) -> bool;
  [[nodiscard]] auto read(std::tuple<uint8_t, uint64_t> key) -> Entry*;
  auto batchDelete(std::vector<std::tuple<uint8_t, uint64_t>> keys) -> bool;
  auto bytesDelete(size_t bytes) -> bool;

 private:
  std::array<std::unique_ptr<Bucket>, 256> m_buckets;
  uint16_t m_currentGeneration;
  absl::BitGen m_bitgen;

};  // class TranspositionTable
}  // namespace cinfinity::core

#endif  // INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_HPP

#ifndef CINFINITY_NO_IMPLEMENTATION
#include "transposition_table.inl"
#endif  // CINFINITY_NO_IMPLEMENTATION