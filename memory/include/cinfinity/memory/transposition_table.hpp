/**
 * @file core/transposition_table.hpp
 * @brief Declares the TranspositionTable class and its supporting structures.
 *
 * @details
 * The TranspositionTable is a core component of the Infinity Chess Engine.
 * It stores previously evaluated positions to avoid redundant computation
 * and accelerate search performance.
 *
 * ### Components
 * - **Entry**: Represents a stored evaluation, including policy, value,
 *   visit count, and last-used generation tag.
 * - **Bucket**: A thread-safe container that maps 64-bit position hashes
 *   to Entries using `absl::flat_hash_map`.
 *
 * ### Concurrency
 * All methods within the table are designed to be fully thread-safe.
 *
 * ### Eviction
 * Eviction can occur based on memory limits or generation counts. The
 * `m_currentGeneration` field ensures that only entries older than the
 * current search generation may be pruned when calling functions such as
 * `bytesDelete()`.
 *
 * ### Methods
 * - `batchCreate()`: Insert multiple entries efficiently.
 * - `read()`: Lookup entries by key.
 * - `batchDelete()`: Remove multiple entries.
 * - `bytesDelete()`: Evict entries until a target byte count is freed.
 *
 * @note This table is designed to support Monte Carlo Tree Search (MCTS)
 * and PUCT-style policies for move selection and backpropagation.
 *
 * @copyright (c) 2025 The Infinity Chess Engine Project
 * @license SPDX-License-Identifier: GPL-3.0-only
 */
#ifndef INCLUDE_CINFINITY_MEMORY_TRANSPOSITION_TABLE_HPP
#define INCLUDE_CINFINITY_MEMORY_TRANSPOSITION_TABLE_HPP

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/random/random.h>
#include <absl/synchronization/mutex.h>

#include <cstddef>
#include <memory>
#include <tuple>

#include "wdl.hpp"

namespace cinfinity::memory {
class TranspositionTable {
  public:
  struct Entry {
    public:
    friend class TranspositionTable;

    Entry(
        absl::btree_map<uint16_t, float> policy, WDL value, uint16_t lastUsed
    );

    [[nodiscard]] auto getPolicy(uint16_t move) const noexcept -> float;
    [[nodiscard]] auto getValue() const noexcept -> WDL;
    [[nodiscard]] auto getVisits() const noexcept -> size_t;
    [[nodiscard]] auto getLastUsed() const noexcept -> uint16_t;

    [[nodiscard]] auto size() const noexcept -> size_t;

    private:
    absl::btree_map<uint16_t, float> m_policy;
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
      std::vector<std::tuple<uint8_t, uint64_t, std::unique_ptr<Entry>>> entries
  ) -> bool;
  [[nodiscard]] auto read(std::tuple<uint8_t, uint64_t> key) -> Entry*;
  auto batchDelete(std::vector<std::tuple<uint8_t, uint64_t>> keys) -> bool;
  auto bytesDelete(size_t bytes) -> bool;

  private:
  std::array<std::unique_ptr<Bucket>, 256> m_buckets{};
  uint16_t m_currentGeneration;
  absl::BitGen m_bitgen{};

};  // class TranspositionTable
}  // namespace cinfinity::memory

#endif  // INCLUDE_CINFINITY_MEMORY_TRANSPOSITION_TABLE_HPP

#ifndef CINFINITY_NO_IMPLEMENTATION
// NOLINTNEXTLINE(misc-include-cleaner)
#include "transposition_table.inl"
#endif  // CINFINITY_NO_IMPLEMENTATION