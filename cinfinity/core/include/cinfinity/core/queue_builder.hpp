/**
 * @copyright (c) 2025 The Infinity Chess Engine Project.
 * This project is released under the GNU GPLv3 License.
 */
#ifndef INCLUDE_CINFINITY_CORE_QUEUE_BUILDER_HPP
#define INCLUDE_CINFINITY_CORE_QUEUE_BUILDER_HPP

#include <memory>

#include "transposition_table.hpp"

namespace cinfinity::core {
class QueueBuilder {
 public:
  explicit QueueBuilder(std::shared_ptr<TranspositionTable> transpositionTable);

  void run();

 private:
  std::shared_ptr<TranspositionTable> m_transpositionTable;
  // std::vector<std::shared_ptr<ModelHandler>> m_modelHandlers;

};  // class QueueBuilder
}  // namespace cinfinity::core

#endif  // INCLUDE_CINFINITY_CORE_QUEUE_BUILDER_HPP