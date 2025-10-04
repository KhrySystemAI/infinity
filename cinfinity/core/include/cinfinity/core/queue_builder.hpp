/** 
 * @file core/queue_builder.hpp
 * @brief Declares the QueueBuilder class.
 * 
 * @details The QueueBuilder is the struct in charge of pushing nodes to the 
 * ModelHandler objects. It follows a Monte Carlo Tree Simulation, with a 
 * slight tweak. Where normally in naive MCTS implementations you can step 
 * through to the end of a game from any node and then backpropagate from the 
 * end to the node you're at, effectively exploring the whole tree, this
 * modified version follows the following general formula:
 * 
 * \f[
 * \begin{aligned}
 * Q(s,a) &= 
 *   \begin{cases} 
 *     \text{score}(V(s,a)) & N(s,a) > 0 \\
 *     0 & N(s,a) = 0
 *   \end{cases} \\[6pt]
 * U(s,a) &= c_{puct} \cdot P(s,a) \cdot 
 *      \sqrt{\frac{N(s) + 1}{N(s,a) + 1}} \\[6pt]
 * \text{PUCT}(s,a) &= Q(s,a) + U(s,a)
 * \end{aligned}
 * \f]
 * 
 * **Where:**
 * - \( Q(s,a) \): exploitation term (expected value of action \(a\) in state \(s\)).  
 * - \( U(s,a) \): exploration bonus.  
 * - \( P(s,a) \): prior probability from the neural network policy.  
 * - \( N(s) \): total visit count of the parent node.  
 * - \( N(s,a) \): visit count of the child node for action \(a\).  
 * - \( V(s,a) \): value estimate of the node \((s,a)\).  
 * - \( c_{puct} \): exploration constant balancing exploration vs. exploitation.  
 * - `score(...)`: function mapping stored value into normalized Q-value.  
 * 
 * This balances exploitation of known strong moves with exploration of 
 * moves that have strong policy priors but less search data. 
 * 
 * @copyright (c) 2025 The Infinity Chess Engine Project.
 * @license SPDX-License-Identifier: GPL-3.0-only
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