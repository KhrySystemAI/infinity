/**
 * @file: transposition_table.hpp
 * 
 * @details Defines the core TranspositionTable, along with its Entry class and the method of 
 * hashing. The general structure of an entry is a dense 4096 one-hot representing the network's 
 * move policy output, a dense 192 one-hot representing the possible promotions in the position, a
 * value WDL output, and then some MCTS and eviction details. 
 * 
 * @note The class should in general support concurrent read, however write access is single-
 * threaded only. For more information, see the @ref QueueBuilder and  @ref ModelHandler classes.
 * 
 * @author KhrySystemAI
*/
#pragma once

#ifndef __INCLUDE_CINFINITY_TRANSPOSITION_TABLE_HPP__
#define __INCLUDE_CINFINITY_TRANSPOSITION_TABLE_HPP__

#include "config.hpp"

namespace cinfinity {
    /**
     * @brief Controls the main hash table for the engine.
     * 
     * @see https://www.chessprogramming.org/Transposition_Table
     * 
     * @details The transposition table holds the outputs of the model relative to its inputs, 
     * preventing the model from repeatedly processing the same position to get the same result. 
     * This implementation certainly could use some optimization.
     * 
    */
    struct TranspositionTable {
        /**
         * @class TranspositionTable::Hash
         * 
         * @brief Defines the hash function for the table.
         * 
         * @details The chess library this project uses has a builtin zobrist hashing function, 
         * but since we don't want to store the larger board object, we use the smaller PackedBoard
         * representation, which is 24 bytes. This does mean that to make it compatible with the 
         * map library, we have to pack the representation to compare, then unpack it to hash it,
         * which by writing one or the other as a custom implementation, could certainly be 
         * optimized. Something for the future I suppose.
        */
        struct Hash {
            using is_avalanching = void;

            inline auto operator()(chess::PackedBoard board) const noexcept {
                return chess::Board::Compact::decode(board).hash();
            }
        }; // struct Hash

        /**
         * @class TranspositionTable::Entry
         * 
         * @brief The table entry for the transposition table.
         * 
         * @details Stores the WDL for the current position, the model policy output, and some
         * additional details for MCTS and eviction .
        */
        struct Entry {
            struct Hash {
                using is_avalanching = void;

                inline auto operator()(chess::Move move) const noexcept {
                    return move.move();
                }
            };
            ankerl::unordered_dense::map<chess::Move, float, Hash> policy;
            std::array<float, 3> value;
            uint16_t visits;
            uint16_t last_used;

            inline float get_policy(chess::Move move) const {
                return policy.at(move);
            }
        }; // struct Entry

        /**
         * @brief Checks if the transposition table contains the given board state.
         * 
         * @details true if the table has an entry for the given board.
        */
        bool contains(chess::PackedBoard&);
        
        /**
         * @brief returns a batch of entries corresponding to the given board states. 
         * 
         * @details Entries are returned in the same order as the states passed in. Invalid states
         * will get a nullptr instead of a valid entry. This is designed to simplify NN batching.
         * 
         * @param keys A vector of all the board states to look for in the table.
         * @param entries An empty vector which will be populated with all the moves. 
        */
        void getDispatch(std::vector<chess::PackedBoard>& keys, std::vector<Entry*>& entries);

        /**
         * @brief Insert a batch of entries into the table with their corresponding board states. 
         * 
         * @details This is mostly designed as a simpler method for NN batching.
         * 
         * @param entries a vector of key/value pairs representing the board to the table entry.
         * 
         * @return A bool representing if a node was not written due to it already existing.
        */
        bool insertDispatch(std::vector<std::pair<chess::PackedBoard, Entry*>>& entries);

        /**
         * @brief Remove a batch entries from the table corresponding to the given board states.
         * 
         * @details This is mostly designed as a simpler method for NN batching.
         * 
         * @param keys a vector of all the board states to remove from the table.
         * 
         * @return A bool representing if all the nodes given existed before being removed.
        */
        bool removeDispatch(std::vector<chess::PackedBoard>& keys);

        bool getSubmap(uint16_t idx, uint16_t count, std::vector<std::pair<chess::PackedBoard, Entry*>>& result);
        

        private:
            /**
             * @brief The underlying map object to the TranspositionTable.
             * 
             * @details The option to use segmented map instead of the regular map was due to the
             * lower peak memory requirement due to not having to deal with reallocation and frees
             * of memory with the regular map to keep the map contiguous. This means that overall,
             * more entries can be stored due to not requiring to always have half the RAM marked
             * as overflow. Perhaps this isn't a problem.
            */
            ankerl::unordered_dense::segmented_map<chess::PackedBoard, Entry*, Hash> m_table;

    }; // struct TranspositionTable
} // namespace cinfinity

#endif // __INCLUDE_CINFINITY_TRANSPOSITION_TABLE_HPP__