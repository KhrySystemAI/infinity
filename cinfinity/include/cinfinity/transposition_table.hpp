/**
 * @file: transposition_table.hpp
 * 
 * @details Defines the core TranspositionTable, along with its Entry class and the method of 
 * hashing. The general structure of an entry is a map representing the network's move policy 
 * output, a value WDL output, and then some MCTS and eviction details. 
 * 
 * @note The class should in general support concurrent read, however write access is single-
 * threaded only, and should be locked to prevent issues. For more information, see the @ref 
 * QueueBuilder and  @ref ModelHandler classes.
 * 
 * @author KhrySystemAI
*/
#pragma once

#ifndef __INCLUDE_CINFINITY_TRANSPOSITION_TABLE_HPP__
#define __INCLUDE_CINFINITY_TRANSPOSITION_TABLE_HPP__

#include "config.hpp"
#include "wdl.hpp"

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
            hashing::map<chess::Move, float, Hash> policy;
            WDL value;
            uint16_t visits;
            uint16_t last_used;

            inline float getPolicy(chess::Move move) const {
                return policy.at(move);
            }

            inline size_t getBytes() const {
                constexpr size_t hashsize = sizeof(uint64_t);
                constexpr size_t keysize = sizeof(chess::PackedBoard);
                constexpr size_t uint16size = sizeof(uint16_t);
                constexpr size_t movesize = sizeof(chess::Move);
                constexpr size_t floatsize = sizeof(float);

                size_t policysize = policy.size() * (uint16size + movesize + floatsize);
                size_t valuesize = value.size() * floatsize;
                size_t visitsize = uint16size;
                size_t lastusedsize = uint16size;
                return hashsize + keysize + policysize + valuesize + visitsize + lastusedsize;
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
        bool insertDispatch(std::vector<std::pair<chess::PackedBoard, std::unique_ptr<Entry>>>& entries);

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

        /**
         * @brief Removes a certain number of bytes from the table.
         * 
         * @details This is designed to be used internally by the table when an insert would go
         * above the maximum table size. 
         * 
         * @param bytes The number of bytes to remove at minimum
        */
       bool removeBytes(size_t bytes);
        

        private:
            /**
             * @brief The underlying map object to the TranspositionTable.
             * 
             * @details The option to use segmented map instead of the regular map was due to the
             * lower peak memory requirement due to not having to deal with reallocation and frees
             * of memory with the regular map to keep the map contiguous. This means that overall,
             * more entries can be stored due to not requiring to always have half the RAM marked
             * as overflow. Perhaps this isn't a problem, or perhaps there are cleverer ways to 
             * deal with this.
            */
            hashing::segmented_map<chess::PackedBoard, std::unique_ptr<Entry>, Hash> m_table;

            /**
             * @brief The maximum size in bytes of the table. 
             * 
             * @details This should be set to around 75% of the maximum memory you want the engine 
             * to use. This does not have to be a power of 2. When an insert would increase the 
             * size of the table beyond this value, any entries with a last_used value less than
             * current generation (which mostly is the half move) could be removed to make space.
            */
            size_t maxSize;

            /**
             * @brief The value that is used for automatic eviction.
             * 
             * @details The removeBytes function will remove any values with a last_used value less
             * than this value.
            */
            uint16_t current_gen = 1;

    }; // struct TranspositionTable
} // namespace cinfinity

#endif // __INCLUDE_CINFINITY_TRANSPOSITION_TABLE_HPP__