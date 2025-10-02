#pragma once

#include <array>
#include <limits>
#include <random>
#include <vector>
#include <queue>
#include <ankerl/unordered_dense.h>
#include "chess.hpp"


struct TranspositionTable {
    struct Hash {
        using is_avalanching = void;

        inline auto operator()(chess::PackedBoard& board) const noexcept {
            return chess::Board::Compact::decode(board).hash();
        }
    };

    struct Entry {
        std::array<float, 4096> policy;
        std::array<float, 192> promo;
        std::array<float, 3> value;
        uint16_t visits;

        inline float get_policy(int from, int to) const {
            return policy[(from << 6) | to];
        }

        inline float get_promo(int from, int to, int piece) const {
            auto fromcol = from % 8;
            auto side = from / 8 >= 4 ? 0 : 1; 
            auto tocol = to % 8;
            auto diff = (fromcol - tocol) + 1;
            auto idx = (fromcol * 24) + (side * 12) + (diff * 4) + piece;
            return promo[idx];
        }
    };

    ankerl::unordered_dense::segmented_map<chess::PackedBoard, Hash, Entry*> m_table;

    bool contains(chess::PackedBoard&);
    Entry* get(chess::PackedBoard&);
    bool insert(chess::PackedBoard&, Entry* entry);
};

class Processor {
    std::queue<chess::Board> m_queue;
    std::mutex m_lock;

    public:
        bool run(); // Batches to the NN
        void lock();
        void unlock();
        bool in_queue(chess::Board&);
        void queue(chess::Board&);
};

class MCTS {
    struct Node {
        chess::Board board;
        chess::Move lastmove;
        std::vector<Node*> children;
        Node* parent;
        // rest of data can be queued from table
    };

    enum EvalMode {
        GREEDY,
        BALANCED,
        SOLID,
        SAFE,
    };

    EvalMode mode;
    TranspositionTable* transpositions;
    Processor* processor;
    float u_coefficient = 1.0;

    float score(std::array<float, 3>& wdl) {
        switch(mode) {
            case GREEDY:
                return wdl[2];
            case BALANCED:
                return wdl[2] + (wdl[1] / 2);
            case SOLID:
                return (1.0 - wdl[0]) + (wdl[1] / 2);
            case SAFE:
                return (1.0 - wdl[0]);
        }
    }
    /*
    Node* select(Node* node) {
        chess::PackedBoard parent = chess::Board::Compact::encode(node->board);
        TranspositionTable::Entry* parent_entry = transpositions->get(parent);

        float best_child_score = -std::numeric_limits<float>::infinity();
        Node* best_child = nullptr;

        for(Node* child : node->children) {
            chess::PackedBoard packed = chess::Board::Compact::encode(child->board);
            TranspositionTable::Entry* entry = transpositions->get(packed);

            uint16_t idx = (child->lastmove.from().index() << 6) + child->lastmove.to().index();
            float q_value = (entry->visits > 0) ? score(entry->value) : 0.0f;
            float u_value = u_coefficient * parent_entry->policy[idx] * std::sqrt((parent_entry->visits + 1) / (entry->visits + 1));
            float score = q_value + u_value;
            if(score > best_child_score) {
                best_child = child;
                best_child_score = score;
            }
        }

        return best_child;
    }
    */

    int sample_move(const std::vector<float>& probs) {
        // compute cumulative distribution (CDF)
        std::vector<float> cdf(probs.size());
        cdf[0] = probs[0];
        for (size_t i = 1; i < probs.size(); ++i) {
            cdf[i] = cdf[i-1] + probs[i];
        }

        // sample a random float in [0, total]
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dis(0.0f, cdf.back());
        float r = dis(gen);

        // find the first index where r <= cdf[i]
        for (size_t i = 0; i < cdf.size(); ++i) {
            if (r <= cdf[i]) return i;
        }

        return cdf.size() - 1; // fallback
    }

    chess::Board& select(chess::Board& board) {
        auto pack = chess::Board::Compact::encode(board);

        if(!transpositions->contains(pack)) {
            processor->lock();
            if(!processor->in_queue(board))
                processor->queue(board);
            processor->unlock();
            return board;
        }

        auto entry = transpositions->get(pack);
        std::array<float, 4096> cdf;

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        std::vector<float> move_probs(moves.size());
        for(int i = 0; i < moves.size(); i++) {
            auto move = moves[i];
            float prob = entry->get_policy(move.from().index(), move.to().index());
            float prom = 1.0f;
            if(move.typeOf() == chess::Move::PROMOTION) {
                auto promo = move.promotionType();
                switch(promo.internal()) {
                    case chess::PieceType::QUEEN:
                        prom = entry->get_promo(move.from().index(), move.to().index(), 0);
                        break;
                    case chess::PieceType::ROOK:
                        prom = entry->get_promo(move.from().index(), move.to().index(), 1);
                        break;
                    case chess::PieceType::BISHOP:
                        prom = entry->get_promo(move.from().index(), move.to().index(), 2);
                        break;
                    case chess::PieceType::KNIGHT:
                        prom = entry->get_promo(move.from().index(), move.to().index(), 3);
                        break;
                };
            }
            move_probs[i] = prob * prom;
        }
        auto idx = sample_move(move_probs);
        board.makeMove(moves[idx]);
        return board;
    }
};