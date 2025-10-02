#pragma once

#ifndef __INCLUDE_CINFINITY_WDL_HPP__
#define __INCLUDE_CINFINITY_WDL_HPP__

#include "config.hpp"
#include "mcts_mode.hpp"

namespace cinfinity {
    class WDL {
        std::array<float, 2> _data;
        
        public:
            inline WDL(float w, float d, float l) {
                float sum = w + d + l;
                _data[0] = w / sum;
                _data[1] = l / sum;
            }

            inline float win_chance() const {
                return _data[0];
            }

            inline float loss_chance() const {
                return _data[1];
            }

            inline float draw_chance() const {
                return 1 - (win_chance() + loss_chance());
            }

            inline size_t size() const {
                return _data.size();
            }

            inline float score(MCTSMode mode) const {
                switch(mode) {
                    case GREEDY:
                        return win_chance();
                    case BALANCED:
                        return win_chance() + (draw_chance() / 2);
                    case SOLID:
                        return (1.0 - loss_chance()) + (draw_chance() / 2);
                    case SAFE:
                        return (1.0 - loss_chance());
                }
            }
    };
}

#endif // __INCLUDE_CINFINITY_WDL_HPP__