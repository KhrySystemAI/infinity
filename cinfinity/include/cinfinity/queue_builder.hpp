#pragma once

#ifndef __INCLUDE_CINFINITY_QUEUE_BUILDER_HPP__
#define __INCLUDE_CINFINITY_QUEUE_BUILDER_HPP__

#include "transposition_table.hpp"
#include "model_handler.hpp"

namespace cinfinity {
    class QueueBuilder {
        TranspositionTable* transpositions = nullptr;
        ModelHandler* handler = nullptr;
        
        float u_coefficient = 1.0;
    }; // class QueueBuilder
} // namepace cinfinity

#endif // __INCLUDE_CINFINITY_QUEUE_BUILDER_HPP__