/**************************************
 * file: include/cinfinity/model.hpp
 * 
**************************************/
#pragma once

#ifndef __INCLUDE_CINFINITY_MODEL_HANDLER_HPP__
#define __INCLUDE_CINFINITY_MODEL_HANDLER_HPP__

#include "config.hpp"

namespace cinfinity {
    class ModelHandler {
        // Model Details
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Env m_env;
        Ort::Session m_session;

        // Threading details
        std::thread m_thread;

        // Batching details
        std::queue<chess::Board> m_queue;
        std::mutex m_lock;

        ModelHandler(std::string modelPath);
        ~ModelHandler();

        bool addToQueue(chess::Board& board);

        void run();
    }; // class ModelHandler
} // namespace cinfinity

#endif // __INCLUDE_CINFINITY_MODEL_HANDLER_HPP__