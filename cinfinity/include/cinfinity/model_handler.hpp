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
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::createCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Env m_env;
        Ort::Session m_session;

        ModelHandler(std::string modelPath);

        void run();
    }; // class ModelHandler
} // namespace cinfinity

#endif // __INCLUDE_CINFINITY_MODEL_HANDLER_HPP__