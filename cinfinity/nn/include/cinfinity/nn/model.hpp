#pragma once

#ifndef __INCLUDE_CINFINITY_NN_MODEL_HPP__
#define __INCLUDE_CINFINITY_NN_MODEL_HPP__

#include <thread>
#include <onnxruntime_cxx_api.h>

namespace cinfinity::nn {
    class Model {
        Ort::MemoryInfo m_memoryInfo;
        Ort::Env m_env;
        Ort::Session m_session;

        std::thread m_thread;
    };
}

#endif // __INCLUDE_CINFINITY_NN_MODEL_HPP__