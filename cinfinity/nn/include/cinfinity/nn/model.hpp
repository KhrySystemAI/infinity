#ifndef INCLUDE_CINFINITY_NN_MODEL_HPP
#define INCLUDE_CINFINITY_NN_MODEL_HPP

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

#endif // INCLUDE_CINFINITY_NN_MODEL_HPP