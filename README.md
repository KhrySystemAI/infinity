# Infinity

A high-performance C++20 transformer chess engine.

## Description

Infinity is designed as a general architecture based on projects like **Lc0** and **AlphaZero**. It
features:

* **C++ inference** via **ONNX Runtime**
* **Python training** via **PyTorch**
* **Lichess bot frontend**

Infinity is fully open-source under the **GPLv3 license**.

---

## Table of Contents

1. [Installation](#installation)

---

## Installation

Infinity uses [Conan](https://conan.io/) for packaging and dependency management, and 
[CMake](https://cmake.org) for its build generator. 

1. Clone the repository

```sh
git clone https://github.com/KhrySystemAI/infinity.git
cd infinity
```

2. Install Conan:

```sh
# You may want to put Conan and its deps into a venv
python -m venv venv
./venv/Scripts/activate

pip install conan
```
3. Build the library. Depending on your compiler, you may need to set the build to C++20.

```sh
conan profile detect default
conan build . 
# Alternatively, force C++20 if the profile didn't load it
conan build . -s compiler.cppstd=20
```