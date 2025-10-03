# Infinity

A high-performance C++20 transformer chess engine.

## Description

Infinity is designed as a general architecture based on projects like **Lc0** and **AlphaZero**. It features:

* **C++ core inference** via **ONNX Runtime**
* **Python training core** via **PyTorch** (bindings in progress)
* **Official Python bindings** (planned)
* **Lichess bot frontend**

Infinity is fully open-source under the **GPLv3 license**.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Development](#development)
4. [Testing](#testing)
5. [Contributing](#contributing)
6. [Issue Reporting](#issue-reporting)
7. [License](#license)
8. [Roadmap](#roadmap)

---

## Installation

Infinity uses **Conan** for C++ dependency management.

1. Install [Conan](https://conan.io/) (v2.x recommended)
2. Install a C++20 compiler (GCC ≥ 10, Clang ≥ 12, MSVC ≥ 2019)
3. Clone the repository:

   ```bash
   git clone https://github.com/KhrySystemAI/infinity.git
   cd infinity
   ```
4. Install dependencies via Conan:

   ```bash
   conan install . --profile <your-profile> --build=missing
   ```
5. Build the project with CMake:

   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build .
   ```

---

## Usage

### C++ Core

Example usage:

```cpp
#include <infinity/engine.h>

int main() {
    Infinity::Engine engine;
    engine.initialize();
    engine.makeMove("e2e4");
    return 0;
}
```

### Python Bindings

Bindings are under development. Example usage will be provided when ready.

---

## Development

* **Style Guide:** Google C++ Style Guide
* **Code Review:** Pull requests are required for all contributions
* **CI/CD:** GitHub Actions runs all tests on PRs and merges

### Building

Follow the instructions under [Installation](#installation).

---

## Testing

Infinity includes automated test suites:

* **GTest:** Unit tests for C++ core
* **FuzzTest:** Fuzzing tests for engine robustness

Run tests via:

```bash
cd build
ctest --output-on-failure
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes following the Google C++ style guide
4. Submit a pull request for review
5. Ensure all tests pass in GitHub Actions

---

## Issue Reporting

* General bugs and feature requests: **GitHub Issues**
* Vulnerability reporting: see `SECURITY.md` (planned)

---

## License

Infinity is released under the **GPLv3 license**. See the `LICENSE` file for details.

---

## Roadmap

* Python bindings
* Improved Lichess bot interface
* Detailed architecture documentation
* Release notes and CHANGELOG