#ifndef INCLUDE_CINFINITY_CHESS_HPP
#define INCLUDE_CINFINITY_CHESS_HPP

#include <cstdint>
#include <type_traits>

namespace cinfinity::chess {
enum Piece : uint8_t {
  NONE = 0,
  PAWN = 1,
  KNIGHT = 2,
  BISHOP = 3,
  ROOK = 4,
  QUEEN = 5,
  KING = 6,
};

namespace filemajor {
enum Square : uint8_t {
  A1 = 0,
  B1 = 1,
  C1 = 2,
  D1 = 3,
  E1 = 4,
  F1 = 5,
  G1 = 6,
  H1 = 7,
  A2 = 8,
  B2 = 9,
  C2 = 10,
  D2 = 11,
  E2 = 12,
  F2 = 13,
  G2 = 14,
  H2 = 15,
  A3 = 16,
  B3 = 17,
  C3 = 18,
  D3 = 19,
  E3 = 20,
  F3 = 21,
  G3 = 22,
  H3 = 23,
  A4 = 24,
  B4 = 25,
  C4 = 26,
  D4 = 27,
  E4 = 28,
  F4 = 29,
  G4 = 30,
  H4 = 31,
  A5 = 32,
  B5 = 33,
  C5 = 34,
  D5 = 35,
  E5 = 36,
  F5 = 37,
  G5 = 38,
  H5 = 39,
  A6 = 40,
  B6 = 41,
  C6 = 42,
  D6 = 43,
  E6 = 44,
  F6 = 45,
  G6 = 46,
  H6 = 47,
  A7 = 48,
  B7 = 49,
  C7 = 50,
  D7 = 51,
  E7 = 52,
  F7 = 53,
  G7 = 54,
  H7 = 55,
  A8 = 56,
  B8 = 57,
  C8 = 58,
  D8 = 59,
  E8 = 60,
  F8 = 61,
  G8 = 62,
  H8 = 63
};

}

namespace rowmajor {
enum Square : uint8_t {
  A1 = 0,
  A2 = 1,
  A3 = 2,
  A4 = 3,
  A5 = 4,
  A6 = 5,
  A7 = 6,
  A8 = 7,
  B1 = 8,
  B2 = 9,
  B3 = 10,
  B4 = 11,
  B5 = 12,
  B6 = 13,
  B7 = 14,
  B8 = 15,
  C1 = 16,
  C2 = 17,
  C3 = 18,
  C4 = 19,
  C5 = 20,
  C6 = 21,
  C7 = 22,
  C8 = 23,
  D1 = 24,
  D2 = 25,
  D3 = 26,
  D4 = 27,
  D5 = 28,
  D6 = 29,
  D7 = 30,
  D8 = 31,
  E1 = 32,
  E2 = 33,
  E3 = 34,
  E4 = 35,
  E5 = 36,
  E6 = 37,
  E7 = 38,
  E8 = 39,
  F1 = 40,
  F2 = 41,
  F3 = 42,
  F4 = 43,
  F5 = 44,
  F6 = 45,
  F7 = 46,
  F8 = 47,
  G1 = 48,
  G2 = 49,
  G3 = 50,
  G4 = 51,
  G5 = 52,
  G6 = 53,
  G7 = 54,
  G8 = 55,
  H1 = 56,
  H2 = 57,
  H3 = 58,
  H4 = 59,
  H5 = 60,
  H6 = 61,
  H7 = 62,
  H8 = 63
};
}

template <typename T>
concept SquareType =
    std::is_same_v<T, filemajor::Square> || std::is_same_v<T, rowmajor::Square>;

class Move {
 public:
  Move(filemajor::Square from, filemajor::Square to, Piece promotion);
  Move(rowmajor::Square from, rowmajor::Square to, Piece promotion);
  Move(uint16_t move);

  template <SquareType T>
  [[nodiscard]] auto fromSquare() const noexcept -> T;
  template <SquareType T>
  [[nodiscard]] auto toSquare() const noexcept -> T;
  [[nodiscard]] auto promotion() const noexcept -> Piece;

 private:
  uint16_t data;
};
}  // namespace cinfinity::chess

#endif  // INCLUDE_CINFINITY_CHESS_HPP