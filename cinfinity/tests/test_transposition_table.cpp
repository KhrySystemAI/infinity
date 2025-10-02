#include <gtest/gtest.h>
#include <cinfinity/transposition_table.hpp>

using namespace cinfinity;

// Helper: make a simple Entry with one policy move
std::unique_ptr<TranspositionTable::Entry> makeEntry(float w, float d, float l, uint16_t visits, uint16_t gen) {
    auto* e = new TranspositionTable::Entry{{}, WDL(w, d, l), visits, gen};
    chess::Move m(static_cast<uint16_t>(1));
    e->policy[m] = 0.5f;
    auto ue = std::unique_ptr<TranspositionTable::Entry>(e);

    return ue;
}

chess::PackedBoard makePacked() {
    return chess::Board::Compact::encode(chess::Board());
}

TEST(TranspositionTableTest, InsertAndContains) {
    TranspositionTable table;
    chess::PackedBoard b = makePacked();

    auto e = makeEntry(0.5f, 0.3f, 0.2f, 10, 1);
    std::vector<std::pair<chess::PackedBoard, std::unique_ptr<TranspositionTable::Entry>>> ins;
    ins.emplace_back(b, std::move(e));
    EXPECT_FALSE(table.insertDispatch(ins)); // should succeed (no collision)

    EXPECT_TRUE(table.contains(b));
}

TEST(TranspositionTableTest, DuplicateInsertFails) {
    TranspositionTable table;
    chess::PackedBoard b = makePacked();

    auto e1 = makeEntry(0.5f, 0.3f, 0.2f, 5, 1);
    auto e2 = makeEntry(0.4f, 0.3f, 0.3f, 6, 2);

    std::vector<std::pair<chess::PackedBoard, std::unique_ptr<TranspositionTable::Entry>>> batch1;
    batch1.emplace_back(b, std::move(e1));
    std::vector<std::pair<chess::PackedBoard, std::unique_ptr<TranspositionTable::Entry>>> batch2;
    batch2.emplace_back(b, std::move(e2));

    EXPECT_FALSE(table.insertDispatch(batch1));   // first insert
    EXPECT_TRUE(table.insertDispatch(batch2));    // collision → returns true
}

TEST(TranspositionTableTest, GetDispatchReturnsEntries) {
    TranspositionTable table;
    chess::PackedBoard b = makePacked();

    auto e = makeEntry(0.7f, 0.1f, 0.2f, 4, 1);
    std::vector<std::pair<chess::PackedBoard, std::unique_ptr<TranspositionTable::Entry>>> ins;
    ins.emplace_back(b, std::move(e));
    table.insertDispatch(ins);

    std::vector<chess::PackedBoard> keys = {b};
    std::vector<TranspositionTable::Entry*> out;
    table.getDispatch(keys, out);

    ASSERT_EQ(out.size(), 1u);
    ASSERT_NE(out[0], nullptr);
    EXPECT_FLOAT_EQ(out[0]->value.win_chance(), 0.7f);
}

TEST(TranspositionTableTest, GetDispatchReturnsNullptrForMissing) {
    TranspositionTable table;
    chess::PackedBoard b = makePacked();

    std::vector<chess::PackedBoard> keys = {b};
    std::vector<TranspositionTable::Entry*> out;
    table.getDispatch(keys, out);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], nullptr);
}

TEST(TranspositionTableTest, RemoveDispatchRemovesEntry) {
    TranspositionTable table;
    chess::PackedBoard b = makePacked();

    auto e = makeEntry(0.5f, 0.25f, 0.25f, 1, 1);
    std::vector<std::pair<chess::PackedBoard, std::unique_ptr<TranspositionTable::Entry>>> ins;
    ins.emplace_back(b, std::move(e));
    table.insertDispatch(ins);

    std::vector<chess::PackedBoard> keys = {b};
    EXPECT_TRUE(table.removeDispatch(keys));

    EXPECT_FALSE(table.contains(b));
}

TEST(TranspositionTableTest, RemoveDispatchOnMissingReturnsFalse) {
    TranspositionTable table;
    chess::PackedBoard b = makePacked();

    std::vector<chess::PackedBoard> keys = {b};
    EXPECT_FALSE(table.removeDispatch(keys));
}

TEST(TranspositionTableTest, RemoveBytesEvictsOldEntries) {
    TranspositionTable table;

    // Insert two entries with different generations
    chess::Board board1;
    board1.setFen("startpos");
    chess::PackedBoard b1 = chess::Board::Compact::encode(board1);

    chess::Board board2;
    board2.setFen("rnbqkbnr/1ppppppp/p7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2");
    chess::PackedBoard b2 = chess::Board::Compact::encode(board2);
    auto e1 = makeEntry(0.5f, 0.2f, 0.3f, 1, 0);
    auto e2 = makeEntry(0.4f, 0.3f, 0.3f, 2, 1000);

    std::vector<std::pair<chess::PackedBoard, std::unique_ptr<TranspositionTable::Entry>>> batch;
    batch.emplace_back(b1, std::move(e1));
    batch.emplace_back(b2, std::move(e2));
    table.insertDispatch(batch);

    // Force eviction, should be false because did not remove required bytes
    EXPECT_FALSE(table.removeBytes(1 << 10));

    // Old entry (gen=1) should be gone, new one (gen=1000) should remain
    EXPECT_FALSE(table.contains(b1));
    EXPECT_TRUE(table.contains(b2));
}
