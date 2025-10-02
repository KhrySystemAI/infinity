#include <fuzztest/fuzztest.h>
#include <cinfinity/transposition_table.hpp>

using namespace cinfinity;

// A small helper struct describing an operation on the table
struct TableOp {
    enum Type { Insert, Get, Remove, RemoveBytes } type;
    chess::PackedBoard board;
    WDL wdl;
    uint16_t visits;
    uint16_t last_used;
    size_t bytes_to_remove;
};

// Domain generator: produce random TableOp values
static auto TableOpDomain() {
    return fuzztest::Arbitrary<TableOp>()
        .WithFields([](auto& g) {
            // type is uniform from 0–3
            g.template AddField<&TableOp::type>(
                fuzztest::UniformEnum<TableOp::Type>());
            g.template AddField<&TableOp::board>(fuzztest::Arbitrary<chess::PackedBoard>());
            g.template AddField<&TableOp::wdl>(fuzztest::Arbitrary<WDL>());
            g.template AddField<&TableOp::visits>(fuzztest::Arbitrary<uint16_t>());
            g.template AddField<&TableOp::last_used>(fuzztest::Arbitrary<uint16_t>());
            g.template AddField<&TableOp::bytes_to_remove>(fuzztest::Arbitrary<size_t>());
        });
}

// The actual fuzz test: apply many random operations in a sequence
FUZZ_TEST(FuzzTranspositionTable, SequenceOfOps)
    .WithDomains(fuzztest::Truncate(fuzztest::ContainerOf<TableOpDomain>(), 100))
    // Limit the sequence length to e.g. 100 ops
    .WithCorpusSize(64)  // start with some seeds
    (std::vector<TableOp> ops) {
    TranspositionTable table;

    for (auto const &op : ops) {
        switch (op.type) {
            case TableOp::Insert: {
                auto* e = new TranspositionTable::Entry();
                e->value = op.wdl;
                e->visits = op.visits;
                e->last_used = op.last_used;
                // minimal policy: one move
                chess::Move m(1);
                e->policy[m] = 0.5f;
                std::vector<std::pair<chess::PackedBoard, TranspositionTable::Entry*>> batch = {
                    { op.board, e }
                };
                table.insertDispatch(batch);
                break;
            }
            case TableOp::Get: {
                std::vector<chess::PackedBoard> keys = { op.board };
                std::vector<TranspositionTable::Entry*> out;
                table.getDispatch(keys, out);
                break;
            }
            case TableOp::Remove: {
                std::vector<chess::PackedBoard> keys = { op.board };
                table.removeDispatch(keys);
                break;
            }
            case TableOp::RemoveBytes: {
                table.removeBytes(op.bytes_to_remove);
                break;
            }
        }
    }

    // Invariant checks: after sequence, table should not crash, and getDispatch vs contains should agree
    // e.g., for each board, if contains(board) true → getDispatch returns non-null
    // We can pick a few random boards for this
    std::vector<chess::PackedBoard> check_keys;
    for (auto const &op : ops) {
        check_keys.push_back(op.board);
    }
    std::vector<TranspositionTable::Entry*> outs;
    table.getDispatch(check_keys, outs);
    for (size_t i = 0; i < check_keys.size(); i++) {
        bool has = table.contains(check_keys[i]);
        bool got_nonnull = (outs[i] != nullptr);
        EXPECT_EQ(has, got_nonnull);
    }
}
