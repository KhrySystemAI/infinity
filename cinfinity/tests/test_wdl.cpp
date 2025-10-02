#include <gtest/gtest.h>
#include <cinfinity/wdl.hpp>

using namespace cinfinity;

TEST(WDLTest, BasicAccessors) {
    WDL wdl(0.5f, 0.2f, 0.3f); // win=0.5, draw=0.2, loss=0.3
    EXPECT_FLOAT_EQ(wdl.win_chance(), 0.5f);
    EXPECT_FLOAT_EQ(wdl.loss_chance(), 0.3f);
    EXPECT_FLOAT_EQ(wdl.draw_chance(), 0.2f);
    EXPECT_EQ(wdl.size(), 2u); // underlying array has 2 elements
}

TEST(WDLTest, ProbabilitiesSumToOne) {
    WDL wdl(0.7f, 0.1f, 0.2f);
    float sum = wdl.win_chance() + wdl.loss_chance() + wdl.draw_chance();
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(WDLTest, GreedyMode) {
    WDL wdl(0.6f, 0.1f, 0.3f);
    EXPECT_FLOAT_EQ(wdl.score(GREEDY), 0.6f);
}

TEST(WDLTest, BalancedMode) {
    WDL wdl(0.6f, 0.1f, 0.3f);
    float expected = 0.6f + 0.1f / 2.0f; // 0.65
    EXPECT_FLOAT_EQ(wdl.score(BALANCED), expected);
}

TEST(WDLTest, SolidMode) {
    WDL wdl(0.6f, 0.1f, 0.3f);
    float expected = (1.0f - 0.3f) + (0.1f / 2.0f); // 0.75
    EXPECT_FLOAT_EQ(wdl.score(SOLID), expected);
}

TEST(WDLTest, SafeMode) {
    WDL wdl(0.6f, 0.1f, 0.3f);
    float expected = 1.0f - 0.3f; // 0.7
    EXPECT_FLOAT_EQ(wdl.score(SAFE), expected);
}

TEST(WDLTest, DegenerateCases) {
    WDL allWin(1.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(allWin.draw_chance(), 0.0f);
    EXPECT_FLOAT_EQ(allWin.score(GREEDY), 1.0f);

    WDL allLoss(0.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(allLoss.draw_chance(), 0.0f);
    EXPECT_FLOAT_EQ(allLoss.score(SAFE), 0.0f);

    WDL allDraw(0.0f, 1.0f, 0.0f);
    EXPECT_FLOAT_EQ(allDraw.draw_chance(), 1.0f);
    EXPECT_FLOAT_EQ(allDraw.score(BALANCED), 0.5f);
}