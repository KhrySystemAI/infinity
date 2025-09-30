#include <cinfinity/transposition_table.hpp>

using namespace cinfinity;

bool TranspositionTable::contains(chess::PackedBoard& key) {
    return m_table.contains(key);
}

void TranspositionTable::getDispatch(std::vector<chess::PackedBoard>& keys, std::vector<TranspositionTable::Entry*>& entries) {
    entries.resize(keys.size());
    int i = 0;
    for(auto& k : keys) {
        if(contains(k)) entries[i] = m_table[k];
        else entries[i] = nullptr;
        i++;
    }
}

bool TranspositionTable::getSubmap(
    uint16_t idx, uint16_t count, 
    std::vector<std::pair<chess::PackedBoard, TranspositionTable::Entry*>>& result
) {
    size_t total = m_table.size();
    size_t chunk = total / count;
    size_t remainder = total % count;

    // distribute remainder: first 'remainder' chunks get an extra element
    size_t start = idx * chunk + std::min<size_t>(idx, remainder);
    size_t length = chunk + (idx < remainder ? 1 : 0);
    size_t end = std::min(start + length, total);

    if (start >= end) {
        return false; // this partition has no elements
    }

    // copy [start, end)
    result.reserve(end - start);
    auto it = m_table.begin();
    std::advance(it, start);
    for (size_t i = start; i < end; ++i, ++it) {
        result.emplace_back(it->first, it->second);
    }

    return true;
}

bool TranspositionTable::insertDispatch(std::vector<std::pair<chess::PackedBoard, TranspositionTable::Entry*>>& values) {
    bool result = false;
    for(auto v : values) {
        if(contains(v.first)) result = true;
        else m_table[v.first] = v.second;
    }
    return result;
}

bool TranspositionTable::removeDispatch(std::vector<chess::PackedBoard>& keys) {
    bool result = false;
    for(auto k : keys) {
        if(contains(k)) m_table.erase(k);
        else result = true;
    }
    return result;
}