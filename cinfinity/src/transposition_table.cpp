#include <cinfinity/transposition_table.hpp>

using namespace cinfinity;

bool TranspositionTable::contains(chess::PackedBoard& key) {
    return m_table.contains(key);
}

void TranspositionTable::getDispatch(std::vector<chess::PackedBoard>& keys, std::vector<TranspositionTable::Entry*>& entries) {
    entries.resize(keys.size());
    int i = 0;
    for(auto& k : keys) {
        if(contains(k)) entries[i] = m_table[k].get();
        else entries[i] = nullptr;
        i++;
    }
}

bool TranspositionTable::insertDispatch(
    std::vector<std::pair<chess::PackedBoard, std::unique_ptr<Entry>>>& values) 
{
    bool result = false;
    for (auto& v : values) {
        auto& key = v.first;
        auto& entry_ptr = v.second;

        if (contains(key)) {
            result = true; // key already existed
        } else {
            m_table.emplace(key, std::move(entry_ptr));
            // entry_ptr is now null, ownership fully transferred
        }
    }
    return result;
}

bool TranspositionTable::removeDispatch(std::vector<chess::PackedBoard>& keys) {
    bool result = true;
    for(auto k : keys) {
        if(contains(k)) m_table.erase(k);
        else result = false;
    }
    return result;
}

bool TranspositionTable::removeBytes(size_t bytes) {
    size_t removed = 0;
    for (auto it = m_table.begin(); it != m_table.end();) {
        if (it->second->last_used < current_gen) {
            removed += it->second->getBytes();
            it = m_table.erase(it); // erase returns next iterator
        } else {
            ++it;
        }
        if (removed >= bytes) break;
    }
    return removed >= bytes;
}