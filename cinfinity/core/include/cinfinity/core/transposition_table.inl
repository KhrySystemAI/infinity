#pragma once

#ifndef __INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_INL__
#define __INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_INL__

#include "transposition_table.hpp"

namespace cinfinity::core {
    bool TranspositionTable::batchCreate(std::vector<std::tuple<uint8_t, uint64_t, std::unique_ptr<TranspositionTable::Entry>>> entries) {
        std::array<std::vector<std::tuple<uint64_t, std::unique_ptr<Entry>>>, 256> bucketsData;
        bool duplicateFound = false;
        
        for(auto& [bucket_id, hash, entry] : entries) {
            bucketsData[bucket_id].emplace_back(hash, std::move(entry));
        }

        for(size_t i = 0; i < bucketsData.size(); i++) {
            auto& vals = bucketsData[i];
            if(vals.empty()) continue;

            auto& bucket = m_buckets[i];
            absl::WriterMutexLock lock(&bucket->m_lock);

            for(auto& [hash, entry] : vals) {
                if(bucket->m_data.contains(hash)) {
                    duplicateFound = true;
                }
                bucket->m_data.emplace(hash, std::move(entry));
            }
        }

        return duplicateFound;
    }

    TranspositionTable::Entry* TranspositionTable::read(std::tuple<uint8_t, uint64_t> key) {
        auto& [bucket_id, hash] = key;
        auto& bucket = m_buckets[bucket_id];
        absl::ReaderMutexLock lock(&bucket->m_lock);
        return bucket->m_data.contains(hash) ? bucket->m_data[hash].get() : nullptr;
    }

    bool TranspositionTable::batchDelete(std::vector<std::tuple<uint8_t, uint64_t>> key) {

    }

    bool TranspositionTable::bytesDelete(size_t bytesToRemove) {
        size_t bytesRemoved = 0;

        for (std::size_t i = 0; i < m_buckets.size(); ++i) {
            auto& bucket = m_buckets[i];

            std::unique_lock lock(bucket->m_lock);

            auto it = bucket->m_data.begin();
            while (it != bucket->m_data.end() && bytesRemoved < bytesToRemove) {
                Entry* entry = it->second.get();
                
                // Only evict old entries
                if (entry->last_used < m_currentGeneration) {
                    size_t entryBytes = sizeof(*entry) + sizeof(it->second);
                    entryBytes += entry->policy.size() * sizeof(std::pair<uint16_t,float>);

                    bytesRemoved += entryBytes;

                    bucket->m_data.erase(it);
                } 
                ++it;
            }

            if (bytesRemoved >= bytesToRemove) {
                return true;
            }
        }

        return false;
    }

} // namespace cinfinity::core

#endif // __INCLUDE_CINFINITY_CORE_TRANSPOSITION_TABLE_INL__