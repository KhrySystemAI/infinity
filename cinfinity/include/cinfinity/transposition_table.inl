#include "transposition_table_new.hpp"

namespace cinfinity {
    TranspositionTable::TranspositionTable() {
        for (auto& bucket : m_buckets) {
            bucket = std::make_unique<Bucket>();
        }
    }

    bool TranspositionTable::batchCreate(std::vector<BucketHashValue>&& entries) {
        bool duplicateFound = false;
        std::array<std::vector<HashValue>, 256> bucketsData;

        for (auto& entry : entries) {
            bucketsData[entry.bucket].emplace_back(entry.hash, std::move(entry.entry));
        }

        for (std::size_t i = 0; i < bucketsData.size(); ++i) {
            auto& vals = bucketsData[i];
            if (vals.empty()) continue;

            auto& bucket = m_buckets[i];
            // We assume multiple writers could access the same block, just in case
            bucket->m_newReadsAllowed.wait(false);
            bucket->m_newReadsAllowed.store(false);
            // Once the lock is aquired, other threads can go back to blocking on the lock aquire rather than the atomic
            std::unique_lock lock(bucket->m_lock);
            bucket->m_newReadsAllowed.store(true);
            bucket->m_newReadsAllowed.notify_all();

            for (auto& v : vals) {
                if (bucket->m_data.contains(v.hash)) {
                    duplicateFound = true;
                }
                bucket->m_data.emplace(v.hash, std::move(v.entry));
            }
        }

        return duplicateFound;
    }

    bool TranspositionTable::batchRead(std::vector<BucketHash>& keys, std::vector<Entry*>& entries) {
        entries.resize(keys.size());
        bool missingFound = false;
        std::array<std::vector<uint64_t>, 256> bucketsKeys;

        for (auto& key : keys) {
            bucketsKeys[key.bucket].push_back(key.hash);
        }

        std::size_t j = 0;
        for (std::size_t i = 0; i < bucketsKeys.size(); ++i) {
            auto& vals = bucketsKeys[i];
            if (vals.empty()) continue;

            auto& bucket = m_buckets[i];
            bucket->m_newReadsAllowed.wait(false);
            std::shared_lock lock(m_buckets[i]->m_lock);

            for (auto hash : vals) {
                auto& bucketData = m_buckets[i]->m_data;
                entries[j] = bucketData.contains(hash) ? bucketData[hash].get() : nullptr;
                if (!entries[j]) missingFound = true;
                ++j;
            }
        }

        return missingFound;
    }

    bool TranspositionTable::batchDelete(std::vector<BucketHash>& keys) {
        bool missingFound = false;
        std::array<std::vector<uint64_t>, 256> bucketsKeys;

        for (auto& key : keys) {
            bucketsKeys[key.bucket].push_back(key.hash);
        }

        for(std::size_t i = 0; i < bucketsKeys.size(); i++) {
            auto& vals = bucketsKeys[i];
            if(vals.empty()) continue;

            std::unique_lock lock(m_buckets[i]->lock);

            for(auto hash : vals) {
                auto& bucketData = m_buckets[i]->m_data;
                if(bucketData.contains(hash))
                    bucketData.erase(hash);
                else
                    missingFound = true;
            }
        }
        return missingFound;
    }
}