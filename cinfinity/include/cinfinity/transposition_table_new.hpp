#pragma once

#include "config.hpp"
#include "wdl.hpp"
#include "write_pref_lock.hpp"

namespace cinfinity {
    
    struct TranspositionTable {
        struct Entry {
            hashing::map<uint16_t, float> policy;
            WDL wdl;
            uint16_t visits;
            uint16_t generation;
        };

        class Bucket {
            hashing::map<uint64_t, std::unique_ptr<Entry>> m_data;
            WritePrefLock m_lock;

            friend class TranspositionTable;
        };

        struct BucketHashValue {
            uint8_t bucket;
            uint64_t hash;
            std::unique_ptr<Entry> entry;
        };

        struct HashValue {
            uint64_t hash; 
            std::unique_ptr<Entry> entry;

            
            inline HashValue(uint64_t h, std::unique_ptr<Entry>&& e)
                : hash(h), entry(std::move(e)) {}
        };

        struct BucketHash {
            uint8_t bucket;
            uint64_t hash;
        };

        TranspositionTable();

        bool batchCreate(std::vector<BucketHashValue>&& entries);
        bool batchRead(std::vector<BucketHash>& keys, std::vector<Entry*>& entries);
        bool batchDelete(std::vector<BucketHash>& keys);
        bool bytesDelete(size_t bytes);

        private:
            std::array<std::unique_ptr<Bucket>, 256> m_buckets;
    };
}