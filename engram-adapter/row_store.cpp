// Exact immutable FP8 engram row retrieval with a bounded direct-mapped RAM
// cache in front of O_DIRECT NVMe reads.
//
// Derived from adapter/row_store.cpp in
// https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000
// Copyright (c) 2026 0xSero, MIT License. Modifications for the KTransformers
// dual-EPYC host (kvcache-ai/ktransformers fork, 05yuki):
//   - never abort inside the callback: failures zero the row, bump an error
//     counter, and are surfaced by the Python side after stream sync;
//   - 64 KiB read staging so one pread covers a row straddling a 4 KiB edge;
//   - stats expose cache slot count and bytes so the budget is auditable.
//
// No CUDA calls in the callback: suitable for cudaLaunchHostFunc graph nodes.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint64_t kDim = 256;       // fp8 e4m3 bytes per row
constexpr uint64_t kScales = 8;      // e8m0 bytes per row (block 32)
constexpr uint64_t kRow = kDim + kScales;
constexpr uint64_t kSlotBytes = kRow + sizeof(uint64_t);
constexpr size_t kPage = 4096;
constexpr int kLockShards = 256;

struct Store {
  int fd = -1;
  uint64_t rows = 0, weight_offset = 0, scale_offset = 0, slots = 0;
  uint64_t row_lo = 0, row_hi = 0;
  uint8_t *cache = nullptr;
  uint64_t *keys = nullptr;
  uint8_t *resident = nullptr;
  size_t resident_size = 0;
  std::mutex locks[kLockShards];
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0}, errors{0};
  // timing: whole callback and the part spent on cache misses, so the share
  // of a decode step that is engram row fetches can be read off
  std::atomic<uint64_t> calls{0}, lookup_ns{0}, miss_ns{0};
  // Misses are NVMe reads at ~170 us each and a decode call has 12 rows, so
  // issuing them one after another costs ~4 ms per call per layer (09-13
  // profile: 8-9 ms of every decode token). A small pool of I/O threads,
  // created at open, runs a call's misses concurrently; the callback thread
  // waits for the batch. Size from DSV41_ENGRAM_IO_THREADS (default 16).
  struct IoPool {
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    std::vector<std::function<void()>> queue;
    size_t next = 0, pending = 0;
    bool stop = false;
  } io;
};

void io_worker(Store *s) {
  for (;;) {
    std::function<void()> job;
    {
      std::unique_lock<std::mutex> lk(s->io.mu);
      s->io.cv.wait(lk, [&] { return s->io.stop || s->io.next < s->io.queue.size(); });
      if (s->io.stop) return;
      job = std::move(s->io.queue[s->io.next++]);
    }
    job();
    std::lock_guard<std::mutex> lk(s->io.mu);
    if (--s->io.pending == 0) {
      s->io.queue.clear();
      s->io.next = 0;
      s->io.done_cv.notify_all();
    }
  }
}

void io_start(Store *s) {
  int n = 16;
  if (const char *e = getenv("DSV41_ENGRAM_IO_THREADS")) n = std::atoi(e);
  if (n < 1) n = 1;
  if (n > 64) n = 64;
  for (int i = 0; i < n; i++) s->io.threads.emplace_back(io_worker, s);
}

void io_stop(Store *s) {
  {
    std::lock_guard<std::mutex> lk(s->io.mu);
    s->io.stop = true;
  }
  s->io.cv.notify_all();
  for (auto &t : s->io.threads) t.join();
  s->io.threads.clear();
}

// Run all jobs on the pool and wait; only one batch is in flight per store
// (the callbacks of one stream are serialised by CUDA).
void io_run_batch(Store *s, std::vector<std::function<void()>> &jobs) {
  if (jobs.empty()) return;
  std::unique_lock<std::mutex> lk(s->io.mu);
  s->io.done_cv.wait(lk, [&] { return s->io.pending == 0; });
  s->io.queue = std::move(jobs);
  s->io.next = 0;
  s->io.pending = s->io.queue.size();
  lk.unlock();
  s->io.cv.notify_all();
  lk.lock();
  s->io.done_cv.wait(lk, [&] { return s->io.pending == 0; });
}

struct Work {
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;
  uint64_t count;
};

// Returns false on a short or failed read. Never exits the process.
bool read_bytes(Store *s, uint64_t offset, uint8_t *out, size_t length) {
  if (s->resident) {
    std::memcpy(out, s->resident + offset, length);
    return true;
  }
  alignas(kPage) static thread_local uint8_t page[16 * kPage];
  const uint64_t base = offset & ~uint64_t(kPage - 1);
  const size_t delta = offset - base;
  const size_t requested = ((delta + length + kPage - 1) / kPage) * kPage;
  if (requested > sizeof(page)) return false;
  ssize_t got;
  do {
    got = pread(s->fd, page, requested, base);
  } while (got < 0 && errno == EINTR);
  if (got < 0 || size_t(got) < delta + length) return false;
  std::memcpy(out, page + delta, length);
  s->reads.fetch_add(1, std::memory_order_relaxed);
  return true;
}

}  // namespace

extern "C" Store *row_store_open(const char *path, uint64_t rows, uint64_t woff,
                                 uint64_t soff, uint64_t budget, int resident_mode) {
  auto *s = new Store;
  s->fd = open(path, O_RDONLY | O_CLOEXEC | (resident_mode ? 0 : O_DIRECT));
  if (s->fd < 0) {
    std::fprintf(stderr, "row_store_open: open(%s) failed errno=%d\n", path, errno);
    delete s;
    return nullptr;
  }
  struct stat st;
  if (fstat(s->fd, &st) || woff > uint64_t(st.st_size) || soff > uint64_t(st.st_size) ||
      rows > (uint64_t(st.st_size) - woff) / kDim || rows > (uint64_t(st.st_size) - soff) / kScales) {
    std::fprintf(stderr, "row_store_open: table extent invalid for %s\n", path);
    close(s->fd);
    delete s;
    return nullptr;
  }
  if (resident_mode) {
    s->resident_size = size_t(st.st_size);
    void *m = mmap(nullptr, s->resident_size, PROT_READ, MAP_SHARED | MAP_POPULATE, s->fd, 0);
    if (m == MAP_FAILED) {
      std::fprintf(stderr, "row_store_open: resident mmap failed errno=%d\n", errno);
      close(s->fd);
      delete s;
      return nullptr;
    }
    s->resident = static_cast<uint8_t *>(m);
    if (mlock(s->resident, s->resident_size)) {
      std::fprintf(stderr, "row_store_open: mlock failed errno=%d (need RLIMIT_MEMLOCK)\n", errno);
      munmap(s->resident, s->resident_size);
      close(s->fd);
      delete s;
      return nullptr;
    }
    budget = 0;
  }
  s->rows = rows;
  s->weight_offset = woff;
  s->scale_offset = soff;
  s->row_lo = 0;
  s->row_hi = rows;
  s->slots = budget / kSlotBytes;
  if (s->slots) {
    void *c = mmap(nullptr, s->slots * kRow, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void *k = mmap(nullptr, s->slots * sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (c == MAP_FAILED || k == MAP_FAILED) {
      std::fprintf(stderr, "row_store_open: cache allocation failed errno=%d\n", errno);
      if (c != MAP_FAILED) munmap(c, s->slots * kRow);
      if (k != MAP_FAILED) munmap(k, s->slots * sizeof(uint64_t));
      close(s->fd);
      delete s;
      return nullptr;
    }
    s->cache = static_cast<uint8_t *>(c);
    s->keys = static_cast<uint64_t *>(k);
    madvise(s->cache, s->slots * kRow, MADV_HUGEPAGE);
  }
  if (!s->resident) io_start(s);
  return s;
}

extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  const auto t_call = std::chrono::steady_clock::now();
  // pass 1: cache hits and invalid ids are served in place; misses are
  // collected, then read concurrently, then inserted into the cache
  struct Miss { uint64_t i; int64_t id; bool ok; uint8_t row[kRow]; };
  std::vector<Miss> misses;
  misses.reserve(work->count);
  for (uint64_t i = 0; i < work->count; ++i) {
    const int64_t id = work->ids[i];
    uint8_t *w_out = work->weights + i * kDim;
    uint8_t *s_out = work->scales + i * kScales;
    if (id < 0 || uint64_t(id) >= s->rows) {
      std::memset(w_out, 0, kDim);
      std::memset(s_out, 0, kScales);
      s->errors.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(w_out, 0, kDim);
      std::memset(s_out, 0, kScales);
      continue;
    }
    const uint64_t slot = s->slots ? uint64_t(id) % s->slots : 0;
    bool hit = false;
    if (s->slots) {
      std::lock_guard<std::mutex> guard(s->locks[slot % kLockShards]);
      if (s->keys[slot] == uint64_t(id) + 1) {
        std::memcpy(w_out, s->cache + slot * kRow, kDim);
        std::memcpy(s_out, s->cache + slot * kRow + kDim, kScales);
        hit = true;
      }
    }
    if (hit) {
      s->hits.fetch_add(1, std::memory_order_relaxed);
    } else {
      misses.push_back(Miss{i, id, false, {}});
    }
  }
  if (!misses.empty()) {
    const auto t_miss = std::chrono::steady_clock::now();
    if (s->resident || misses.size() == 1 || s->io.threads.empty()) {
      for (auto &m : misses)
        m.ok = read_bytes(s, s->weight_offset + uint64_t(m.id) * kDim, m.row, kDim) &&
               read_bytes(s, s->scale_offset + uint64_t(m.id) * kScales, m.row + kDim, kScales);
    } else {
      std::vector<std::function<void()>> jobs;
      jobs.reserve(misses.size());
      for (auto &m : misses)
        jobs.emplace_back([s, &m] {
          m.ok = read_bytes(s, s->weight_offset + uint64_t(m.id) * kDim, m.row, kDim) &&
                 read_bytes(s, s->scale_offset + uint64_t(m.id) * kScales, m.row + kDim, kScales);
        });
      io_run_batch(s, jobs);
    }
    s->miss_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_miss).count(),
                         std::memory_order_relaxed);
    for (auto &m : misses) {
      uint8_t *w_out = work->weights + m.i * kDim;
      uint8_t *s_out = work->scales + m.i * kScales;
      if (!m.ok) {
        std::memset(w_out, 0, kDim);
        std::memset(s_out, 0, kScales);
        s->errors.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      s->misses.fetch_add(1, std::memory_order_relaxed);
      if (s->slots) {
        const uint64_t slot = uint64_t(m.id) % s->slots;
        std::lock_guard<std::mutex> guard(s->locks[slot % kLockShards]);
        std::memcpy(s->cache + slot * kRow, m.row, kRow);
        s->keys[slot] = uint64_t(m.id) + 1;
      }
      std::memcpy(w_out, m.row, kDim);
      std::memcpy(s_out, m.row + kDim, kScales);
    }
  }
  s->calls.fetch_add(1, std::memory_order_relaxed);
  s->lookup_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_call).count(),
                         std::memory_order_relaxed);
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load();
  out[1] = s->misses.load();
  out[2] = s->reads.load();
  out[3] = s->slots * kSlotBytes;
  out[4] = s->errors.load();
  out[5] = s->slots;
  out[6] = s->calls.load();
  out[7] = s->lookup_ns.load();
  out[8] = s->miss_ns.load();
}

extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) {
    std::fprintf(stderr, "row_store_range: invalid [%llu,%llu) for %llu rows\n",
                 (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)s->rows);
    return;
  }
  s->row_lo = lo;
  s->row_hi = hi;
}

extern "C" void row_store_close(Store *s) {
  if (!s) return;
  io_stop(s);
  if (s->resident) munmap(s->resident, s->resident_size);
  if (s->slots) {
    munmap(s->cache, s->slots * kRow);
    munmap(s->keys, s->slots * sizeof(uint64_t));
  }
  if (s->fd >= 0) close(s->fd);
  delete s;
}
