"""Read a layer's expert weights into the page cache with parallel preads
before kt-kernel copies them (KT_LOAD_PREFETCH=<threads>, 0 = off).

kt-kernel's load_weights hands the C++ side zero-copy views on safetensors'
private mmap and copies each expert into the memfd arenas with memcpy. The
disk reads are the page faults those memcpys take, one readahead window at a
time (128 KiB on one of the host's NVMe drives, 1 MiB on the other), so the drive
runs at about 1 GB/s during a launch when it can do 5-6 GB/s. madvise /
fadvise WILLNEED did not help (10-07, 141 vs 142 s for Vision's 43 layers):
the kernel clamps each call to the same readahead window.

With KT_LOAD_PREFETCH=N, load_experts resolves every weight tensor's byte
range from the safetensors headers, merges neighbouring ranges, and N threads
pread them in 16 MiB pieces into a throwaway buffer before the C++ copy
starts. The copy then finds the pages in the cache. While a layer copies, the next
layer's read runs in the background; the previous layer's ranges are evicted
with fadvise(DONTNEED), so the cache holds at most two layers. With this on,
the sglang-side per-layer drop of the whole checkpoint must be skipped after
layer 0 (patches/sglang-kt-drop-cache-skip-prefetch.py) or it evicts the
background read. Off by default.

Usage: python kt-kernel-load-prefetch.py <kt_kernel/utils dir or site-packages>
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <kt_kernel/utils dir>")
d = Path(sys.argv[1])
if (d / "kt_kernel").is_dir():
    d = d / "kt_kernel" / "utils"
F = d / "loader.py"
s = F.read_text()
if "KT_LOAD_PREFETCH" in s:
    print(f"already patched {F}")
    raise SystemExit(0)


def once(old, new, text):
    if text.count(old) != 1:
        raise SystemExit("anchor found %d times:\n%s" % (text.count(old), old[:200]))
    return text.replace(old, new)


HELPER = '''

_PREFETCH_THREADS = int(os.environ.get("KT_LOAD_PREFETCH", "0") or 0)
_PREFETCH_PIECE = int(os.environ.get("KT_LOAD_PREFETCH_PIECE_MB", "16")) << 20
# the last TAIL layers are read after their arena exists, not ahead of it: with
# the next layer's pages in the cache the last arenas came out on 4 KB pages
# and rank 1 took 70-200 s to register them (10-07)
_PREFETCH_TAIL = int(os.environ.get("KT_LOAD_PREFETCH_TAIL", "4"))
_header_cache: dict = {}
_pending: dict = {}      # expert prefix -> Future of its prefetch, started during the previous layer
_executor = None


def _st_header(path):
    """{key: (abs_offset, nbytes)} for one safetensors file."""
    h = _header_cache.get(path)
    if h is None:
        import json
        import struct

        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            meta = json.loads(fh.read(n))
        h = {}
        for k, v in meta.items():
            if k == "__metadata__":
                continue
            b, e = v["data_offsets"]
            h[k] = (8 + n + b, e - b)
        _header_cache[path] = h
    return h


def _prefetch_ranges(ranges):
    """ranges: [(path, offset, nbytes)]. Merge neighbours per file, cut into
    pieces, pread them on a thread pool so the drive sees a deep queue. The
    bytes are discarded; the page cache keeps them for the copy."""
    if _PREFETCH_THREADS <= 0 or not ranges:
        return
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    by_file: dict = {}
    for path, off, n in ranges:
        by_file.setdefault(path, []).append((off, n))
    pieces = []
    for path, rs in by_file.items():
        rs.sort()
        cur_off, cur_end = rs[0][0], rs[0][0] + rs[0][1]
        merged = []
        for off, n in rs[1:]:
            if off <= cur_end + (1 << 20):
                cur_end = max(cur_end, off + n)
            else:
                merged.append((cur_off, cur_end))
                cur_off, cur_end = off, off + n
        merged.append((cur_off, cur_end))
        for a, b in merged:
            p = a
            while p < b:
                pieces.append((path, p, min(_PREFETCH_PIECE, b - p)))
                p += _PREFETCH_PIECE
    total = sum(n for _, _, n in pieces)
    local = threading.local()
    fds: dict = {}
    lock = threading.Lock()

    def fd_for(path):
        with lock:
            fd = fds.get(path)
            if fd is None:
                fd = fds[path] = os.open(path, os.O_RDONLY)
            return fd

    def read(piece):
        path, off, n = piece
        buf = getattr(local, "buf", None)
        if buf is None or len(buf) < n:
            buf = local.buf = bytearray(max(n, _PREFETCH_PIECE))
        mv = memoryview(buf)[:n]
        fd = fd_for(path)
        got = 0
        while got < n:
            r = os.preadv(fd, [mv[got:]], off + got)
            if r <= 0:
                break
            got += r
        return got

    with ThreadPoolExecutor(_PREFETCH_THREADS) as ex:
        done = sum(ex.map(read, pieces))
    for fd in fds.values():
        os.close(fd)
    dt = time.time() - t0
    print(f"[kt-kernel] prefetch {done / 2**30:.2f} GiB in {dt:.2f} s "
          f"({done / 2**30 / max(dt, 1e-6):.2f} GiB/s, {len(pieces)} pieces, "
          f"{_PREFETCH_THREADS} threads)", flush=True)
    return done


def _evict_ranges(ranges):
    """posix_fadvise(DONTNEED) over the ranges a finished layer read, so the
    cache holds at most the layer being copied and the one being prefetched."""
    by_file: dict = {}
    for path, off, n in ranges:
        by_file.setdefault(path, []).append((off, n))
    for path, rs in by_file.items():
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            for off, n in rs:
                os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _next_prefix(prefix):
    """layers.5.ffn.experts -> layers.6.ffn.experts (any 'layers.<n>.' form)."""
    import re

    m = re.search(r"(layers\\.)(\\d+)(\\.)", prefix)
    if not m:
        return None
    return prefix[: m.start(2)] + str(int(m.group(2)) + 1) + prefix[m.end(2):]
'''

s = once(
    "from gguf.gguf_reader import GGUFReader\n",
    "from gguf.gguf_reader import GGUFReader\n" + HELPER,
    s,
)

# keep the full path of each shard; file_handle_map is keyed by basename
s = once(
    """                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
""",
    """                    file_path = os.path.join(root, file)
                    self.file_path_map[file] = file_path
                    if file not in self.file_handle_map:
""",
    s,
)
s = once(
    """        self.file_handle_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

        found_safetensor = False
""",
    """        self.file_handle_map = {}
        self.file_path_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

        found_safetensor = False
""",
    s,
)

# the method every loader can call with the keys it is about to hand over
s = once(
    """    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)
""",
    """    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)

    def _ranges_for(self, keys):
        ranges = []
        for k in keys:
            file = self.tensor_file_map.get(k)
            path = getattr(self, "file_path_map", {}).get(file)
            if path is None:
                continue
            ent = _st_header(path).get(k)
            if ent is not None:
                ranges.append((path, ent[0], ent[1]))
        return ranges

    def prefetch_keys(self, keys):
        \"\"\"Pull these tensors' bytes into the page cache with parallel preads
        (KT_LOAD_PREFETCH=<threads>); a no-op when off.\"\"\"
        if _PREFETCH_THREADS <= 0:
            return
        _prefetch_ranges(self._ranges_for(keys))

    def prefetch_layer(self, prefix, suffixes, expert_count, proj_names):
        \"\"\"Called by load_experts before it collects the views. Evicts the
        previous layer's bytes, makes sure this layer's are in the cache
        (waiting on the read started during the previous layer, or reading
        now), and starts the next layer's read in the background.\"\"\"
        global _executor
        if _PREFETCH_THREADS <= 0:
            return
        import time
        from concurrent.futures import ThreadPoolExecutor

        def keys_of(pfx, count):
            return [f"{pfx}.{e}.{p}.{sfx}" for e in range(count) for p in proj_names for sfx in suffixes]

        # the previous layer: prefix with the layer number one lower
        import re

        m = re.search(r"(layers\\.)(\\d+)(\\.)", prefix)
        if m and int(m.group(2)) > 0:
            prev = prefix[: m.start(2)] + str(int(m.group(2)) - 1) + prefix[m.end(2):]
            _evict_ranges(self._ranges_for(keys_of(prev, expert_count)))

        fut = _pending.pop(prefix, None)
        if fut is not None:
            t0 = time.time()
            fut.result()
            print(f"[kt-kernel] prefetch of {prefix} was ready after {time.time() - t0:.2f} s wait", flush=True)
        else:
            _prefetch_ranges(self._ranges_for(keys_of(prefix, expert_count)))

        nxt = _next_prefix(prefix)
        if nxt and _PREFETCH_TAIL > 0:
            # how many layers remain after this one
            m2 = re.search(r"(layers\\.)(\\d+)(\\.)", prefix)
            last = int(m2.group(2))
            while f"{prefix[: m2.start(2)]}{last + 1}{prefix[m2.end(2):]}.0.{proj_names[0]}.weight" in self.tensor_file_map:
                last += 1
            if last - int(m2.group(2)) <= _PREFETCH_TAIL:
                nxt = None
        if nxt and f"{nxt}.0.{proj_names[0]}.weight" in self.tensor_file_map:
            count = 0
            while f"{nxt}.{count}.{proj_names[0]}.weight" in self.tensor_file_map:
                count += 1
            ranges = self._ranges_for(keys_of(nxt, count))
            if _executor is None:
                _executor = ThreadPoolExecutor(1)
            _pending[nxt] = _executor.submit(_prefetch_ranges, ranges)
""",
    s,
)

# MXFP4 (V4.1, V4-Flash, Vision, MiMo) and NVFP4 (Flash-Next, GLM):
# prefetch before the Python loop collects the views, so the loop's scale
# conversions also hit the cache
# (a loader without SCALE_SUFFIX names the MXFP4 scales ".scale")
for fmt, scales in (("MXFP4", '("weight", getattr(self, "SCALE_SUFFIX", "scale"))'),
                    ("NVFP4", '("weight", "weight_scale", "weight_scale_2")')):
    anchor = f"""                f"No {fmt} experts found under any of: {{self._experts_prefix_candidates(base_key)}}"
            )

        gate_weights = [None] * expert_count
"""
    s = once(
        anchor,
        anchor + f"""        self.prefetch_layer(prefix, {scales}, expert_count, (gate_name, up_name, down_name))
""",
        s,
    )

F.write_text(s)
print(f"patched {F}")
