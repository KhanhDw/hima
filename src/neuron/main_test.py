"""
main.py - Benchmark hiệu suất / tốc độ cho SNN (Network + FastSynapseGroup + Neuron).

Đặt file này ở src/neuron/main.py (cùng cấp với thư mục core/ và network/).

Ví dụ:
    python main.py                               # chạy các preset tiny..large
    python main.py --presets all --profile       # thêm preset xlarge + phân tích từng giai đoạn
    python main.py --custom 2048 512 256         # n_input n_neurons fan_in
    python main.py --device cpu --threads 8
    python main.py --build connector             # đo cả thời gian dựng mạng bằng Connector thật
    python main.py --csv result.csv

Mô hình topology (phù hợp với ràng buộc hiện tại của Network):
    - Mỗi FastSynapseGroup có đúng n_input synapse: synapse k của mọi group nhận spike
      của input neuron k (vì FastSynapseGroup.receive yêu cầu shape == (num_synapses,)).
    - Số group = n_neurons * fan_in / n_input; tổng synapse = n_neurons * fan_in.
    - Synapse được gán ngẫu nhiên (hoán vị) cho các neuron, mỗi neuron có đúng fan_in synapse.
    - Chỉ số toàn cục của synapse trong Tensor đã nối = group_idx * n_input + k,
      đúng với cách Dendrite.current() đang index.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.input_neuron import InputNeuronGroup  # noqa: E402
from core.neuron import Neuron  # noqa: E402
from core.synapse import FastSynapseGroup  # noqa: E402
from network.connector import Connector  # noqa: E402
from network.network import Network  # noqa: E402

try:
    import psutil

    _PROC = psutil.Process()
except ImportError:  # psutil là tuỳ chọn
    _PROC = None


# ======================================================================
# CẤU HÌNH
# ======================================================================


@dataclass(frozen=True)
class BenchConfig:
    name: str
    n_input: int
    n_neurons: int
    fan_in: int

    def __post_init__(self):
        if min(self.n_input, self.n_neurons, self.fan_in) < 1:
            raise ValueError("n_input, n_neurons, fan_in phải >= 1.")
        if (self.n_neurons * self.fan_in) % self.n_input != 0:
            raise ValueError(
                f"[{self.name}] n_neurons * fan_in ({self.n_neurons * self.fan_in}) "
                f"phải chia hết cho n_input ({self.n_input})."
            )

    @property
    def n_synapses(self) -> int:
        return self.n_neurons * self.fan_in

    @property
    def n_groups(self) -> int:
        return self.n_synapses // self.n_input


PRESETS: dict[str, BenchConfig] = {
    "tiny": BenchConfig("tiny", 64, 16, 16),            # 256 synapse
    "small": BenchConfig("small", 256, 64, 64),         # 4K synapse
    "medium": BenchConfig("medium", 1024, 256, 128),    # 32K synapse
    "large": BenchConfig("large", 2048, 1024, 256),     # 262K synapse
    "xlarge": BenchConfig("xlarge", 4096, 2048, 512),   # 1M synapse
}
DEFAULT_PRESETS = ["tiny", "small", "medium", "large"]


# ======================================================================
# TIỆN ÍCH
# ======================================================================


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rss_mb() -> float:
    return _PROC.memory_info().rss / 2**20 if _PROC else float("nan")


def percentile(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, int(round(q / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def fmt_si(x: float) -> str:
    for unit in ("", "K", "M", "G"):
        if abs(x) < 1000:
            return f"{x:.2f}{unit}"
        x /= 1000
    return f"{x:.2f}T"


def resolve_device(name: str) -> torch.device:
    # Lưu ý: dùng "cuda:0" (có index) thay vì "cuda". Tensor trên GPU luôn có
    # device = cuda:0, mà torch.device("cuda") != torch.device("cuda:0"), nên các
    # phép so sánh `x.device != self.device` trong Network/Neuron sẽ báo lỗi sai.
    if name in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if name == "cuda":
            print("[WARN] Không tìm thấy CUDA, chuyển sang CPU.")
    return torch.device("cpu")


# ======================================================================
# DỰNG MẠNG
# ======================================================================


class _AxonProxy:
    """Axon tối giản, chỉ để Connector.connect() gọi branch()."""

    def __init__(self):
        self.count = 0

    def branch(self, weight: float, initial_state: float = 0.0) -> int:
        idx = self.count
        self.count += 1
        return idx


class _InputSource:
    """Đại diện cho một input neuron (InputNeuronGroup không có Axon)."""

    def __init__(self):
        self.axon = _AxonProxy()


class _DendriteView:
    """Dendrite thật nhưng cộng offset: Connector dùng chỉ số cục bộ của group,
    còn Dendrite cần chỉ số toàn cục trên Tensor đã nối của mọi group."""

    def __init__(self, dendrite: Any, offset: int):
        self._dendrite = dendrite
        self._offset = offset

    def attach(self, synapse_idx: int):
        return self._dendrite.attach(synapse_idx + self._offset)

    def build(self):  # Network.build() sẽ build Dendrite thật
        return self


class _PostView:
    def __init__(self, dendrite_view: _DendriteView):
        self.dendrite = dendrite_view


def build_network(
    cfg: BenchConfig,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> Network:
    n_in, n_out, fan_in = cfg.n_input, cfg.n_neurons, cfg.fan_in
    total = cfg.n_synapses

    input_group = InputNeuronGroup(n_in, device=device, dtype=dtype)
    net = Network(input_group, device=device, dtype=dtype)

    groups = [
        FastSynapseGroup(
            n_in,
            max_delay=args.max_delay,
            device=str(device),
            dtype=dtype,
        )
        for _ in range(cfg.n_groups)
    ]
    for group in groups:
        net.add_synapse_group(group)

    neurons = [
        Neuron(
            theta=1.0,
            leak=0.9,
            refractory_period=2,
            device=str(device),
            dtype=dtype,
        )
        for _ in range(n_out)
    ]
    for neuron in neurons:
        net.add_neuron(neuron)

    # Hoán vị ngẫu nhiên: neuron j sở hữu các synapse topo[j, :].
    topo = torch.randperm(total, generator=gen).view(n_out, fan_in)

    # Trọng số: dòng kích thích trung bình mỗi bước ≈ args.drive.
    w_mean = args.drive / (fan_in * args.rate)
    weights = torch.rand(total, generator=gen) * (2.0 * w_mean)
    delays = torch.randint(1, args.max_delay + 1, (total,), generator=gen)

    if args.build == "fast":
        # Ghi trọng số/delay vector hoá, chỉ dùng Dendrite.attach() cho kết nối.
        weights_by_slot = torch.empty(total)
        weights_by_slot[topo.flatten()] = weights
        delays_by_slot = torch.empty(total, dtype=torch.int64)
        delays_by_slot[topo.flatten()] = delays

        for g, group in enumerate(groups):
            sl = slice(g * n_in, (g + 1) * n_in)
            group.weights.copy_(weights_by_slot[sl].to(device=device, dtype=dtype))
            group.delays.copy_(delays_by_slot[sl].to(device))

        for j, neuron in enumerate(neurons):
            for slot in topo[j].tolist():
                neuron.dendrite.attach(slot)

    else:  # "connector": đi qua Connector.connect() thật cho từng kết nối
        owner = torch.empty(total, dtype=torch.int64)
        owner[topo.flatten()] = torch.arange(n_out).repeat_interleave(fan_in)
        w_by_slot = torch.empty(total)
        w_by_slot[topo.flatten()] = weights
        d_by_slot = torch.empty(total, dtype=torch.int64)
        d_by_slot[topo.flatten()] = delays

        owner_l, w_l, d_l = owner.tolist(), w_by_slot.tolist(), d_by_slot.tolist()
        sources = [_InputSource() for _ in range(n_in)]

        for g, group in enumerate(groups):
            connector = Connector(group)
            views: dict[int, _PostView] = {}
            base = g * n_in
            # Connector cấp synapse_idx tuần tự => gọi connect() theo thứ tự k = 0..n_in-1
            for k in range(n_in):
                slot = base + k
                j = owner_l[slot]
                view = views.get(j)
                if view is None:
                    view = _PostView(_DendriteView(neurons[j].dendrite, base))
                    views[j] = view
                connector.connect(
                    sources[k], view, weight=w_l[slot], delay=d_l[slot]
                )
            connector.build()

    net.build()
    return net


def make_inputs(
    n_rows: int,
    n_in: int,
    rate: float,
    device: torch.device,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> torch.Tensor:
    spikes = torch.rand(n_rows, n_in, generator=gen) < rate
    return spikes.to(device=device, dtype=dtype)


def full_reset(net: Network) -> None:
    """Network.reset() không reset FastSynapseGroup (group chỉ có reset_state(),
    còn Network tìm `group.reset`), nên reset thủ công ở đây."""
    net.reset()
    for group in net.synapse_groups:
        group.reset_state()


# ======================================================================
# PHÂN TÍCH TỪNG GIAI ĐOẠN CỦA MỘT TIMESTEP
# ======================================================================


def profile_stages(
    net: Network,
    inputs: torch.Tensor,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    """Lặp lại đúng các bước của Network.step() nhưng đo riêng từng giai đoạn."""
    acc = {"input": 0.0, "synapse": 0.0, "neurons": 0.0, "stack": 0.0}
    n_rows = inputs.shape[0]
    clock = time.perf_counter

    for i in range(steps):
        x = inputs[i % n_rows]
        now = net.time

        sync(device)
        t0 = clock()
        x = net.input_group.stimulate(x, now)
        sync(device)
        t1 = clock()
        syn_out = net._process_synapses(x)  # noqa: SLF001
        sync(device)
        t2 = clock()
        spikes = [neuron.step(syn_out, now) for neuron in net.neurons]
        sync(device)
        t3 = clock()
        torch.stack(spikes, dim=0)
        sync(device)
        t4 = clock()

        net.time += 1
        acc["input"] += t1 - t0
        acc["synapse"] += t2 - t1
        acc["neurons"] += t3 - t2
        acc["stack"] += t4 - t3

    return {k: v / steps * 1e3 for k, v in acc.items()}  # ms/step


# ======================================================================
# BENCHMARK MỘT CẤU HÌNH
# ======================================================================


def run_benchmark(
    cfg: BenchConfig,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(args.seed)
    is_cuda = device.type == "cuda"

    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        cuda_base = torch.cuda.memory_allocated(device)
    rss_base = rss_mb()

    # ---- 1. Dựng mạng -------------------------------------------------
    sync(device)
    t0 = time.perf_counter()
    net = build_network(cfg, args, device, dtype, gen)
    sync(device)
    build_s = time.perf_counter() - t0

    n_rows = max(args.steps, args.warmup, 1)
    inputs = make_inputs(n_rows, cfg.n_input, args.rate, device, dtype, gen)

    result: dict[str, Any] = {
        "name": cfg.name,
        "n_input": cfg.n_input,
        "n_neurons": cfg.n_neurons,
        "n_groups": cfg.n_groups,
        "n_synapses": cfg.n_synapses,
        "build_s": build_s,
        "status": "ok",
    }

    with torch.no_grad():
        # ---- 2. Warmup + ước lượng thời gian mỗi bước ------------------
        for i in range(args.warmup):
            net.step(inputs[i % n_rows])
        sync(device)

        probe = 3
        t0 = time.perf_counter()
        for i in range(probe):
            net.step(inputs[i % n_rows])
        sync(device)
        est = max((time.perf_counter() - t0) / probe, 1e-6)

        steps = args.steps
        if est * (steps + min(steps, 50)) > args.time_limit:
            steps = max(args.min_steps, min(steps, int(args.time_limit / est / 1.5)))
        lat_steps = min(steps, 50)
        result["steps"] = steps
        result["steps_reduced"] = steps < args.steps

        # ---- 3. Throughput: chạy liền bằng Network.run() ---------------
        full_reset(net)
        sync(device)
        t0 = time.perf_counter()
        out = net.run(inputs[:steps])
        sync(device)
        elapsed = time.perf_counter() - t0

        out_rate = out.float().mean().item()
        result.update(
            run_s=elapsed,
            steps_per_s=steps / elapsed,
            ms_per_step=elapsed / steps * 1e3,
            us_per_neuron_step=elapsed / steps / cfg.n_neurons * 1e6,
            syn_events_per_s=steps * cfg.n_synapses / elapsed,
            neuron_updates_per_s=steps * cfg.n_neurons / elapsed,
            realtime_factor=(steps / elapsed) * (args.dt_ms / 1e3),
            out_spike_rate=out_rate,
            in_spike_rate=inputs[:steps].float().mean().item(),
        )
        if out_rate == 0.0:
            result["status"] = "warn: mạng không phát spike nào (đổi --drive/--rate)"

        # ---- 4. Latency từng bước (sync mỗi bước) ----------------------
        full_reset(net)
        lat_ms: list[float] = []
        for i in range(lat_steps):
            x = inputs[i % n_rows]
            sync(device)
            t0 = time.perf_counter()
            net.step(x)
            sync(device)
            lat_ms.append((time.perf_counter() - t0) * 1e3)
        lat_sorted = sorted(lat_ms)
        result.update(
            lat_mean_ms=sum(lat_ms) / len(lat_ms),
            lat_p50_ms=percentile(lat_sorted, 50),
            lat_p95_ms=percentile(lat_sorted, 95),
            lat_max_ms=lat_sorted[-1],
        )

        # ---- 5. Phân tích từng giai đoạn (tuỳ chọn) ---------------------
        if args.profile:
            full_reset(net)
            result["stages_ms"] = profile_stages(
                net, inputs, min(steps, 30), device
            )

    # ---- 6. Bộ nhớ ------------------------------------------------------
    if is_cuda:
        result["mem_mb"] = (
            torch.cuda.max_memory_allocated(device) - cuda_base
        ) / 2**20
    else:
        result["mem_mb"] = rss_mb() - rss_base  # nan nếu chưa cài psutil

    del net, inputs, out
    return result


# ======================================================================
# IN KẾT QUẢ
# ======================================================================


def print_result(r: dict[str, Any]) -> None:
    print(
        f"  Dựng mạng      : {r['build_s']:.3f} s"
        f"   | steps đo: {r['steps']}"
        + ("  (đã giảm để vừa --time-limit)" if r["steps_reduced"] else "")
    )
    print(
        f"  Throughput     : {r['steps_per_s']:.1f} step/s"
        f"   ({r['ms_per_step']:.3f} ms/step, "
        f"{r['us_per_neuron_step']:.1f} µs/neuron/step)"
    )
    print(
        f"  Synapse events : {fmt_si(r['syn_events_per_s'])}/s"
        f"   | neuron updates: {fmt_si(r['neuron_updates_per_s'])}/s"
        f"   | realtime x{r['realtime_factor']:.3f}"
    )
    print(
        f"  Latency (ms)   : mean {r['lat_mean_ms']:.3f}  p50 {r['lat_p50_ms']:.3f}"
        f"  p95 {r['lat_p95_ms']:.3f}  max {r['lat_max_ms']:.3f}"
    )
    print(
        f"  Spike rate     : in {r['in_spike_rate']:.3f}  out {r['out_spike_rate']:.3f}"
        f"   | bộ nhớ ≈ {r['mem_mb']:.1f} MB"
    )
    stages = r.get("stages_ms")
    if stages:
        total = sum(stages.values())
        parts = "  ".join(
            f"{k} {v:.3f}ms ({v / total * 100:.0f}%)" for k, v in stages.items()
        )
        print(f"  Giai đoạn      : {parts}")
    if r["status"] != "ok":
        print(f"  [{r['status']}]")


def print_summary(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'config':<8}{'input':>7}{'neurons':>9}{'synapses':>11}{'build(s)':>10}"
        f"{'ms/step':>10}{'p95(ms)':>10}{'step/s':>10}{'syn-ev/s':>11}"
        f"{'out-rate':>10}{'mem(MB)':>10}"
    )
    print("\n" + "=" * len(header))
    print("TỔNG KẾT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"].startswith("FAILED"):
            print(f"{r['name']:<8}{r['n_input']:>7}{r['n_neurons']:>9}"
                  f"{r['n_synapses']:>11}   {r['status']}")
            continue
        print(
            f"{r['name']:<8}{r['n_input']:>7}{r['n_neurons']:>9}{r['n_synapses']:>11}"
            f"{r['build_s']:>10.3f}{r['ms_per_step']:>10.3f}{r['lat_p95_ms']:>10.3f}"
            f"{r['steps_per_s']:>10.1f}{fmt_si(r['syn_events_per_s']):>11}"
            f"{r['out_spike_rate']:>10.3f}{r['mem_mb']:>10.1f}"
        )
    print("=" * len(header))


def write_csv(path: str, results: list[dict[str, Any]]) -> None:
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "stages_ms"}
        for k, v in r.get("stages_ms", {}).items():
            row[f"stage_{k}_ms"] = v
        rows.append(row)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi CSV: {path}")


# ======================================================================
# MAIN
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SNN: nhiều neuron & synapse")
    p.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS,
                   choices=[*PRESETS, "all"], help="preset cần chạy")
    p.add_argument("--custom", nargs=3, type=int, metavar=("N_INPUT", "N_NEURONS", "FAN_IN"),
                   help="thêm cấu hình tuỳ chọn; n_neurons*fan_in phải chia hết cho n_input")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (CPU)")
    p.add_argument("--steps", type=int, default=200, help="số timestep đo")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--min-steps", type=int, default=5)
    p.add_argument("--time-limit", type=float, default=30.0,
                   help="ngân sách giây mỗi cấu hình; vượt thì tự giảm số step")
    p.add_argument("--rate", type=float, default=0.05, help="xác suất spike của input mỗi step")
    p.add_argument("--drive", type=float, default=0.15,
                   help="dòng kích thích trung bình mỗi neuron mỗi step (theta=1, leak=0.9)")
    p.add_argument("--max-delay", type=int, default=8)
    p.add_argument("--dt-ms", type=float, default=1.0, help="ms mô phỏng mỗi timestep (tính realtime)")
    p.add_argument("--build", choices=["fast", "connector"], default="fast",
                   help="fast: ghi vector hoá; connector: gọi Connector.connect() từng kết nối")
    p.add_argument("--profile", action="store_true", help="phân tích thời gian từng giai đoạn")
    p.add_argument("--csv", type=str, default="", help="ghi kết quả ra file CSV")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    names = list(PRESETS) if "all" in args.presets else args.presets
    configs = [PRESETS[n] for n in names]
    if args.custom:
        configs.append(BenchConfig("custom", *args.custom))

    print(f"PyTorch {torch.__version__} | device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(device)})")
    else:
        print(f" | threads: {torch.get_num_threads()}")
    print(f"dtype: {args.dtype} | build: {args.build} | steps: {args.steps} "
          f"(warmup {args.warmup}) | input rate: {args.rate} | max_delay: {args.max_delay}")
    if _PROC is None and device.type == "cpu":
        print("[INFO] Cài psutil (pip install psutil) để đo RAM trên CPU.")

    results: list[dict[str, Any]] = []
    for cfg in configs:
        print(
            f"\n--- {cfg.name}: {cfg.n_input} input, {cfg.n_neurons} neuron, "
            f"fan-in {cfg.fan_in}, {cfg.n_synapses:,} synapse ({cfg.n_groups} group) ---"
        )
        try:
            r = run_benchmark(cfg, args, device, dtype)
            print_result(r)
        except (RuntimeError, MemoryError) as exc:  # gồm cả CUDA OOM
            msg = str(exc).splitlines()[0][:80]
            print(f"  [LỖI] {msg}")
            r = {
                "name": cfg.name, "n_input": cfg.n_input, "n_neurons": cfg.n_neurons,
                "n_groups": cfg.n_groups, "n_synapses": cfg.n_synapses,
                "status": f"FAILED: {msg}",
            }
        results.append(r)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_summary(results)
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()