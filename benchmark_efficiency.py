#!/usr/bin/env python3
"""Comprehensive Efficiency & Resource Benchmark for Raspberry Pi 4."""
import os
import time
import numpy as np
import onnxruntime as ort
import psutil

MODELS = {
    "Apple": "best.onnx",
    "Mango": "best_mango.onnx",
    "Guava": "best_guava.onnx",
    "Pomegranate": "best_pomegranate.onnx",
}

def benchmark():
    dummy_input = np.random.randn(1, 3, 416, 416).astype(np.float32)
    process = psutil.Process(os.getpid())
    base_ram = process.memory_info().rss / (1024 * 1024)

    print("=" * 82)
    print(f"{'Target Fruit':14s} | {'Size':8s} | {'CPU Mean':10s} | {'CPU P95':10s} | {'FPS (4T)':9s} | {'RAM Footprint':13s}")
    print("=" * 82)

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Model not found: {path}")
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Test with 4 threads (standard for Pi 4 4-core Cortex-A72)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name

        # Warmup
        for _ in range(10):
            sess.run(None, {inp_name: dummy_input})

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            sess.run(None, {inp_name: dummy_input})
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(latencies)
        p95_ms = np.percentile(latencies, 95)
        fps = 1000.0 / mean_ms
        current_ram = process.memory_info().rss / (1024 * 1024)
        model_ram = current_ram - base_ram

        print(f"{name:14s} | {size_mb:5.2f} MB | {mean_ms:7.2f} ms | {p95_ms:7.2f} ms | {fps:6.1f} fps | {current_ram:6.1f} MB (active)")

    print("=" * 82)

if __name__ == "__main__":
    benchmark()
