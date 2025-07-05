import os
from concurrent.futures import ThreadPoolExecutor, as_completed

scripts = "bench.sh"

def run_one(batch):
    cmd = f"bash {scripts} 1 {batch} 1"
    print(f"[Batch {batch}] Running: {cmd}")
    os.system(cmd)

def run(start, end, step, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_one, batch)
            for batch in range(start, end + 1, step)
        ]
        for future in as_completed(futures):
            pass  # 또는 future.result()로 예외 발생 시 추적 가능

if __name__ == "__main__":
    # run(start=16, end=8192, step=16, max_workers=1)
    # run(start=16, end=64, step=16, max_workers=1)
    cmd = f"bash {scripts} 32 15 8192"
    os.system(cmd)

