import time
import torch
import matplotlib.pyplot as plt
from engine import SpeculativeEngine

import os
import platform

from config import DEVICE

def get_hardware_info():
    if torch.cuda.is_available() and DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{gpu_name} ({total_vram_gb:.1f} GB VRAM)"

    return f"CPU: {platform.processor()}"

def run_gamma_experiment():
    print(f"\n{'='*60}")
    print(f" EXPERIMENT: Finding the Perfect Gamma")
    print(f"{'='*60}")

    engine = SpeculativeEngine()
    
    # We use a consistent prompt so the comparison is fair
    prompt = "The rapid development of artificial intelligence has led to"
    max_tokens = 40
    
    # Test different draft lengths
    max_gamma = 50
    gammas = list(range(1, max_gamma + 1))
    
    speeds = []
    acceptance_rates = []

    print("\nStarting stress test...")
    print("-" * 50)
    print(f"{'Gamma':<10} | {'Speed (t/s)':<15} | {'Acceptance %':<15}")
    print("-" * 50)

    for gamma in gammas:
        # Run the engine
        start_time = time.time()
        
        _ = engine.generate(prompt, max_new_tokens=max_tokens, gamma=gamma)
        
        end_time = time.time()
        
        # Calculate Speed
        total_time = end_time - start_time
        speed = max_tokens / total_time
        
        acc_rate = engine.metrics['acceptance_rate'] * 100
        
        speeds.append(speed)
        acceptance_rates.append(acc_rate)
        
        print(f"{gamma:<10} | {speed:<15.2f} | {acc_rate:<15.2f}")

    print("-" * 50)
    print("✅ Experiment Complete. Generating Graph...")

    # graph
    plt.figure(figsize=(10, 5))

    # Subplot 1: Speed vs Gamma
    plt.subplot(1, 2, 1)
    plt.plot(gammas, speeds, marker='o', color='b', linewidth=2)
    plt.title("Speed vs. Draft Length (Gamma)")
    plt.xlabel("Gamma (Draft Tokens)")
    plt.ylabel("Tokens / Second")
    plt.grid(True)

    # Subplot 2: Acceptance Rate vs Gamma
    plt.subplot(1, 2, 2)
    plt.plot(gammas, acceptance_rates, marker='o', color='g', linewidth=2)
    plt.title("Acceptance Rate vs. Gamma")
    plt.xlabel("Gamma (Draft Tokens)")
    plt.ylabel("Acceptance Rate (%)")
    plt.grid(True)

    plt.tight_layout()

    import os
    os.makedirs("results", exist_ok=True)  
    plt.savefig("results/gamma_analysis.png") 
    print(f"\n📊 Graph saved to: results/gamma_analysis.png")
    
    hardware = get_hardware_info()
    
    results_path = "results/gamma_results.csv"
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("hardware,gamma,tokens_per_second,acceptance_rate_percent\n")

        for gamma, speed, acceptance_rate in zip(
            gammas, speeds, acceptance_rates
        ):
            file.write(
                f'"{hardware}",{gamma},{speed:.4f},{acceptance_rate:.4f}\n'
            )

    print(f"Raw results saved to: {results_path}")

if __name__ == "__main__":
    run_gamma_experiment()
