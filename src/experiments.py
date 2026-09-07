import time
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from engine import SpeculativeEngine
import os
import platform
import argparse
import textwrap

from config import DEVICE, TARGET_MODEL_NAME, DRAFT_MODEL_NAME, TEMPERATURE, TOP_K, TOP_P

PROMPT = "The rapid development of artificial intelligence has led to"

def get_hardware_info():
    if torch.cuda.is_available() and DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{gpu_name} ({total_vram_gb:.1f} GB VRAM)"

    return f"CPU: {platform.processor()}"



def run_gamma_experiment(max_tokens: int, max_gamma: int):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hardware = get_hardware_info()
    
    # metadata = (
    #     f"Time: {timestamp}\n"
    #     f"Hardware: {hardware}\n"
    #     f"Target model: {TARGET_MODEL_NAME}\n"
    #     f"Draft model: {DRAFT_MODEL_NAME}\n"
    #     f"Generated tokens per run: {max_tokens}"
    # )
    
    print(f"\n{'='*60}")
    print(f" EXPERIMENT: Finding the Perfect Gamma")
    print(f"{'='*60}")

    engine = SpeculativeEngine()
    
    # We use a consistent prompt so the comparison is fair
    # prompt = "The rapid development of artificial intelligence has led to"
    # max_tokens = 40
    
    # Test different draft lengths
    # max_gamma = 20
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
        
        _ = engine.generate(PROMPT, max_new_tokens=max_tokens, gamma=gamma)
        
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
    # plt.figure(figsize=(10, 5))

    # # Subplot 1: Speed vs Gamma
    # plt.subplot(1, 2, 1)
    # plt.plot(gammas, speeds, marker='o', color='b', linewidth=2)
    # plt.title("Speed vs. Draft Length (Gamma)")
    # plt.xlabel("Gamma (Draft Tokens)")
    # plt.ylabel("Tokens / Second")
    # plt.grid(True)

    # # Subplot 2: Acceptance Rate vs Gamma
    # plt.subplot(1, 2, 2)
    # plt.plot(gammas, acceptance_rates, marker='o', color='g', linewidth=2)
    # plt.title("Acceptance Rate vs. Gamma")
    # plt.xlabel("Gamma (Draft Tokens)")
    # plt.ylabel("Acceptance Rate (%)")
    # plt.grid(True)

    # plt.tight_layout()
    
    # Single graph
    # fig, ax_speed = plt.subplots(figsize=(10, 5))
    fig, ax_speed = plt.subplots(figsize=(10, 6))
    

    # Left y-axis: speed
    speed_line = ax_speed.plot(
        gammas, speeds,
        marker="o", color="tab:blue", linewidth=2,
        label="Speed (tokens/s)"
    )
    ax_speed.set_xlabel("Gamma (Draft Tokens)")
    ax_speed.set_ylabel("Tokens / Second", color="tab:blue")
    ax_speed.tick_params(axis="y", labelcolor="tab:blue")
    ax_speed.grid(True, alpha=0.3)

    # Right y-axis: acceptance rate
    ax_acceptance = ax_speed.twinx()
    acceptance_line = ax_acceptance.plot(
        gammas, acceptance_rates,
        marker="s", color="tab:green", linewidth=2,
        label="Acceptance Rate (%)"
    )
    ax_acceptance.set_ylabel("Acceptance Rate (%)", color="tab:green")
    ax_acceptance.tick_params(axis="y", labelcolor="tab:green")
    ax_acceptance.set_ylim(0, 100)

    # Combine the two legends
    lines = speed_line + acceptance_line
    labels = [line.get_label() for line in lines]
    ax_speed.legend(lines, labels, loc="best")

    # plt.title("Speculative Decoding: Speed and Acceptance Rate vs. Gamma")
    
    # fig.subplots_adjust(bottom=0.27, top=0.85)
    
    # fig.text(
    #     0.5, 0.02,
    #     metadata,
    #     ha="center",
    #     va="bottom",
    #     fontsize=9,
    #     bbox={
    #         "boxstyle": "round,pad=0.5",
    #         "facecolor": "#f5f5f5",
    #         "edgecolor": "#cccccc",
    #     },
    # )
    
    # # TODO: FIXME. This runs for too long
    # fig.text(
    #     0.79, 0.5,          # x, y position in the whole figure
    #     metadata,
    #     ha="left",
    #     va="center",
    #     fontsize=9,
    #     bbox={
    #         "boxstyle": "round,pad=0.6",
    #         "facecolor": "#f5f5f5",
    #         "edgecolor": "#bdbdbd",
    #     },
    # )
    fig.suptitle(
        "Speculative Decoding: Speed and Acceptance Rate vs. Gamma",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    metadata = (
        f"Time: {timestamp}    •    "
        f"Target: {TARGET_MODEL_NAME}    •    "
        f"Draft: {DRAFT_MODEL_NAME}    •    "
        f"Output tokens: {max_tokens}\n"
        f"Hardware: {hardware}"
    )

    fig.text(
        0.5, 0.90,
        textwrap.fill(metadata, width=125),
        ha="center",
        va="top",
        fontsize=9,
        color="dimgray",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f7f7f7",
            "edgecolor": "#d0d0d0",
        },
    )

    # Make room for the note above the axes.
    fig.subplots_adjust(top=0.75)
    
    
    # plt.tight_layout()
    # fig.subplots_adjust(right=0.75, top=0.88)
    
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    graph_path = f"results/gamma_analysis_{timestamp}.png"
    plt.savefig(graph_path)
    # plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Graph saved to: {graph_path}")

    # hardware = get_hardware_info()

    results_path = f"results/gamma_results_{timestamp}.csv"
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
    parser = argparse.ArgumentParser(description="Run gamma sweep experiment")
    parser.add_argument("--max-tokens", type=int, default=40, help="Maximum number of tokens to generate")
    parser.add_argument("--max-gamma", type=int, default=20, help="Maximum gamma value to test")
    args = parser.parse_args()
    
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")

    if args.max_gamma <= 0:
        parser.error("--max-gamma must be positive.")

    run_gamma_experiment(max_tokens=args.max_tokens, max_gamma=args.max_gamma)
