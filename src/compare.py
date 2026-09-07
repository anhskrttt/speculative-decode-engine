import argparse
import time
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
from engine import SpeculativeEngine
from baseline import BaselineGenerator

def run_race(prompts, max_new_tokens=40, gamma=4):
    baseline = BaselineGenerator()
    speculative = SpeculativeEngine()
    
    # Warm up
    warmup_prompt="The secret to happiness is"
    
    baseline.generate(warmup_prompt, max_new_tokens=10)
    speculative.generate(warmup_prompt, max_new_tokens=10, gamma=4)
    
    if torch.cuda.is_available() and torch.cuda.current_device() is not None:
        torch.cuda.synchronize()
    
    # Real run
    print(f"\n{'='*60}")
    print(f"  THE RACE: Baseline vs. Speculative (Gamma={gamma})")
    print(f"{'='*60}")

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n\n Lap {i+1}: '{prompt}'")
        print("-" * 40)
        
        #baseline
        print(" Running Baseline...")
        base_speed, base_text = baseline.generate(prompt, max_new_tokens)
        print(f"   Speed: {base_speed:.2f} tokens/sec")
        
        #speculative
        print(" Running Speculative...")
        
       
        spec_speed, spec_text = speculative.generate(prompt, max_new_tokens, gamma)
        
        print(f"   Speed: {spec_speed:.2f} tokens/sec")
        

        if base_speed > 0:
            speedup = spec_speed / base_speed
        else:
            speedup = 1.0
            
        print(f" SPEEDUP: {speedup:.2f}x")
        
        # results.append({"speedup": speedup,"base_speed": base_speed,"spec_speed": spec_speed})
        results.append({
            "prompt": prompt,
            "speedup": speedup,
            "base_speed": base_speed,
            "spec_speed": spec_speed,
        })

    
    prompt_labels = [f"{i + 1}" for i in range(len(results))]
    baseline_speeds = [r["base_speed"] for r in results]
    speculative_speeds = [r["spec_speed"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        prompt_labels,
        baseline_speeds,
        marker="o",
        linewidth=2,
        color="tab:gray",
        label="Baseline",
    )
    ax.plot(
        prompt_labels,
        speculative_speeds,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label=f"Speculative Decoding (gamma={gamma})",
    )

    ax.set_title("Baseline vs. Speculative Decoding Speed")
    ax.set_xlabel("Prompt Index")
    ax.set_ylabel("Generation speed (tokens/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.text(
        0.5, 0.02,
        (
            f"Gamma: {gamma}  •  Output tokens per prompt: {max_new_tokens}\n"
            f"Each point uses one distinct prompt."
        ),
        ha="center",
        fontsize=9,
        color="dimgray",
    )

    fig.subplots_adjust(bottom=0.20)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)

    graph_path = f"results/race_comparison_{timestamp}.png"
    plt.savefig(graph_path, dpi=150)
    print(f"\n📊 Graph saved to: {graph_path}")

    if len(results) > 0:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
    else:
        avg_speedup = 0
        
    print(f"\n{'='*60}")
    print(f" Average baseline speed: {sum(r['base_speed'] for r in results)/len(results):.2f} tokens/sec")
    print(f" Average speculative speed: {sum(r['spec_speed'] for r in results)/len(results):.2f} tokens/sec")
    print(f" FINAL VERDICT: {avg_speedup:.2f}x AVERAGE SPEEDUP")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline decoding with speculative decoding."
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=4,
        help="Number of draft tokens proposed per speculative step (default: 4).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum tokens to generate per prompt (default: 40).",
    )

    args = parser.parse_args()

    if args.gamma < 1:
        parser.error("--gamma must be at least 1")
    
    prompts = [
            "The future of Artificial Intelligence is",
            "Once upon a time in a distant galaxy",
            "Python is a great programming language because",
            "The recipe for a perfect chocolate cake involves"
            "The most important lesson life has taught me is"
            "If I could travel anywhere in the world, I would go to"
            "The secret to happiness is"
            "Looking back at the last decade, the biggest change has been"
            "My favorite childhood memory is"
    ]
    
    run_race(
        prompts,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
    )
