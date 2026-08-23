import argparse
import time
import torch
from engine import SpeculativeEngine
from baseline import BaselineGenerator

def run_race(prompts, max_new_tokens=40, gamma=4):
    print(f"\n{'='*60}")
    print(f"  THE RACE: Baseline vs. Speculative (Gamma={gamma})")
    print(f"{'='*60}")

    baseline = BaselineGenerator()
    speculative = SpeculativeEngine()
    
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
        
        results.append(speedup)

    if len(results) > 0:
        avg_speedup = sum(results) / len(results)
    else:
        avg_speedup = 0

    print(f"\n{'='*60}")
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
    ]
    
    run_race(
        prompts,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
    )
