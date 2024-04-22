import subprocess
import argparse
import re

def run_experiment(args):
    # load file including all the scripts
    with open(f"scripts/{args.exp}.txt") as f:
        for line in f.readlines():
            c = line.replace("\n","")
            c = c.strip()
            if c.startswith("python"):
                for seed in args.seeds:
                    command = c+f" --{seed=}"
                    if args.dry_run:
                        command = re.sub(r"epoch \d*", "epoch 2", command)
                    subprocess.run(command, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument('--exp', type=str, required=True, help='Experiment script name')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='List of seeds')
    parser.add_argument('--dry_run', action="store_true", help='run for 2 epochs only')
    
    args = parser.parse_args()
    if args.dry_run:
        args.seeds = [1,2]
    run_experiment(args)

if __name__ == "__main__":
    main()
