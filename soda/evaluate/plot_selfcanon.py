import os
import json
import argparse
import matplotlib.pyplot as plt


class PlotSelfCanon:
    def __init__(self, base_dir, iters, out_dir):
        self.base_dir = base_dir
        self.iters = iters
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.metrics = {
            "iter": [],
            "CIR": [],
            "singleton_rate": [],
            "orphan_rate": [],
            "schema_size": [],
            "num_relations": [],
            "redundancy": [],
        }

    # ======================================================
    # Load metrics per iteration
    # ======================================================
    def load_metrics(self):
        for i in self.iters:
            iter_dir = os.path.join(self.base_dir, f"iter{i}")

            sd_sc_path = os.path.join(iter_dir, "eval_sd_sc.json")
            red_path = os.path.join(iter_dir, "eval_relation_redundancy.json")

            if not os.path.exists(sd_sc_path):
                print(f"[WARN] Missing {sd_sc_path}")
                continue

            with open(sd_sc_path, "r", encoding="utf-8") as f:
                sdsc = json.load(f)

            self.metrics["iter"].append(i)
            self.metrics["CIR"].append(
                sdsc["sc_robustness"]["CIR_all"]
            )
            self.metrics["singleton_rate"].append(
                sdsc["canonical_kg"]["singleton_relation_rate"]
            )
            self.metrics["orphan_rate"].append(
                sdsc["schema_kg"]["orphan_relation_rate"]
            )
            self.metrics["schema_size"].append(
                sdsc["schema_kg"]["schema_size"]
            )
            self.metrics["num_relations"].append(
                sdsc["canonical_kg"]["num_relations"]
            )

            # -------- redundancy --------
            if os.path.exists(red_path):
                with open(red_path, "r", encoding="utf-8") as f:
                    red = json.load(f)
                self.metrics["redundancy"].append(
                    red["redundancy_semantic"]
                )
            else:
                self.metrics["redundancy"].append(None)
                print(f"[WARN] Missing {red_path}")

        print("[✓] Loaded metrics for iterations:", self.metrics["iter"])

    # ======================================================
    # Plot all figures
    # ======================================================
    def plot_all(self):
        self.plot_sc_convergence()
        self.plot_schema_kg_evolution()
        self.plot_redundancy_convergence()

    # ------------------------------------------------------
    # SC convergence: CIR / singleton / orphan
    # ------------------------------------------------------
    def plot_sc_convergence(self):
        iters = self.metrics["iter"]

        plt.figure(figsize=(7, 4.8))
        plt.plot(iters, self.metrics["singleton_rate"], marker="o", label="Singleton relation rate")
        plt.plot(iters, self.metrics["orphan_rate"], marker="^", label="Orphan schema rate")
        plt.plot(iters, self.metrics["CIR"], marker="s", label="Canonical Identity Rate (CIR)")

        plt.xlabel("Iteration")
        plt.ylabel("Metric value")
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(self.out_dir, "sc_convergence_core.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[✓] Saved {out}")

    # ------------------------------------------------------
    # Schema vs KG size evolution
    # ------------------------------------------------------
    def plot_schema_kg_evolution(self):
        iters = self.metrics["iter"]

        plt.figure(figsize=(7, 4.8))
        plt.plot(iters, self.metrics["schema_size"], marker="o", label="Schema size")
        plt.plot(iters, self.metrics["num_relations"], marker="s", label="#Relations in KG")

        plt.xlabel("Iteration")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(self.out_dir, "schema_kg_evolution.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[✓] Saved {out}")

    # ------------------------------------------------------
    # Semantic redundancy convergence
    # ------------------------------------------------------
    def plot_redundancy_convergence(self):
        iters = []
        values = []

        for i, r in zip(self.metrics["iter"], self.metrics["redundancy"]):
            if r is not None:
                iters.append(i)
                values.append(r)

        if not values:
            print("[INFO] No redundancy metrics to plot")
            return

        plt.figure(figsize=(6.5, 4.5))
        plt.plot(iters, values, marker="o", color="tab:red")
        plt.xlabel("Iteration")
        plt.ylabel("Semantic redundancy")
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out = os.path.join(self.out_dir, "semantic_redundancy_convergence.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[✓] Saved {out}")

    # ======================================================
    def run(self):
        self.load_metrics()
        self.plot_all()


# ==========================================================
# CLI
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser("Plot Self Canon evaluator")
    parser.add_argument("--base_dir", required=True,
                        help="Directory containing iter*/eval_*.json")
    parser.add_argument("--iters", required=True,
                        help="Comma-separated iterations, e.g. 0,1,2,3")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to save plots")
    return parser.parse_args()


def main():
    args = parse_args()
    iters = [int(x) for x in args.iters.split(",")]

    evaluator = PlotSelfCanon(
        base_dir=args.base_dir,
        iters=iters,
        out_dir=args.out_dir,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
