"""Fixed-scope figures from the completed numerical index, without new reduction."""
import argparse
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now

COLORS = ["#155e75", "#b45309", "#6d28d9", "#374151"]


def run(source, output):
    if output.exists():
        raise ValueError("figures require a new retained output directory")
    receipt = read(source/"RECEIPT.json")
    if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or receipt["native_cards"] != 30:
        raise ValueError("figures await the completed numerical index")
    path = source/"NATIVE_CARDS.json"
    if file_digest(path) != receipt["output_hashes"][path.name]:
        raise ValueError("figure inputs changed after the completed numerical projection")
    cards = {row["card_id"]: row for row in read(path)["cards"]}
    output.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    captions = {}
    def save(fig, name, caption):
        fig.text(.06, .015, textwrap.fill(caption, 150), va="bottom", fontsize=9)
        fig.savefig(output/(name+".svg"), bbox_inches="tight")
        fig.savefig(output/(name+".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
        captions[name] = caption
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(bottom=.24, wspace=.28)
    for ax, cid, xkey, ykey, arms, title in [
        (axes[0], "K01", "search_primitives_mean", "success_mean", ["personal", "pooled", "primitive"], "Learned fragments and generic training"),
        (axes[1], "K04", "search_cost", "success", ["motif", "options", "generic-options", "primitive"], "Fragments and transition-derived options")]:
        card = cards[cid]
        for arm_index, (color, arm) in enumerate(zip(COLORS, arms)):
            rows = sorted([row for row in card["performance_cost"] if row["arm"] == arm],
                          key=lambda row: row["condition_factors"]["search_primitive_budget"])
            if len(rows) != 3:
                raise ValueError("construction figure omitted a frozen budget regime")
            xs = [row["recorded_metrics"][xkey] for row in rows]
            ys = [row["recorded_metrics"][ykey] for row in rows]
            ax.plot(xs, ys, "o-", label=arm.replace("-", " "), color=color, linewidth=1.7, markersize=5)
            for budget_index, (x, y, row) in enumerate(zip(xs, ys, rows)):
                offset = 4+10*arm_index+12*(budget_index == 1)
                ax.annotate(str(row["condition_factors"]["search_primitive_budget"]), (x,y), xytext=(4,offset), textcoords="offset points", fontsize=8, color=color)
        ax.set(title=title+f"\n{cid}: {card['n_per_condition']} makers per budget",
               xlabel="Mean search primitive evaluations", ylabel="Mean successful task fraction", ylim=(-.04,1.15))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    save(fig, "construction-cost", "Each point is one original budget regime (labels 8, 32, 128); lines connect regimes for the same arm and are not pooled estimates. Training has equal attempt counts; personal and pooled training contain different histories. Search primitives exclude training, definition costs and OS overhead, which remain separately recorded. These are descriptive arm means; paired uncertainty is in the complete comparison table.")

    card = cards["P03"]
    fig, ax = plt.subplots(figsize=(9,5.3))
    fig.subplots_adjust(bottom=.25)
    for color, arm in zip(COLORS, ["maker", "direct-table", "generic"]):
        rows = sorted([row for row in card["performance_cost"] if row["arm"] == arm],
                      key=lambda row: row["condition_factors"]["prior_works"])
        if [row["condition_factors"]["prior_works"] for row in rows] != [0,1,2,4,8]:
            raise ValueError("evidence curve omitted a frozen evidence dose")
        ax.plot([row["condition_factors"]["prior_works"] for row in rows],
                [row["recorded_metrics"]["future_log_score"] for row in rows],
                marker="o", linestyle="--" if arm == "direct-table" else "-", color=color, label=arm.replace("-", " "))
    ax.set(title=f"Evidence dose and future-event prediction\nP03: {card['n_per_condition']} makers per dose",
           xlabel="Permitted prior works from the same maker", ylabel="Mean log probability of the next event (nats)", xticks=[0,1,2,4,8])
    ax.grid(alpha=.18)
    ax.legend()
    save(fig, "evidence-dose", "Higher log probability means better prediction of the actual next event. The maker model and direct prediction table receive the same permitted model and evidence; the generic model omits maker-specific history. Doses are separate conditions sharing the declared nested sampling structure. Coincident curves remain coincident; lines do not imply a fitted dose-response law or additional confirmation.")

    card = cards["P04"]
    condition_ids = [row["id"] for row in card["definition"]["conditions"]]
    fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True, sharex=True)
    fig.subplots_adjust(bottom=.24, left=.15, wspace=.3)
    for ax, rival, title in [(axes[0], "softmax-maker", "Mixture minus softmax-only maker model"),
                             (axes[1], "direct-mixture", "Mixture minus direct prediction mixture")]:
        selected = {row["condition"]: row for row in card["comparisons"]
            if row["estimand"]["arm"] == "mixture" and row["estimand"]["rival"] == rival
            and row["estimand"]["target"] == "future_log_score"}
        if set(selected) != set(condition_ids):
            raise ValueError("mechanism figure omitted a registered family")
        for index, condition in enumerate(condition_ids):
            row = selected[condition]
            low, high = row["interval_95"]
            ax.plot([low, high], [index,index], color=COLORS[0], linewidth=2)
            ax.plot(row["mean"], index, "o", color=COLORS[0])
        ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Paired future log-score difference (nats)", yticks=range(len(condition_ids)),
               yticklabels=[name.replace("W1-", "Graphic: ").replace("W2-", "Assembly: ") for name in condition_ids])
        ax.grid(axis="x", alpha=.18)
    axes[0].invert_yaxis()
    fig.suptitle(f"Finite production families, with matched information\nP04: {card['n_per_condition']} makers per family", y=1.01)
    save(fig, "mechanism-comparison", "Dots are equal-maker means of paired predictive-score differences; bars are the original 95% constructor-then-maker bootstrap intervals. Both panels use the same horizontal scale; numerical roundoff remains in the full tables. Rows are separately configured graphic or assembly production families. Both comparisons use four permitted prior works. This compares finite reader families within each world; it does not transfer fitted graphic-world parameters to assembly or establish behavior in real text.")
    (output/"README.md").write_text("# V16 figures\n\nGenerated from the final numerical index without pooling conditions or recomputing estimates.\n\n"+
        "\n\n".join(f"![{name}]({name}.svg)\n\n{caption}" for name,caption in captions.items())+"\n", encoding="utf-8", newline="\n")
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "source_projection_sha256": file_digest(source/"RECEIPT.json"), "source_data_sha256": file_digest(path),
        "producer_sha256": file_digest(Path(__file__)), "figures": captions,
        "output_hashes": {item.name: file_digest(item) for item in output.iterdir() if item.is_file()},
        "scope": "Fixed K01/K04 budget, P03 evidence-dose and P04 finite-family views chosen for the commissioned report; recorded descriptive means and paired intervals, with no new scientific reduction or additional confirmatory claim"}
    write(output/"RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print({"figures": len(run(args.source, args.output)["figures"])}, flush=True)


if __name__ == "__main__":
    main()
