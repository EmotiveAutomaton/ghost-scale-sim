"""File the complete accepted card inventory without pretending planned designs ran."""
from collections import Counter
import re
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import write, read, file_digest
from ghostscale.validation.soundingline.v16.admission import CONSUMERS
from ghostscale.validation.soundingline.v16.craft import DESIGN


def main():
    source = REPO / "docs/versions/v16-acquired-craft/CODING_PACKAGE.md"
    cards = []
    for line in source.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\| ([KPOSRMVXB][0-9]{2}) \| (.*?) \| (.*?) \|$", line)
        if not match:
            continue
        card, question, output = match.groups()
        cards.append({"card_id": card, "question_and_comparison": question,
                      "required_output_and_dependencies": output,
                      "source": "docs/versions/v16-acquired-craft/CODING_PACKAGE.md",
                      "execution_state": "planned", "instrument_state": "untested",
                      "criterion_state": "untested",
                      "evidence_scope": "discovery" if card[0] not in "XB" else "fixture",
                      "access_tier": "pending per-packet design",
                      "dependency_ids": [attack for attack, consumers in CONSUMERS.items() if card in consumers],
                      "pursuit": "OPENED", "warrant": "DESCRIPTIVE ONLY",
                      "queue_eligible": False,
                      "design_state": "commission recorded; implementation packet not frozen"})
    counts = dict(Counter(card["card_id"][0] for card in cards))
    if counts != {"K": 5, "P": 4, "O": 4, "S": 5, "R": 5, "M": 4, "V": 3, "X": 8, "B": 4}:
        raise ValueError(f"commission inventory differs: {counts}")
    write(REPO / "results/v16/COMMISSION_MANIFEST.json", {
        "commission_sha256": file_digest(source), "n_cards": len(cards), "counts": counts,
        "cards": cards, "counts_are": "inventory only, not execution, sample size or evidence"})
    write(REPO / "results/v16/K01_DESIGN.json", DESIGN)
    print(f"Filed all {len(cards)} commission cards; only separately admitted packets can execute.")


if __name__ == "__main__":
    main()
