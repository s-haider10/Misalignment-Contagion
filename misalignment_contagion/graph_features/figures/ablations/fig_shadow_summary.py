"""Shadow elicitation ablation: agent-generated summary vs own-last utterance."""
from ._shadow_template import build_shadow_figure


def build():
    return build_shadow_figure(
        ablation_dir="shadow_summary_vs_primary",
        primary_variant="own-last",
        abl_variant="summary",
        primary_label="Primary (own-last utterance)",
        abl_label="Ablation (LLM-generated summary)",
        figure_title="Shadow elicitation: summary substitution",
        out_stem="fig_shadow_summary",
        caption=(
            "Stage III shadow probe substitutes an LLM-generated deliberation summary "
            "for the agent's own last utterance. Headline persistence metrics (II, SRF) "
            "and the public–private gap (Δshadow) shift modestly, consistent with the "
            "summary nudging shadow EV toward the public consensus."
        ),
    )


if __name__ == "__main__":
    build()
