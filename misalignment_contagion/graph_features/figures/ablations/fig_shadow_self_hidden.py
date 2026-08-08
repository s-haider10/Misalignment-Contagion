"""Shadow elicitation ablation: self-hidden context vs primary.

Note: self_hidden was only run on 4 datasets (no harmbench_copyright).
"""
from ._shadow_template import build_shadow_figure


def build():
    return build_shadow_figure(
        ablation_dir="shadow_self_hidden_vs_primary",
        primary_variant="own-last",
        abl_variant="self-hidden",
        primary_label="Primary (own utterances visible)",
        abl_label="Ablation (own utterances hidden)",
        figure_title="Shadow elicitation: hiding the agent's own deliberation",
        out_stem="fig_shadow_self_hidden",
        caption=(
            "Stage III shadow probe in which the agent cannot see its own prior "
            "utterances. Persistence metrics (II, SRF) are slightly higher "
            "(agents internalize the consensus more without their own dissent in "
            "view), but the qualitative finding is unchanged. n_agents drops to "
            "80 per dataset (subset run); harmbench_copyright not run."
        ),
    )


if __name__ == "__main__":
    build()
