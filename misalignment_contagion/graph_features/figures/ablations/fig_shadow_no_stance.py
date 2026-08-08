"""Shadow elicitation ablation: stance-stripped prompt vs primary."""
from ._shadow_template import build_shadow_figure


def build():
    return build_shadow_figure(
        ablation_dir="shadow_no_stance_vs_primary",
        primary_variant="own-last",
        abl_variant="no-stance",
        primary_label="Primary (stance-cued)",
        abl_label="Ablation (no stance cue)",
        figure_title="Shadow elicitation: removing the stance cue",
        out_stem="fig_shadow_no_stance",
        caption=(
            "Stage III shadow probe with the explicit stance prompt stripped. "
            "Persistence metrics (II, SRF) remain in the same direction as primary; "
            "Δshadow widens modestly because the stripped prompt no longer anchors "
            "the agent to the deliberation outcome."
        ),
    )


if __name__ == "__main__":
    build()
