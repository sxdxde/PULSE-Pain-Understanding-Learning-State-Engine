# =============================================================================
# visualise_slab.py — Slab Heatmap Visualisation
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 2
# =============================================================================
# WHY this script exists:
# The Slab's pain landscape is a 3D object (8×8 grid of 16-dim vectors), but
# humans can only intuitively read 2D. By projecting each cell's vector down
# to its L2 norm (a scalar), we get a 2D "pain heatmap" — the Slab's state
# rendered as colour. This is the Phase 2 equivalent of visualise_phase1.py's
# path plot: a diagnostic tool for understanding the agent's internal state.
# =============================================================================

# WHY matplotlib: Same reasons as Phase 1 — static, saveable, no window setup needed.
import matplotlib.pyplot as plt

# WHY matplotlib.patches: We reuse the Rectangle-per-cell drawing style from
# visualise_phase1.py so the two visualisations look consistent and could be
# overlaid if needed.
import matplotlib.patches as mpatches

# WHY matplotlib.colors / matplotlib.cm:
# We need to programmatically map pain score values (floats) to RGBA colours.
# Normalize() maps a data range to [0, 1]; ScalarMappable turns [0, 1] into RGB.
# Together they let us apply any matplotlib colormap (like "Reds") to our pain scores.
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# WHY numpy: For building the random force vectors and any array manipulation.
import numpy as np

# WHY torch: We need torch to create force_vector tensors that deform() expects.
import torch

# WHY: Import our Slab from the file we just created.
from slab import SlabNetwork


# =============================================================================
# SIMULATION PARAMETERS — all in one place at the top
# =============================================================================

# WHY: Must match PulseGridWorld's grid size so the Slab overlaps perfectly.
GRID_SIZE = 8

# WHY 16: Must match the vector_dim used when the Slab is integrated with training.
# Changing this here without changing slab.py would break the force_vector shape.
VECTOR_DIM = 16

# WHY: These are the known trap positions from pulse_env.py. We simulate damage
# at these locations to model what happens after the agent visits them multiple times.
TRAP_POSITIONS = [(3, 3), (4, 5), (2, 6)]

# WHY: Goal and start are marked on the plot for orientation — same as Phase 1 visuals.
GOAL_POS = (7, 7)
START_POS = (0, 0)

# WHY: The number of times we "hit" each trap in the simulation.
# More hits → higher pain score → darker red on the heatmap.
# 50 hits at lr=0.01 means each trap dimension accumulates ~0.5 → L2 ≈ 2.0
TRAP_HIT_COUNT = 50

# WHY: We also add some random "noise" damage to non-trap cells to simulate
# the agent bumping walls and exploring. This makes the heatmap more realistic —
# in practice, the Slab won't be completely zero everywhere except traps.
RANDOM_DAMAGE_CELLS = 6       # how many random non-trap cells get minor damage
RANDOM_DAMAGE_HITS = 10       # how many times each random cell is hit


def build_force_vector(scalar: float, dim: int = VECTOR_DIM) -> torch.Tensor:
    """
    Constructs a uniform force vector where every dimension gets the same value.

    WHY uniform force vector?
    In Phase 2 we don't yet have a rich encoding of what KIND of pain the trap
    caused — we only know that it caused pain. So we broadcast the scalar pain
    signal uniformly across all 16 dimensions. Every dimension deforms equally.

    In future phases (Phase 3+) we might encode:
      - dim[0:4]: spatial direction of the collision
      - dim[4:8]: velocity at time of impact
      - dim[8:12]: proximity to other traps (contextual risk)
      - dim[12:16]: time-since-last-pain (recency)
    Each channel would then carry distinct information, and the L2 norm would
    integrate ALL of those signals into one holistic pain score.

    Args:
        scalar: The pain intensity for this hit.
        dim:    Number of vector dimensions (must match SlabNetwork.vector_dim).

    Returns:
        A 1D tensor of shape (dim,) with every element equal to scalar.
    """
    # WHY torch.ones * scalar: Creates a (dim,) tensor of [scalar, scalar, ..., scalar].
    # This is our "uniform pain encoding" for Phase 2.
    return torch.ones(dim) * scalar


def simulate_damage(slab: SlabNetwork) -> None:
    """
    Applies simulated deformations to the slab to create a non-trivial pain landscape.

    WHY simulate damage rather than load a real trained agent's slab?
    We don't yet have coupling between the PPO agent and the Slab (that's Phase 3).
    This simulation lets us TEST and VALIDATE the Slab and its visualisation before
    wiring it into the training loop. It also gives us a controlled scenario where
    we know exactly which cells should be most painful (the traps), letting us
    verify the heatmap is correct.
    """
    print("Simulating trap damage...")

    for (row, col) in TRAP_POSITIONS:
        # WHY pain_intensity=1.0 for traps?
        # The trap reward in pulse_env.py is -10 (penalty). We normalise by 10
        # to get a unit pain signal: 1.0. Uniform across all 16 dimensions.
        # At lr=0.01 and 50 hits: each dim accumulates 50 * 0.01 * 1.0 = 0.5
        # L2 norm over 16 dims = sqrt(16 * 0.5²) = sqrt(4.0) = 2.0
        # So each trap cell will have a pain score of ~2.0 — clearly visible.
        force = build_force_vector(scalar=1.0)

        for _ in range(TRAP_HIT_COUNT):
            slab.deform(row, col, force)

        print(
            f"  Trap at ({row},{col}): "
            f"pain score = {slab.deformation_depth(row, col):.3f} "
            f"after {TRAP_HIT_COUNT} hits"
        )

    print(f"\nSimulating random minor damage on {RANDOM_DAMAGE_CELLS} cells...")

    # WHY random seed? Makes this simulation reproducible — same random cells chosen
    # every run, so we can compare heatmaps between code versions.
    rng = np.random.default_rng(seed=42)

    # WHY: Generate candidate cells, filter out trap/goal/start positions so
    # the "background noise" cells are visually separate from the key locations.
    all_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    special_cells = set(TRAP_POSITIONS) | {GOAL_POS, START_POS}
    candidate_cells = [cell for cell in all_cells if cell not in special_cells]

    # WHY replace=False: Sample without replacement so we don't pick the same
    # cell twice for random damage.
    chosen_indices = rng.choice(len(candidate_cells), size=RANDOM_DAMAGE_CELLS, replace=False)
    random_cells = [candidate_cells[i] for i in chosen_indices]

    for (row, col) in random_cells:
        # WHY scalar=0.3 for random damage?
        # Bumping a wall or exploring is much less painful than a trap (-10 reward).
        # We model it as 30% of full trap intensity. After 10 hits:
        # each dim = 10 * 0.01 * 0.3 = 0.03 → L2 = sqrt(16 * 0.03²) ≈ 0.12
        # Low, but visible as a faint pink on the heatmap.
        force = build_force_vector(scalar=0.3)

        for _ in range(RANDOM_DAMAGE_HITS):
            slab.deform(row, col, force)

        print(
            f"  Random cell ({row},{col}): "
            f"pain score = {slab.deformation_depth(row, col):.3f}"
        )


def visualise_heatmap(slab: SlabNetwork, title_suffix: str = "") -> None:
    """
    Renders the slab's current pain landscape as a colour heatmap.

    WHY per-cell Rectangle patches (not imshow)?
    imshow would give us a heatmap in fewer lines, but:
      1. We need to overlay text labels (TRAP, GOAL, pain scores) on specific cells.
      2. We want to match the visual style of visualise_phase1.py for consistency.
      3. Patches give us per-cell control — we can add borders, different edge colours,
         and mix heatmap colouring with categorical markers.
    imshow is simpler but less compositional. Patches are the right tool here.

    WHY white-to-deep-red colourmap?
    - White represents zero pain (resting, undamaged skin) — neutral, safe.
    - Deep red represents high pain — universally understood as danger/alert.
    - The gradient between them gives immediate intuition about DEGREE of pain,
      not just presence/absence. A pink cell is mildly damaged; a crimson cell
      is severely damaged.
    - We avoid blue (used for the agent path in Phase 1) and green (goal) to
      prevent confusion with the overlaid markers.

    Args:
        slab:         The SlabNetwork to read pain scores from.
        title_suffix: Optional string appended to the plot title (e.g. "after healing").
    """

    # WHY: Get the full pain map as a (8, 8) numpy array for colour mapping.
    pain = slab.pain_map()

    # WHY: figsize=(9, 8) — slightly wider than tall to make room for the colorbar.
    fig, ax = plt.subplots(figsize=(9, 8), dpi=120)

    # ==========================================================================
    # STEP 1: Set up the colour mapping (pain score → RGBA colour)
    # ==========================================================================

    # WHY "Reds" colormap?
    # matplotlib's "Reds" goes from very light pink (0) to deep crimson (1).
    # It's perceptually uniform — equal steps in data produce equal steps in
    # perceived colour change, so the viewer can estimate pain scores visually.
    # Alternatives: "hot" (black→red→yellow→white), "YlOrRd" (yellow→orange→red).
    # We prefer "Reds" because the white baseline is cleaner than black.
    cmap = cm.get_cmap("Reds")

    # WHY vmin=0, vmax anchored to max pain?
    # We want the full colour range to span from zero (white) to the highest
    # observed pain score (deep red). Anchoring vmax to the data max ensures
    # the most painful cell is always fully saturated — no wasted range.
    # If we hardcoded vmax=10, a max pain of 2.0 would look pale and misleading.
    pain_max = pain.max() if pain.max() > 0 else 1.0   # avoid division by zero

    # WHY Normalize? It maps the [vmin, vmax] data range to [0, 1] so the
    # colormap (which expects [0, 1] inputs) can convert pain scores to colours.
    normalizer = mcolors.Normalize(vmin=0, vmax=pain_max)

    # WHY ScalarMappable? It combines the normalizer and colormap into one object
    # that we can call .to_rgba(value) on for any individual cell, AND pass to
    # plt.colorbar() to generate the legend bar on the right of the plot.
    scalar_map = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # ==========================================================================
    # STEP 2: Draw each cell as a coloured Rectangle
    # ==========================================================================

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            # WHY: Flip row for matplotlib (same reason as visualise_phase1.py).
            cell_x = col
            cell_y = GRID_SIZE - 1 - row

            pain_score = pain[row, col]

            # WHY: Map this cell's pain score to an RGBA tuple.
            # scalar_map.to_rgba() uses the normalizer + cmap we set up above.
            face_color = scalar_map.to_rgba(pain_score)

            # WHY: Special cells get a distinctive border colour so they stand out
            # even when their pain colour is similar to neighbouring cells.
            if (row, col) in TRAP_POSITIONS:
                edge_color = "#CC0000"    # red border — danger landmark
                edge_width = 2.5
            elif (row, col) == GOAL_POS:
                edge_color = "#006600"    # green border — goal landmark
                edge_width = 2.5
            elif (row, col) == START_POS:
                edge_color = "#0044AA"    # blue border — start landmark
                edge_width = 2.0
            else:
                edge_color = "#BBBBBB"    # neutral grey for ordinary cells
                edge_width = 0.8

            rect = mpatches.Rectangle(
                (cell_x, cell_y),
                width=1, height=1,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=edge_width,
            )
            ax.add_patch(rect)

    # ==========================================================================
    # STEP 3: Overlay text labels on key cells
    # ==========================================================================

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_cx = col + 0.5           # centre x of cell
            cell_cy = GRID_SIZE - 1 - row + 0.5   # centre y (flipped)

            pain_score = pain[row, col]

            # WHY: Always print the pain score so the viewer can read exact values
            # instead of estimating from colour alone. Formatted to 2 decimal places.
            score_text = f"{pain_score:.2f}"

            # WHY: Choose text colour based on background brightness.
            # Dark (high pain) cells need white text; light (low pain) cells need
            # dark text. The threshold 0.5 on the normalised value (pain/pain_max)
            # approximates where the background becomes too dark for black text.
            text_color = "white" if (pain_score / pain_max) > 0.5 else "#333333"

            # WHY: Add a categorical label on top of the pain score for key cells.
            if (row, col) in TRAP_POSITIONS:
                label = "TRAP"
            elif (row, col) == GOAL_POS:
                label = "GOAL"
            elif (row, col) == START_POS:
                label = "START"
            else:
                label = None

            # WHY: Two-line text (label + score) only for special cells.
            # For ordinary cells, just the score.
            if label:
                ax.text(
                    cell_cx, cell_cy + 0.15,
                    label,
                    ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color=text_color,
                )
                ax.text(
                    cell_cx, cell_cy - 0.18,
                    score_text,
                    ha="center", va="center",
                    fontsize=7,
                    color=text_color,
                )
            else:
                ax.text(
                    cell_cx, cell_cy,
                    score_text,
                    ha="center", va="center",
                    fontsize=7,
                    color=text_color,
                )

    # ==========================================================================
    # STEP 4: Colorbar, axis labels, title
    # ==========================================================================

    # WHY: The colorbar maps colours back to pain score values so the viewer
    # can read off exact intensities from the gradient without doing mental maths.
    # ax=ax is required for correct placement when using subplots.
    plt.colorbar(
        scalar_map,
        ax=ax,
        label="Pain Score (L2 norm of slab vector)",
        fraction=0.046,    # WHY: Fraction of axes width the bar occupies.
        pad=0.04,          # WHY: Padding between plot and colorbar.
    )

    # WHY: Keep axis limits tight to the grid.
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))

    # WHY: Flip y-axis tick labels to match row-0-at-top matrix convention.
    ax.set_xticklabels([str(c) for c in range(GRID_SIZE + 1)], fontsize=8)
    ax.set_yticklabels([str(r) for r in range(GRID_SIZE, -1, -1)], fontsize=8)

    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)

    # WHY: Title includes key statistics so the figure is self-describing.
    title = (
        f"PULSE Phase 2 — Slab Pain Heatmap{title_suffix}\n"
        f"vector_dim={VECTOR_DIM} | "
        f"max pain={pain.max():.3f} | "
        f"mean pain={pain.mean():.3f} | "
        f"cells damaged={(pain > 0.01).sum()}/{GRID_SIZE**2}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN: Build → Damage → Visualise → Heal → Visualise Again
# =============================================================================

if __name__ == "__main__":

    import os

    # WHY: Save output to the same "models" folder as Phase 1 visuals.
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("PULSE Phase 2 — Slab Network Visualisation")
    print("=" * 60)

    # --- STEP 1: Initialise the Slab ---
    # WHY: Start fresh (all zeros) to see the before/after contrast clearly.
    slab = SlabNetwork(grid_size=GRID_SIZE, vector_dim=VECTOR_DIM)
    print(f"\nSlab initialised: {GRID_SIZE}×{GRID_SIZE} grid, {VECTOR_DIM} dims/cell")
    print(f"Total parameters: {GRID_SIZE * GRID_SIZE * VECTOR_DIM:,}")

    # --- STEP 2: Apply simulated damage ---
    print()
    simulate_damage(slab)

    # --- STEP 3: Print summary ---
    slab.summary()

    # --- STEP 4: Visualise BEFORE healing ---
    print("\nRendering heatmap (before healing)...")
    fig_before = visualise_heatmap(slab, title_suffix=" — Before Elastic Recovery")
    save_before = os.path.join("models", "slab_heatmap_before_healing.png")
    fig_before.savefig(save_before, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_before}")

    # --- STEP 5: Apply elastic recovery for 100 timesteps ---
    # WHY 100 steps of decay=0.99?
    # 0.99^100 ≈ 0.366 → each cell retains ~37% of its original pain.
    # This demonstrates that:
    #   1. Healing is gradual — cells don't snap to zero instantly.
    #   2. The RANKING of cells is preserved — the most painful cells remain
    #      the most painful after healing (just at lower absolute values).
    #      This is exactly the behaviour we want: scars from traps fade, but
    #      the relative danger map persists in the pain topology.
    HEALING_STEPS = 100
    print(f"\nApplying elastic recovery: {HEALING_STEPS} steps at decay=0.99...")

    for step in range(HEALING_STEPS):
        slab.elastic_recovery(decay=0.99)

    # --- STEP 6: Print summary after healing ---
    print("After healing:")
    slab.summary()

    # --- STEP 7: Visualise AFTER healing ---
    print("\nRendering heatmap (after healing)...")
    fig_after = visualise_heatmap(slab, title_suffix=" — After 100 Steps of Elastic Recovery")
    save_after = os.path.join("models", "slab_heatmap_after_healing.png")
    fig_after.savefig(save_after, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_after}")

    # --- STEP 8: Demonstrate reset_cell on one trap ---
    # WHY: Show that complete healing of a single cell works independently
    # of the global elastic recovery. This models targeted treatment.
    print(f"\nDemonstrating reset_cell on trap at (3,3)...")
    print(f"  Pain before reset: {slab.deformation_depth(3, 3):.4f}")
    slab.reset_cell(3, 3)
    print(f"  Pain after  reset: {slab.deformation_depth(3, 3):.4f}")

    # WHY: Show both plots at once so the user can compare before/after side by side.
    print("\nDisplaying plots (close to exit)...")
    plt.show()
