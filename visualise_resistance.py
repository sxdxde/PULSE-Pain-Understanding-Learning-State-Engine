# =============================================================================
# visualise_resistance.py — Heatmap Renderer for the PULSE Resistance Field
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 4
# =============================================================================
#
# WHAT this script does:
#   Renders the ResistanceField as a colour-coded heatmap (blue=low, red=high)
#   and overlays the trap positions as X marks so you can see the "danger
#   gradient" — the spatial reach of resistance around each trap.
#
# WHY visualise the resistance field separately from the pain heatmap?
#   The slab's pain heatmap (visualise_pain.py) shows ACCUMULATED DAMAGE —
#   the history of every hit the agent has taken, growing over training.
#   The resistance heatmap shows CURRENT PHYSICS — how thick the jelly is
#   right now, including both the static trap-neighbour stamps and the dynamic
#   slab-derived component. Together, the two heatmaps tell a complete story:
#     - Pain map: "where has the agent suffered?"
#     - Resistance map: "where does the world push back right now?"
# =============================================================================

# WHY matplotlib: The de-facto visualisation library for scientific Python.
# imshow() renders 2D arrays as colour images — exactly what we need for a grid heatmap.
import matplotlib.pyplot as plt

# WHY matplotlib.patches? We use Rectangle (for cell grid lines) and
# Text annotations. patches gives us the artist primitives we need.
import matplotlib.patches as patches

# WHY numpy: For building coordinate arrays when overlaying trap markers.
import numpy as np

# WHY import ResistanceField directly? This script is a standalone visualiser —
# it constructs its own ResistanceField to show the "fresh state" topology, and
# also accepts an external field for showing the dynamic state mid-training.
from resistance_field import ResistanceField


def render_resistance_field(
    resistance_field: ResistanceField,
    trap_positions: set,
    title: str = "PULSE Phase 4 — Resistance Field",
    save_path: str = None,
) -> None:
    """
    Renders the given ResistanceField as a heatmap and overlays trap markers.

    WHY accept resistance_field as an argument rather than constructing it inside?
    This function can render BOTH the initial static field AND a mid-training
    dynamic field (after dynamic_resistance() has been called). If we constructed
    the field internally we could only ever show the initial state. Accepting it
    as an argument makes this function useful at any point in the training loop.

    Args:
        resistance_field: The ResistanceField instance to visualise.
        trap_positions:   Set of (row, col) tuples for trap overlay markers.
        title:            Plot title string shown at the top of the figure.
        save_path:        If provided, saves the figure to this path instead of
                          displaying it. Useful for headless training runs.
    """
    # WHY resistance_map() (a copy) instead of resistance_field.field directly?
    # resistance_map() returns a numpy copy, which matplotlib can safely read
    # and which cannot accidentally mutate the field's internal state.
    grid = resistance_field.resistance_map()

    # WHY figsize=(7, 6)? The grid is 8×8, plus we need space for the colour bar.
    # 7 wide gives the axes room; 6 tall keeps cells roughly square.
    fig, ax = plt.subplots(figsize=(7, 6))

    # WHY imshow with RdYlBu_r (reversed Red-Yellow-Blue)?
    # Our colour convention: BLUE = low resistance (safe, easy to move through)
    #                        RED  = high resistance (dangerous, thick jelly)
    # RdYlBu goes red→yellow→blue by default. _r reverses it to blue→yellow→red,
    # which maps low values (0.0) to blue and high values (1.0) to red — matching
    # the "blue=safe, red=dangerous" intuition we want the reader to internalise.
    # This is the same colour convention used in most neuroscience activation maps.
    img = ax.imshow(
        grid,
        cmap="RdYlBu_r",      # reversed Red-Yellow-Blue: low=blue, high=red
        vmin=0.0,              # anchor the colour scale: 0.0 always maps to blue
        vmax=1.0,              # anchor the colour scale: 1.0 always maps to red
        interpolation="nearest",  # no blurring — show crisp cell boundaries
        origin="upper",        # row 0 at the top, matching grid matrix convention
    )

    # WHY add a colour bar?
    # Without it, the viewer cannot tell what shade of red/blue corresponds to
    # what resistance value. The colour bar is the legend for the physics values.
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(             # WHY label it? Disambiguation from the pain heatmap.
        "Resistance (0.0 = free flow, 1.0 = maximum jelly)",
        fontsize=9,
    )

    # --- GRID LINES ---
    # WHY draw grid lines manually instead of relying on imshow's default?
    # imshow renders cells as pixels — it has no built-in cell boundary lines.
    # Drawing Rectangle patches gives us precise, styled grid lines that clearly
    # separate cells and make it obvious that resistance is per-cell, not continuous.
    grid_size = resistance_field.grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            # WHY Rectangle(xy=(col, row), ...)?
            # matplotlib's imshow places pixel (col, row) at data coordinates
            # (col, row). The Rectangle origin must therefore be at (col - 0.5,
            # row - 0.5) to frame that pixel. We use edgecolor only (facecolor=none)
            # so the patch border appears without hiding the heatmap underneath.
            rect = patches.Rectangle(
                (col - 0.5, row - 0.5),  # bottom-left corner of the cell patch
                1, 1,                     # width and height in data coordinates
                linewidth=0.5,            # thin lines so they don't dominate the colour
                edgecolor="white",        # white shows on both light and dark cells
                facecolor="none",         # transparent fill — don't hide the heatmap
            )
            ax.add_patch(rect)

    # --- RESISTANCE VALUE ANNOTATIONS ---
    # WHY annotate each cell with its numeric value?
    # The heatmap gives a qualitative impression; the numbers give exact values.
    # Together they let the reader verify that baseline = 0.10 and trap zone = 0.70.
    for row in range(grid_size):
        for col in range(grid_size):
            val = grid[row, col]

            # WHY choose text colour based on background brightness?
            # Dark red cells (high resistance) need white text to be readable.
            # Light blue cells (low resistance) need black text. The threshold 0.5
            # splits the [0,1] range in the middle — above 0.5 = dark background.
            text_color = "white" if val > 0.5 else "black"

            ax.text(
                col, row,           # data coordinates: (x=col, y=row)
                f"{val:.2f}",       # two decimal places: enough precision, not cluttered
                ha="center",        # horizontally centred in the cell
                va="center",        # vertically centred in the cell
                fontsize=6,         # small enough to fit in an 8×8 grid cell
                color=text_color,
                fontweight="bold",  # bold so it's legible on saturated colours
            )

    # --- TRAP MARKERS ---
    # WHY X marks for traps, separate from the resistance colour?
    # The resistance colour already shows HIGH resistance at trap cells, but
    # it doesn't distinguish "this is a trap" from "this is a heavily hit safe cell."
    # The X marker is the CATEGORICAL label (trap vs. not-trap) while the colour
    # is the QUANTITATIVE resistance value. Separating them respects Tufte's rule:
    # use visual channels for distinct types of information.
    for (tr, tc) in trap_positions:
        ax.text(
            tc, tr,     # x=column, y=row in imshow coordinates
            "✕",        # Unicode heavy multiplication sign — visually distinct
            ha="center",
            va="center",
            fontsize=18,            # large enough to be instantly visible
            color="black",
            fontweight="bold",
            alpha=0.85,             # slight transparency so resistance colour shows through
        )

    # --- AXIS LABELS AND TITLE ---
    # WHY label axes? The reader must immediately know which axis is row vs column.
    ax.set_xlabel("Column (x)", fontsize=10)
    ax.set_ylabel("Row (y)", fontsize=10)

    # WHY set tick labels explicitly?
    # imshow does not automatically label tick marks with grid indices. Setting
    # ticks at 0..7 with matching labels ties every pixel back to a cell address.
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels(range(grid_size), fontsize=8)
    ax.set_yticklabels(range(grid_size), fontsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # --- LEGEND ANNOTATION ---
    # WHY add a text annotation explaining the X symbol?
    # The X could be confused with a close button or a crossing marker.
    # An explicit legend entry removes ambiguity for new readers.
    fig.text(
        0.13, 0.01,    # figure coordinates: bottom-left area below the axes
        "✕ = Trap cell   |   Blue = low resistance   |   Red = high resistance",
        fontsize=8,
        color="dimgray",
        ha="left",
    )

    plt.tight_layout()

    if save_path:
        # WHY dpi=150? High enough for printed or zoomed display of an 8×8 grid.
        # The default 72 dpi makes the small cell text blurry when zoomed in.
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualise_resistance] Saved to {save_path}")
    else:
        # WHY plt.show()? Interactive display — the script can be run standalone
        # from the terminal to inspect the current resistance state at any time.
        plt.show()

    # WHY close the figure? plt.close() frees memory. In long training loops,
    # forgetting to close accumulates open figures until the process runs out of
    # GUI handles or RAM. Defensive close is always the right practice.
    plt.close(fig)


# =============================================================================
# MAIN — run standalone to see the initial (static) resistance topology
# =============================================================================
if __name__ == "__main__":
    # WHY hardcode the same trap_positions as pulse_env.py here?
    # This script is a standalone visualiser — it needs the trap layout to draw
    # the X markers and to construct a matching ResistanceField. In a full
    # training pipeline, you would pass the env's resistance_field directly.
    TRAP_POSITIONS = {(3, 3), (4, 5), (2, 6)}

    # WHY construct a fresh ResistanceField here (not import the env)?
    # We want to show the INITIAL static topology without starting a full training
    # run. A fresh field shows only the trap stamps; the dynamic component is zero.
    # This is the "before training" baseline — useful for understanding the physics
    # before any learning has occurred.
    rf = ResistanceField(grid_size=8, trap_positions=TRAP_POSITIONS)

    print("[visualise_resistance] Rendering initial resistance field...")
    print(f"  Baseline resistance:  {ResistanceField.BASELINE_RESISTANCE}")
    print(f"  Trap zone resistance: {ResistanceField.TRAP_RESISTANCE}")
    print(f"  Trap positions:       {sorted(TRAP_POSITIONS)}")
    print()

    # WHY print the resistance map to terminal as well as plotting?
    # Quick numerical inspection without matplotlib — useful in headless environments
    # or when debugging a CI run where opening a window is not possible.
    print("Resistance map (rows × cols):")
    print(np.round(rf.resistance_map(), 2))
    print()

    # WHY show the full static field first (no save_path → interactive)?
    render_resistance_field(
        resistance_field=rf,
        trap_positions=TRAP_POSITIONS,
        title="PULSE Phase 4 — Initial Static Resistance Field\n"
              "(blue=baseline 0.10, red=danger zone 0.70, ✕=trap)",
    )
