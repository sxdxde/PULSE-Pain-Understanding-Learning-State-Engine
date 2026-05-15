# =============================================================================
# visualise_phase1.py — Post-Training Visualisation for PulseGridWorld
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 1
# =============================================================================
# WHY this file exists:
# Training a model produces numbers, but humans understand images. This script
# runs the trained PPO agent through one episode and plots its path on the grid,
# making it immediately clear whether the agent learned a good strategy.
# =============================================================================

# WHY: matplotlib is Python's standard plotting library. pyplot is the
# high-level interface that lets us build figures with a MATLAB-like API.
import matplotlib.pyplot as plt

# WHY: matplotlib.patches lets us draw geometric shapes (rectangles, circles)
# on plots. We use Rectangle to draw grid cells with custom colours.
import matplotlib.patches as mpatches

# WHY: numpy for array operations and coordinate handling.
import numpy as np

# WHY: We import PPO to load the saved model. You must load with the same
# algorithm class that was used to save. Loading a PPO model with SAC would fail.
from stable_baselines3 import PPO

# WHY: Import our custom environment to run the agent post-training.
from pulse_env import PulseGridWorld

# WHY: os for path construction — same reason as in train_phase1.py.
import os

# WHY: sys lets us exit with a helpful message if required files are missing,
# rather than crashing with a cryptic FileNotFoundError.
import sys


# =============================================================================
# CONFIGURATION — paths must match what train_phase1.py used
# =============================================================================
# WHY: We default to the "best model" saved by EvalCallback. This is the
# checkpoint where the agent performed best during training, not necessarily
# the final weights. If you want to compare, change this to:
#   MODEL_PATH = os.path.join("models", "ppo_pulse_phase1")
MODEL_PATH = os.path.join("models", "best_ppo_pulse_phase1", "best_model")

# WHY: Grid constants are re-declared here rather than imported from pulse_env
# so this script can be read and understood independently. In a production
# codebase you'd centralise these in a config file, but for a learning project
# clarity > DRY (Don't Repeat Yourself).
GRID_SIZE = 8
GOAL_POS = (7, 7)
TRAP_POSITIONS = [(3, 3), (4, 5), (2, 6)]
START_POS = (0, 0)


def load_model(model_path):
    """
    WHY a dedicated load function:
    Separating concerns (loading vs. running vs. plotting) makes each piece
    testable and replaceable. If we switch from PPO to SAC in Phase 2, we
    only change the load function.
    """
    # WHY: Check if the model file exists before trying to load it.
    # SB3 appends ".zip" to model paths automatically — we check for that.
    full_path = model_path + ".zip"
    if not os.path.exists(full_path):
        print(f"ERROR: Model not found at '{full_path}'")
        print("Please run train_phase1.py first to generate the model.")
        # WHY: sys.exit(1) signals to the OS that the script failed (exit code 1).
        # This is cleaner than raising an exception for a user-facing script.
        sys.exit(1)

    print(f"Loading model from: {full_path}")
    # WHY: PPO.load() reconstructs the full policy network with its trained weights.
    # We don't need to pass the env here because we're only doing inference (no
    # training), so SB3 doesn't need to validate action/observation space compatibility.
    model = PPO.load(model_path)
    print("Model loaded successfully.")
    return model


def run_episode(model, max_steps=200):
    """
    WHY max_steps=200:
    A safeguard against infinite loops. If the trained agent never reaches the
    goal or a trap (unlikely but possible if training failed), this cap prevents
    the script from hanging forever. 200 steps >> 14 (optimal path length) so
    a well-trained agent will finish well before this limit.

    Returns:
        path: list of (row, col) tuples recording every position the agent visited
        outcome: string — "goal", "trap", or "timeout"
    """
    # WHY: Create a fresh environment without rendering. We record positions
    # manually and then draw them all at once in matplotlib — this is far more
    # flexible than watching the text grid scroll by.
    env = PulseGridWorld(render_mode=None)

    # WHY: Reset the environment to get the initial observation.
    # seed=42 makes this episode reproducible — same starting conditions every run.
    obs, _ = env.reset(seed=42)

    # WHY: Record the starting position immediately. The path should include
    # every cell the agent visited, including the first one.
    path = [tuple(env.agent_pos)]

    outcome = "timeout"   # WHY: default assumption; overwritten if we finish early

    print(f"\nRunning evaluation episode (max {max_steps} steps)...")

    for step in range(max_steps):
        # WHY: model.predict() runs the policy network forward pass.
        # deterministic=True means we use the GREEDY action (argmax of the
        # policy logits) rather than sampling stochastically. This gives us the
        # agent's "best guess" behaviour for visualisation purposes — stochastic
        # actions would introduce noise that makes the path harder to interpret.
        action, _ = model.predict(obs, deterministic=True)

        # WHY: Take the action in the environment.
        obs, reward, terminated, truncated, info = env.step(action)

        # WHY: Record current position AFTER the step so the path reflects
        # the positions the agent actually visited (not where it came from).
        path.append(tuple(env.agent_pos))

        # WHY: Print step details so we can manually verify the agent's logic.
        # Seeing "step 3: pos=(0,3), action=3(right), reward=-0.1" lets us
        # trace the agent's decisions and catch bugs in reward calculation.
        action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
        print(
            f"  Step {step+1:3d}: pos={tuple(env.agent_pos)}, "
            f"action={action}({action_names[int(action)]}), "
            f"reward={reward:.1f}"
        )

        if terminated:
            # WHY: Check if we ended on the goal or a trap to label the outcome.
            # This information feeds into the visualisation title.
            if tuple(env.agent_pos) == GOAL_POS:
                outcome = "goal"
                print(f"\n  Agent reached the GOAL in {step+1} steps!")
            elif tuple(env.agent_pos) in [tuple(t) for t in TRAP_POSITIONS]:
                outcome = "trap"
                print(f"\n  Agent fell into a TRAP at step {step+1}!")
            break

        if truncated:
            # WHY: truncated would happen if we had a TimeLimit wrapper on the env.
            # We don't, but it's good practice to handle it anyway.
            outcome = "timeout"
            break

    env.close()
    print(f"Episode outcome: {outcome.upper()} | Total steps: {len(path)-1}")
    return path, outcome


def visualise(path, outcome):
    """
    WHY matplotlib for visualisation (not pygame, not curses):
    matplotlib produces publication-quality static figures that can be saved
    as PNG/PDF. It's universally installed and requires no window manager setup.
    For a research project, being able to save figures is more important than
    smooth animation (which pygame would give us but at much higher complexity).
    """

    # WHY: figsize=(8, 8) creates a square figure — appropriate for a square grid.
    # dpi=120 gives crisp output on Retina/HiDPI displays.
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    # ==========================================================================
    # STEP 1: Draw the base grid (all cells as light grey squares)
    # ==========================================================================
    # WHY: We draw each cell explicitly as a Rectangle patch. This gives us
    # full control over each cell's colour, edge style, and label. An imshow()
    # approach would be simpler but harder to customise per-cell.

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # WHY: Each cell is drawn as a unit square at (col, GRID_SIZE-1-row).
            # We flip the row coordinate (GRID_SIZE-1-row) because matplotlib's
            # y-axis points UP but matrix row 0 is at the TOP of the grid.
            # Without this flip, the grid would appear upside-down.
            cell_x = col
            cell_y = GRID_SIZE - 1 - row

            # WHY: We check each special cell type in a specific priority order:
            # trap → goal → normal. This determines the cell's background colour.
            if (row, col) in TRAP_POSITIONS:
                # WHY: Traps are red — the universal colour of danger/stop.
                # alpha=0.7 keeps it visible but not overwhelming.
                face_color = "#FF4444"   # vivid red
                edge_color = "#CC0000"   # darker red border
            elif (row, col) == GOAL_POS:
                # WHY: Goal is green — the universal colour of success/go.
                face_color = "#44CC44"   # vivid green
                edge_color = "#008800"   # darker green border
            elif (row, col) == START_POS:
                # WHY: Highlight the start in light blue so it's easy to find.
                face_color = "#AADDFF"   # pale blue
                edge_color = "#5599BB"
            else:
                # WHY: Empty cells are a neutral light grey.
                face_color = "#F0F0F0"
                edge_color = "#CCCCCC"

            # WHY: Rectangle(xy, width, height) — xy is the bottom-left corner.
            # linewidth=1.5 gives a clear grid line without being too thick.
            rect = mpatches.Rectangle(
                (cell_x, cell_y),          # bottom-left corner in plot coords
                width=1, height=1,         # unit square
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.5,
            )
            ax.add_patch(rect)

    # ==========================================================================
    # STEP 2: Add text labels inside key cells
    # ==========================================================================
    # WHY: Text labels make the plot self-explanatory — you don't need a legend
    # to know which cell is which. ha/va = horizontal/vertical alignment.

    # WHY: We label every trap cell so the viewer can immediately count them.
    for (trap_r, trap_c) in TRAP_POSITIONS:
        ax.text(
            trap_c + 0.5,                          # centre of cell horizontally
            GRID_SIZE - 1 - trap_r + 0.5,          # centre of cell vertically (flipped)
            "TRAP",
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="white",
        )

    # WHY: Label the goal cell.
    ax.text(
        GOAL_POS[1] + 0.5,
        GRID_SIZE - 1 - GOAL_POS[0] + 0.5,
        "GOAL",
        ha="center", va="center",
        fontsize=9, fontweight="bold",
        color="white",
    )

    # WHY: Label the start cell.
    ax.text(
        START_POS[1] + 0.5,
        GRID_SIZE - 1 - START_POS[0] + 0.5,
        "START",
        ha="center", va="center",
        fontsize=8, fontweight="bold",
        color="#336699",
    )

    # ==========================================================================
    # STEP 3: Draw the agent's path as a blue line + dots
    # ==========================================================================
    # WHY: We separate path drawing from cell drawing so the path always appears
    # on TOP of the cell backgrounds (matplotlib draws in layering order).

    if len(path) > 1:
        # WHY: Convert path (list of row,col tuples) to separate x and y arrays
        # for matplotlib. x = col + 0.5 (cell centre), y = flipped row + 0.5.
        # Adding 0.5 centres the path in each cell rather than anchoring to corners.
        path_x = [col + 0.5 for (row, col) in path]
        path_y = [GRID_SIZE - 1 - row + 0.5 for (row, col) in path]

        # WHY: Draw the path as a connected blue line.
        # zorder=3 ensures the line renders above the cell rectangles (zorder=1).
        # alpha=0.8 makes it slightly transparent so overlapping cells are visible.
        ax.plot(
            path_x, path_y,
            color="#3366CC",      # medium blue
            linewidth=2.5,
            alpha=0.8,
            zorder=3,
            label="Agent path",
        )

        # WHY: Draw a circle at each step position so we can count exact steps
        # and see where the agent paused or reversed direction.
        ax.scatter(
            path_x, path_y,
            color="#3366CC",
            s=60,               # marker size in points²
            zorder=4,           # above the line
            alpha=0.9,
        )

        # WHY: Mark the starting position with a larger, distinct marker (a star)
        # so it's visually clear where the episode began.
        ax.scatter(
            [path_x[0]], [path_y[0]],
            color="#0033AA",    # darker blue for start marker
            s=200,
            marker="*",         # star shape for the starting point
            zorder=5,
            label="Start position",
        )

        # WHY: Mark the final position with a diamond shape to show where the
        # episode ended (goal, trap, or timeout). Colour depends on outcome.
        end_color = {
            "goal": "#008800",      # green = success
            "trap": "#CC0000",      # red = failure
            "timeout": "#FF8800",   # orange = ran out of steps
        }.get(outcome, "gray")

        ax.scatter(
            [path_x[-1]], [path_y[-1]],
            color=end_color,
            s=200,
            marker="D",             # diamond shape for end point
            zorder=5,
            label=f"End ({outcome})",
        )

        # WHY: Annotate each step with its step number so we can follow the
        # agent's trajectory in chronological order. fontsize=7 keeps it small
        # enough not to clutter the plot.
        for i, (x, y) in enumerate(zip(path_x, path_y)):
            ax.text(
                x + 0.05, y + 0.05,    # slight offset so text doesn't overlap dot
                str(i),                 # step number (0 = start)
                fontsize=6,
                color="#1a1a6e",        # dark blue for readability
                zorder=6,
            )

    # ==========================================================================
    # STEP 4: Grid coordinates and axis formatting
    # ==========================================================================
    # WHY: Set axis limits to exactly fit the grid (0 to GRID_SIZE on both axes).
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    # WHY: We want ticks at cell boundaries (0,1,2,...,8) so the grid lines up.
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))

    # WHY: Replace tick numbers with column/row indices that match our environment
    # coordinate system. This prevents confusion between "plot x=3" and "grid col=3".
    ax.set_xticklabels([str(c) for c in range(GRID_SIZE + 1)], fontsize=8)
    # WHY: y-ticks are flipped because we inverted the y-axis for matrix convention.
    ax.set_yticklabels([str(r) for r in range(GRID_SIZE, -1, -1)], fontsize=8)

    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)

    # WHY: The title includes the outcome so the figure is self-describing
    # when saved and shared. We format the step count for quick assessment.
    steps_taken = len(path) - 1   # subtract 1 because start doesn't count as a step
    title_color = {"goal": "#005500", "trap": "#880000", "timeout": "#885500"}.get(
        outcome, "black"
    )
    ax.set_title(
        f"PULSE Phase 1 — PPO Agent Path\n"
        f"Outcome: {outcome.upper()} | Steps taken: {steps_taken}",
        fontsize=13,
        fontweight="bold",
        color=title_color,
        pad=12,
    )

    # WHY: Add a legend so the viewer understands the path markers without
    # having to read the code. loc="upper right" avoids the goal cell area.
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # WHY: tight_layout() automatically adjusts subplot margins so labels and
    # title aren't clipped. Without it, long titles often get cut off on save.
    plt.tight_layout()

    # WHY: Save the figure before showing it. plt.show() can block execution
    # on some systems, and if you close the window, unsaved work is lost.
    save_path = os.path.join("models", "pulse_phase1_path.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {save_path}")

    # WHY: plt.show() opens an interactive matplotlib window so the user can
    # zoom in, pan, and inspect the path in detail. This is the final output
    # of the visualisation pipeline.
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PULSE Phase 1 — Post-Training Visualisation")
    print("=" * 60)

    # WHY: Load → run → visualise in three clear sequential steps.
    # Each step is a function call, making the script's logic scannable at a glance.

    # Step 1: Load the trained model from disk.
    model = load_model(MODEL_PATH)

    # Step 2: Run one episode with the trained policy, recording the path.
    path, outcome = run_episode(model, max_steps=200)

    # Step 3: Render the path on the grid with matplotlib.
    visualise(path, outcome)
