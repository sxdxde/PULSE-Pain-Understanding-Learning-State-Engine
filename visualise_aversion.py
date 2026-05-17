# =============================================================================
# visualise_aversion.py — Step-by-Step Aversion Diagnostics + Path Renderer
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 5
# =============================================================================
#
# WHAT this script does:
#   1. Loads a trained PPO model (Phase 5 checkpoint if available, else Phase 1).
#   2. Runs 10 evaluation episodes with PainShapedPolicy active.
#   3. At every step, prints to console:
#        - Current cell
#        - Aversion score of every adjacent (4-directional) neighbour
#        - Which actions were blocked by the filter
#        - Which action was ultimately taken
#   4. At the end of each episode, renders a matplotlib figure showing:
#        - The 8×8 grid coloured by current aversion scores
#        - The agent's path as a green line
#        - Blocked actions at each step as faded red arrows
#        - Traps as X markers, goal as a star
#
# WHY a separate visualiser script (not bundled into train_phase5.py)?
#   The visualiser is an ANALYSIS tool — it is run after training to inspect
#   the mechanism, not during training. Keeping it separate lets researchers:
#     1. Swap in any trained model without re-running training.
#     2. Adjust visualisation parameters (episode count, threshold) without
#        touching the training script.
#     3. Run it on demand: `python3 visualise_aversion.py` at any time.
# =============================================================================

# WHY matplotlib.pyplot and matplotlib.patches?
# pyplot is the top-level plotting API. patches gives us FancyArrow and
# Rectangle for drawing arrows and grid overlays on the axes.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# WHY matplotlib.colors?
# We use Normalize to map raw aversion scores to the [0,1] colour range,
# and a colormap to translate that range to RGBA values for semi-transparent
# blocked arrows.
import matplotlib.colors as mcolors

# WHY numpy? Observations, score arrays, and path coordinate arrays are all numpy.
import numpy as np

# WHY os? Model path resolution — checking which phase checkpoint exists.
import os

# WHY logging? To see PULSE block messages inline with visualisation output.
import logging

# WHY: Import our custom modules. These are the three pillars of Phase 5.
from pulse_env import PulseGridWorld
from pain_shaped_policy import PainShapedPolicy, ACTION_NAMES

# WHY stable_baselines3? We load the trained model to wrap with PainShapedPolicy.
from stable_baselines3 import PPO


# --- LOGGING ---
# WHY INFO level? We want to see every blocked-action message that
# PainShapedPolicy emits. DEBUG would also show SB3 internals — too noisy.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# WHY suppress matplotlib DEBUG messages?
# Matplotlib logs font and backend messages at DEBUG level by default, which
# would clutter the aversion output we actually care about.
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# =============================================================================
# CONFIGURATION
# =============================================================================

# WHY try the phase5 best model first, fall back to phase5 final, then phase1?
# Phase 5 training saves two checkpoints: the best-during-training and the final.
# We prefer the best checkpoint because it has the highest evaluation reward.
# Falling back to phase1 lets this visualiser run even before phase5 training.
MODEL_PATHS = [
    os.path.join("models", "best_ppo_pulse_phase5", "best_model.zip"),
    os.path.join("models", "ppo_pulse_phase5.zip"),
    os.path.join("models", "best_ppo_pulse_phase1", "best_model.zip"),
    os.path.join("models", "ppo_pulse_phase1.zip"),
]

# WHY 10 episodes? Enough to see the aversion filter fire multiple times while
# keeping console output and figure count manageable for interactive inspection.
N_EPISODES = 10

# WHY threshold=0.5? Matches the default in PainShapedPolicy.should_avoid().
# Override here to experiment without changing the class default.
AVERSION_THRESHOLD = 0.5

# WHY define trap positions here (not import from env)?
# We need them for the matplotlib overlay BEFORE creating the env, so we can
# pass them into the ResistanceField at construction time. They must match
# pulse_env.py's self.trap_positions — keep in sync by inspection.
TRAP_POSITIONS = {(3, 3), (4, 5), (2, 6)}
GOAL_POS       = (7, 7)
GRID_SIZE      = 8


def load_model() -> PPO:
    """
    Loads the best available trained PPO model from disk.

    WHY try multiple paths in order?
    Scripts are run in any order — a user might run visualise_aversion.py before
    completing phase5 training. A graceful fallback means the visualiser always
    runs on SOME model rather than crashing with a FileNotFoundError.
    """
    for path in MODEL_PATHS:
        if os.path.exists(path):
            print(f"[visualise_aversion] Loading model from: {path}")
            # WHY PPO.load (not torch.load)?
            # SB3's .load() restores the full PPO object including policy network,
            # hyperparameters, and normalisation statistics. torch.load would give
            # us a raw state_dict, which requires manually reconstructing the model.
            return PPO.load(path)

    # WHY raise a RuntimeError (not return None)?
    # Returning None would cause a cryptic AttributeError later when the caller
    # tries to use the model. A RuntimeError with a clear message tells the user
    # exactly what to do: train a model first.
    raise RuntimeError(
        "No trained model found. Run train_phase1.py or train_phase5.py first.\n"
        f"Checked paths: {MODEL_PATHS}"
    )


def print_step_diagnostics(
    step: int,
    obs: np.ndarray,
    step_info: dict,
    chosen_action: int,
) -> None:
    """
    Prints a structured console line for every step during evaluation.

    Output format:
      step=  3 pos=(2,4) | scores: UP=0.000 DOWN=0.028 LEFT=0.000 RIGHT=0.196
             BLOCKED: RIGHT(2,5) | took: DOWN

    WHY this exact format?
    - step and pos: orientation — where are we in time and space?
    - scores: the full aversion picture — what does every direction "feel like"?
    - BLOCKED: which directions were vetoed — the mechanism in action
    - took: what actually happened — the filter's outcome
    This gives a complete read of the aversion system's reasoning at every step.

    Args:
        step:         Current step number within the episode.
        obs:          Observation [row, col] as float32.
        step_info:    Info dict returned by PainShapedPolicy.select_action().
        chosen_action: The action that was actually taken.
    """
    row, col = int(obs[0]), int(obs[1])

    # WHY format scores in action order (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)?
    # Consistent ordering makes it easy to scan vertically across many steps
    # and notice when a direction's score suddenly spikes.
    scores_str = " ".join(
        f"{ACTION_NAMES[a]}={step_info['aversion_scores'][a]:.3f}"
        for a in [0, 1, 2, 3]
    )

    # WHY format blocked actions with their target cells?
    # "BLOCKED: RIGHT" is less informative than "BLOCKED: RIGHT→(3,4)".
    # Seeing the target cell makes it immediately clear WHY the action was blocked
    # without needing to mentally compute the resulting position.
    if step_info["blocked"]:
        blocked_str = ", ".join(
            f"{ACTION_NAMES[a]}→{step_info['target_cells'][a]}"
            for a in step_info["blocked"]
        )
        blocked_display = f"BLOCKED: {blocked_str}"
    else:
        blocked_display = "no blocks"

    cornered_tag = " [CORNERED]" if step_info["cornered"] else ""

    print(
        f"  step={step:3d} pos=({row},{col}) | {scores_str} | "
        f"{blocked_display} | took: {ACTION_NAMES[chosen_action]}{cornered_tag}"
    )


def render_episode(
    episode_num: int,
    path: list,           # list of (row, col) visited in order
    blocked_events: list, # list of (row, col, blocked_action_int) per step
    aversion_grid: np.ndarray,  # 8×8 of aversion scores at episode end
    save_prefix: str = None,
) -> None:
    """
    Renders one episode as a matplotlib figure.

    WHY three visual layers?
    1. BACKGROUND (aversion heatmap): "where was the danger landscape?"
       The grid is coloured by aversion scores — blue=safe, red=accumulated danger.
       This shows the dynamically evolving danger topology that the filter consulted.
    2. PATH (green line): "where did the agent actually go?"
       Overlaid on the heatmap so we can see whether the agent navigated safely
       through the danger gradient.
    3. BLOCKED ARROWS (faded red): "where were actions vetoed?"
       Short red arrows from each position where a block occurred, pointing in the
       blocked direction. These are faded (alpha) to show they were NOT taken —
       they are counterfactual paths, not actual movement.

    Args:
        episode_num:   Episode number for the figure title.
        path:          Ordered list of (row, col) positions, including start.
        blocked_events: List of (row, col, action_int) for each blocked action.
        aversion_grid: 8×8 numpy array of aversion scores at episode end.
        save_prefix:   If given, saves the figure as "{save_prefix}_ep{n}.png".
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # --- LAYER 1: Aversion heatmap background ---
    # WHY RdYlBu_r? Reversed Red-Yellow-Blue maps 0=blue (safe) → 1=red (danger).
    # Consistent with visualise_resistance.py — readers build one colour intuition.
    # WHY vmax=0.6? Aversion scores at threshold=0.5 cluster below 0.6.
    # Setting vmax=0.6 spreads the colour range over the interesting region
    # so near-threshold cells appear clearly yellow/orange rather than faint blue.
    im = ax.imshow(
        aversion_grid,
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=0.6,
        interpolation="nearest",
        origin="upper",
        alpha=0.7,          # WHY 0.7 alpha? Path and arrows need to be visible on top.
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Aversion Score (depth × resistance)", fontsize=8)

    # WHY draw grid lines? The 8×8 cell boundaries make it clear this is a
    # discrete grid, not a continuous field. Without them, the heatmap looks
    # like a continuous colour image.
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ax.add_patch(mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=0.4, edgecolor="white", facecolor="none",
            ))

    # --- TRAP MARKERS ---
    # WHY X markers? Traps are CATEGORICAL labels (this cell is a trap), not
    # aversion scores (which are on a continuous scale shown by the heatmap).
    # Using a symbol keeps the two data channels visually distinct.
    for (tr, tc) in TRAP_POSITIONS:
        ax.text(
            tc, tr, "✕",
            ha="center", va="center",
            fontsize=20, color="black", fontweight="bold", alpha=0.9,
        )

    # --- GOAL MARKER ---
    # WHY a star? Stars are universally understood as "destination" or "prize."
    goal_r, goal_c = GOAL_POS
    ax.text(
        goal_c, goal_r, "★",
        ha="center", va="center",
        fontsize=20, color="gold", fontweight="bold",
    )

    # --- LAYER 2: Agent path as green line + step numbers ---
    if len(path) > 1:
        # WHY unpack cols for x and rows for y?
        # imshow's x axis corresponds to COLUMNS and y axis to ROWS.
        # Plotting (col, row) rather than (row, col) aligns the path with
        # the heatmap cells correctly.
        path_cols = [p[1] for p in path]
        path_rows = [p[0] for p in path]

        ax.plot(
            path_cols, path_rows,
            color="limegreen",
            linewidth=2.5,
            zorder=3,           # WHY zorder=3? Draw path above heatmap (z=0) and
                                # above blocked arrows (z=2), below text (z=4).
            alpha=0.9,
        )

        # WHY a dot at the start?
        # The start cell (0,0) is where the episode begins — marking it makes
        # the path direction unambiguous.
        ax.plot(path_cols[0], path_rows[0], "o", color="limegreen",
                markersize=8, zorder=4)

        # WHY step number annotations?
        # The path line shows WHERE the agent went; the step numbers show WHEN.
        # Together they reconstruct the temporal sequence of decisions.
        for step_idx, (pr, pc) in enumerate(path[:-1]):  # skip last (terminal)
            ax.text(
                pc + 0.05, pr - 0.25,   # slight offset so numbers don't cover the line
                str(step_idx + 1),
                fontsize=5.5,
                color="darkgreen",
                zorder=5,
                fontweight="bold",
            )

    # WHY mark the final position specially?
    # The last cell in the path is the TERMINAL state — goal or trap.
    # A distinct marker makes it immediately clear how the episode ended.
    if path:
        final_r, final_c = path[-1]
        final_color = "gold" if path[-1] == GOAL_POS else "red"
        ax.plot(final_c, final_r, "*", color=final_color, markersize=14, zorder=5)

    # --- LAYER 3: Blocked actions as faded red arrows ---
    # WHY faded (alpha=0.45)? Blocked actions did NOT happen — they are
    # counterfactuals. Fading them shows "this direction was considered but
    # refused" without suggesting the agent actually moved that way.
    #
    # WHY short arrows (0.35 length)?
    # Full-length arrows (0.5) would point to the centre of the next cell,
    # which is misleading for blocked actions (the agent never got there).
    # Short arrows originate from the current cell and point in the blocked
    # direction without reaching the target — visually suggesting "started
    # to go this way, pulled back."
    action_deltas = {
        0: (-0.35,  0.00),   # up    (row decreases → y-direction in plot goes up)
        1: ( 0.35,  0.00),   # down
        2: ( 0.00, -0.35),   # left
        3: ( 0.00,  0.35),   # right
    }

    for (br, bc, blocked_action) in blocked_events:
        dy, dx = action_deltas[blocked_action]
        # WHY annotate with arrowprops instead of ax.arrow?
        # ax.arrow's head size is in data coordinates and doesn't scale well.
        # annotate with arrowprops uses a FancyArrow which auto-scales the head
        # and looks cleaner on the grid.
        ax.annotate(
            "",                                 # no text — pure arrow
            xy=(bc + dx, br + dy),              # arrow tip
            xytext=(bc, br),                    # arrow base (current cell centre)
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                alpha=0.45,                     # faded: counterfactual direction
                lw=2.0,
            ),
            zorder=2,
        )

    # --- AXIS LABELS AND TITLE ---
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE), fontsize=8)
    ax.set_yticklabels(range(GRID_SIZE), fontsize=8)
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row",    fontsize=9)

    final_pos = path[-1] if path else "?"
    reached = "GOAL ★" if final_pos == GOAL_POS else f"TRAP ✕ at {final_pos}"
    ax.set_title(
        f"PULSE Phase 5 — Episode {episode_num} Path\n"
        f"Reached: {reached}  |  {len(path)-1} steps  |  "
        f"threshold={AVERSION_THRESHOLD}",
        fontsize=10,
        fontweight="bold",
    )

    # WHY a legend entry for blocked arrows?
    # The faded red arrows might be confused with other visual elements without
    # an explicit label.
    red_patch  = mpatches.Patch(color="red",      alpha=0.45, label="Blocked action")
    green_line = mpatches.Patch(color="limegreen", label="Agent path")
    ax.legend(
        handles=[green_line, red_patch],
        loc="lower left",
        fontsize=8,
        framealpha=0.8,
    )

    plt.tight_layout()

    if save_prefix:
        fname = f"{save_prefix}_ep{episode_num:02d}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight")
        print(f"[visualise_aversion] Saved figure: {fname}")
    else:
        plt.show()

    # WHY close after every episode?
    # 10 episodes × 1 figure each = 10 open figures without close().
    # Each figure consumes memory. Closing immediately frees resources.
    plt.close(fig)


def run_visualisation(save_figures: bool = False) -> None:
    """
    Main evaluation + visualisation loop.

    Args:
        save_figures: If True, saves PNGs to disk instead of showing interactively.
                      Set to True for headless environments or batch analysis.
    """
    # WHY load the model before creating the env?
    # If no model exists, we fail fast with a clear message rather than
    # running episodes that can't use a policy.
    model = load_model()

    # WHY a single env for all episodes (not reset between episodes)?
    # The slab accumulates pain across episodes — that's the PULSE design.
    # Using one env means pain from episode 1's trap hits are still in the slab
    # when episode 2 begins, just like the training scenario. The aversion filter
    # should get stronger as the evaluation progresses.
    env = PulseGridWorld(render_mode=None)
    obs, _ = env.reset()

    # WHY create the policy after the env (not before)?
    # PainShapedPolicy holds REFERENCES to env.slab and env.resistance_field.
    # These objects must exist before we can pass them to the policy constructor.
    pulse_policy = PainShapedPolicy(
        model=model,
        slab=env.slab,
        resistance_field=env.resistance_field,
        grid_size=env.grid_size,
    )

    print("=" * 65)
    print(f"PULSE Phase 5 — Aversion Visualiser ({N_EPISODES} episodes)")
    print(f"Aversion threshold: {AVERSION_THRESHOLD}")
    print("=" * 65)

    save_prefix = "aversion_episode" if save_figures else None

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        step = 0

        # WHY record the path as a list of (row, col)?
        # We need the full path for the trajectory overlay. Each element is one
        # position visited, in chronological order.
        path = [(int(obs[0]), int(obs[1]))]

        # WHY a separate list for blocked events (not just path)?
        # Blocked events include the DIRECTION that was vetoed — information that
        # isn't in the path (the path only shows where the agent went, not where
        # it was stopped from going). We need both to draw the arrows correctly.
        blocked_events = []   # list of (row, col, blocked_action_int)

        print(f"\n{'─'*65}")
        print(f"Episode {ep}/{N_EPISODES}")
        print(f"{'─'*65}")

        total_reward = 0.0

        while not done:
            # WHY select_action (not model.predict)?
            # The whole point of this script is to observe the aversion filter.
            # Using model.predict would bypass the filter entirely.
            action, step_info = pulse_policy.select_action(
                obs, threshold=AVERSION_THRESHOLD
            )

            # WHY print diagnostics before stepping?
            # We print BEFORE stepping so the log shows "I am at (row,col),
            # considering these aversions, blocking these, taking this action."
            # Printing after would show the NEXT position, losing the pre-decision context.
            print_step_diagnostics(step + 1, obs, step_info, action)

            # WHY record blocked events at the CURRENT position (before step)?
            # The blocked arrow must be drawn FROM the current cell, pointing in
            # the blocked direction. The current cell is obs before stepping.
            current_row, current_col = int(obs[0]), int(obs[1])
            for blocked_action in step_info["blocked"]:
                blocked_events.append((current_row, current_col, blocked_action))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # WHY append AFTER step? The path records where the agent IS,
            # not where it was planning to go. Post-step obs is the new position.
            path.append((int(obs[0]), int(obs[1])))
            done = terminated or truncated

        # --- Build aversion grid for visualisation at this episode's end ---
        # WHY build the grid from scratch at episode end (not during the loop)?
        # The aversion scores change every step as dynamic_resistance() runs.
        # We visualise the state at the END of the episode — the fully updated
        # landscape after all the pain accumulated this episode.
        aversion_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                aversion_grid[r, c] = pulse_policy.aversion_score(r, c)

        final_pos = path[-1]
        outcome = "GOAL" if final_pos == GOAL_POS else f"TRAP{final_pos}"
        print(f"  → {outcome} | steps={step} | reward={total_reward:.2f} | "
              f"blocks this episode={sum(1 for e in blocked_events)}")

        # WHY render after every episode (not at the end of all episodes)?
        # Rendering episode-by-episode lets the user inspect early episodes
        # and close figures before the later ones appear — better for interactive use.
        render_episode(
            episode_num=ep,
            path=path,
            blocked_events=blocked_events,
            aversion_grid=aversion_grid,
            save_prefix=save_prefix,
        )

    # WHY print stats at the very end?
    # Aggregate statistics summarise what the individual episode prints showed —
    # the filter's overall activity rate across all 10 episodes.
    pulse_policy.print_stats()
    env.close()


if __name__ == "__main__":
    # WHY pass save_figures=False (interactive) as the default?
    # Most users running this script want to SEE the figures immediately.
    # They can flip to save_figures=True for headless servers or batch runs.
    run_visualisation(save_figures=False)
