# =============================================================================
# resistance_field.py — Jelly Resistance Layer for the PULSE Grid World
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 4
# =============================================================================
#
# WHAT this file does:
#   Defines a ResistanceField: an 8×8 grid of scalar resistance values that
#   impose a physics-like "viscosity cost" on every move the agent makes.
#   The field is separate from the reward system — it is a PHYSICAL LAYER
#   that the reward system then consults, not a learning signal by itself.
#
# WHY resistance is separate from the reward (physics vs. learning signal):
#   Reward in RL is a scalar signal the algorithm directly optimises against.
#   Resistance is a property of SPACE — it describes how hard it is to move
#   through a region, independent of the agent's history or the algorithm
#   being used. By keeping them separate we can:
#     1. Change the resistance topology without touching the reward function.
#     2. Visualise the physical danger gradient independently of training state.
#     3. Swap reward functions (e.g. sparse vs. dense) without losing the
#        spatial resistance model underneath.
#   Think of it like friction in a physics simulator: friction is a property
#   of the floor, not a property of the score. The score uses friction, but
#   they live in different modules.
#
# WHY neighbouring cells of traps have high resistance (pre-entry signal):
#   A biological nervous system does NOT wait until a nociceptor fires to warn
#   the organism — it uses anticipatory signals (visual cues, memory, proprioceptive
#   warnings) to raise caution before the painful event. In RL terms, this
#   corresponds to giving the agent a SHAPED REWARD that decays smoothly away
#   from danger zones. Without the neighbour signal, the agent only learns "trap
#   cell = bad" — it has no gradient to steer away one step earlier. With the
#   neighbour signal, the agent experiences increasing cost as it approaches a
#   trap, giving the gradient descent something to climb before the terminal hit.
#   This is the "pre-entry signal concept": danger has a spatial reach, not just
#   a point location.
#
# Difference between jelly resistance (always there) and spike pain (fires on entry):
#   JELLY RESISTANCE: Continuous, spatial, always-active. Every move into any cell
#     incurs a resistance cost proportional to that cell's viscosity. It is graded,
#     smooth, and acts like a force field. The agent feels it before committing.
#     Analogous to nociceptive SENSITISATION — raised baseline alertness.
#   SPIKE PAIN (from slab.py): Discrete, event-triggered, fires only ON ENTRY to
#     a trap. It is a binary event (either you entered the trap or you didn't) with
#     a large negative reward. The agent cannot feel it until it is too late.
#     Analogous to an acute nociceptive RESPONSE — the actual pain signal.
#   Together: jelly resistance teaches caution; spike pain teaches avoidance.
#     The agent that has both feels DREAD near traps and AGONY inside them.
#
# WHY we want the agent to feel danger before committing to a bad state:
#   RL policy gradients need SIGNAL at every decision point to update the policy.
#   If danger only fires inside trap cells, only the one timestep of entry gets
#   a gradient — all the preceding steps that led there get no corrective signal.
#   By making the approach to danger costly, every step toward a trap carries
#   a negative gradient, not just the final fatal step. This dramatically speeds
#   convergence: instead of needing to fully commit to a trap to learn it is bad,
#   the agent receives increasing discouragement as it approaches, allowing the
#   policy to back off earlier and earlier across training iterations.
# =============================================================================

# WHY numpy: The resistance field is a 2D array of floats. numpy gives us
# efficient initialisation, clipping, and element-wise operations with no loops.
import numpy as np


class ResistanceField:
    """
    ResistanceField — an 8×8 grid of movement-viscosity values.

    Each cell holds a float in [0.0, 1.0]. When the agent moves into a cell,
    it pays a cost proportional to that cell's resistance. This makes the space
    "feel thick" near danger zones — the agent slows before it burns.
    """

    # WHY 0.1 as the baseline instead of 0.0?
    # A zero baseline would mean every step in open space is free — movement
    # has no cost except the fixed -0.1 living penalty already in step().
    # A non-zero baseline (0.1) means movement is NEVER FREE anywhere.
    # This models the "jelly world" concept: every region has some viscosity,
    # even the safe ones. The agent must always pay to exist and move, which:
    #   1. Encourages the agent to find the SHORTEST path, not just the safe path.
    #   2. Makes the contrast between safe cells (0.1) and danger neighbours (0.7)
    #      meaningful — a 7× cost increase, not an infinite-to-zero contrast.
    #   3. Prevents the agent from treating any direction as "free to explore,"
    #      which forces more deliberate movement from early training.
    BASELINE_RESISTANCE = 0.1

    # WHY 0.7 for trap cells and their neighbours?
    # 0.7 is 7× the baseline — a strong but not overwhelming signal.
    #   - Too close to baseline (e.g. 0.2): the pre-entry signal is too weak
    #     to consistently steer the policy; noise in the gradient swamps it.
    #   - Too high (e.g. 1.0): the neighbour cost equals or exceeds the trap cost,
    #     making every cell adjacent to a trap as terrifying as the trap itself.
    #     The agent might freeze rather than find a route around dangers.
    #   - 0.7 is a strong "caution zone" signal that is clearly distinct from
    #     baseline but still clearly lower than the spike pain on trap entry.
    TRAP_RESISTANCE = 0.7

    def __init__(self, grid_size: int = 8, trap_positions: set = None):
        """
        Build the resistance field for the given grid and trap layout.

        Args:
            grid_size:      Side length of the square grid (must match pulse_env.py).
            trap_positions: Set of (row, col) tuples marking trap cells.
                            If None, no traps are marked (field stays at baseline).
        """
        # WHY store grid_size? Every method that indexes into the field uses it
        # to validate bounds or compute neighbour offsets without hardcoding 8.
        self.grid_size = grid_size

        # WHY default to empty set instead of None after the guard?
        # An empty set is a safe sentinel — "in empty_set" is always False and
        # never raises a TypeError. Keeping None would force every method to
        # guard against it separately.
        self.trap_positions = trap_positions if trap_positions is not None else set()

        # WHY np.full instead of np.zeros?
        # np.full(shape, BASELINE) initialises every cell to the baseline at once.
        # np.zeros would initialise to 0.0 and require a second pass to fill in
        # the baseline — extra work for no gain.
        # dtype=np.float32 matches the observation_space dtype in pulse_env.py,
        # keeping the entire pipeline in the same numeric type.
        self.field = np.full(
            (grid_size, grid_size),  # shape: one float per grid cell
            self.BASELINE_RESISTANCE, # every cell starts at the jelly baseline
            dtype=np.float32
        )

        # WHY call _stamp_traps() in __init__?
        # The field's danger topology is determined by the trap layout, which is
        # known at construction time. We stamp it once here rather than lazily on
        # every get_resistance() call — the topology is static until dynamic_resistance
        # updates it, so there is no reason to recompute it repeatedly.
        self._stamp_traps()

    def _stamp_traps(self) -> None:
        """
        Raises resistance at trap cells AND their immediate neighbours to TRAP_RESISTANCE.

        WHY private (underscore prefix)?
        _stamp_traps() is called only during __init__ and dynamic_resistance().
        It is an internal bookkeeping step, not part of the public API. Marking
        it private tells readers "you don't need to call this; the class manages it."

        WHY stamp NEIGHBOURS, not just trap cells?
        See module-level comment: "pre-entry signal concept." The agent must feel
        the danger gradient BEFORE stepping into the trap. Stamping neighbours
        gives the policy a one-step warning before the fatal cell.

        The eight compass directions cover all adjacent cells (orthogonal + diagonal).
        WHY include diagonal neighbours?
        An agent can approach a trap diagonally in 4 of 8 possible approach
        directions. Excluding diagonal neighbours would leave blind spots — the
        agent could sidle up to a trap corner with no resistance warning.
        """
        # WHY all 8 directions (not just 4 orthogonal)?
        # See above — diagonal approaches must also feel the pre-entry resistance.
        eight_neighbours = [
            (-1, -1), (-1, 0), (-1, 1),  # row above
            ( 0, -1),          ( 0, 1),  # same row, left and right
            ( 1, -1), ( 1, 0), ( 1, 1),  # row below
        ]

        for (tr, tc) in self.trap_positions:
            # WHY stamp the trap cell itself?
            # The trap cell's resistance is separate from its spike pain.
            # The resistance cost fires ON APPROACH (before the spike fires).
            # The spike fires ON ENTRY (after the move is committed).
            # Giving the trap cell itself high resistance creates an additional
            # deterrent: even if the agent has never seen the spike, the resistance
            # cost alone makes trap cells feel more costly than empty cells.
            self.field[tr, tc] = self.TRAP_RESISTANCE

            # WHY clip to [0, grid_size-1]?
            # Traps near the grid edge have fewer than 8 neighbours. Clipping
            # with max(0, ...) and min(grid_size-1, ...) silently skips out-of-bounds
            # cells rather than raising an IndexError.
            for (dr, dc) in eight_neighbours:
                nr = tr + dr   # neighbour row
                nc = tc + dc   # neighbour column

                # WHY not use np.clip() here?
                # np.clip works on scalars but wrapping two separate ints is
                # cleaner with min/max in a guard — easier to read than
                # np.clip(nr, 0, self.grid_size - 1) twice inline.
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    # WHY only raise, never lower?
                    # A cell could be a neighbour of TWO traps — already stamped
                    # at 0.7 from the first trap. Using max() ensures we never
                    # accidentally lower a high-resistance cell when processing
                    # the second trap's neighbour list.
                    self.field[nr, nc] = max(
                        self.field[nr, nc],  # keep existing value if already high
                        self.TRAP_RESISTANCE  # raise to danger level
                    )

    def get_resistance(self, x: int, y: int) -> float:
        """
        Returns the raw resistance value at cell (x, y).

        Args:
            x: Row index.
            y: Column index.

        Returns:
            Float in [0.0, 1.0]. Higher = more viscosity.
        """
        # WHY return a Python float (not numpy float32)?
        # Downstream callers (step() in pulse_env.py, the visualiser) mostly
        # use plain Python arithmetic. .item() extracts a Python scalar, which
        # avoids subtle type-mismatch warnings when mixed with Python reward floats.
        return float(self.field[x, y])

    def movement_cost(self, x: int, y: int) -> float:
        """
        Returns the resistance at (x, y) as a NEGATIVE reward cost.

        WHY negative?
        Costs in RL rewards are negative. A movement cost of 0.7 means the
        agent should pay -0.7 for entering that cell, on top of any other
        reward components. Keeping the sign convention consistent here means
        the caller (step()) can do: reward += resistance_field.movement_cost(x, y)
        without needing to negate anything — the addition of a negative number
        is subtraction, which is what we want.

        WHY not bake the negation into get_resistance()?
        get_resistance() returns a raw physics value (unsigned, in [0, 1]).
        It is used by the visualiser, by dynamic_resistance(), and for debugging.
        Mixing the sign flip into get_resistance() would confuse every caller
        that just wants to KNOW the resistance value. Keeping it separate means:
          - get_resistance() → physics query ("how thick is the jelly here?")
          - movement_cost()  → reward query ("how much does moving here cost?")
        These are the same number, different interpretations. Two names, no ambiguity.

        Args:
            x: Row index.
            y: Column index.

        Returns:
            A float ≤ 0.0 representing the cost of entering this cell.
        """
        # WHY negate? See docstring above — RL reward convention.
        return -self.get_resistance(x, y)

    def dynamic_resistance(self, slab, scale: float = 0.5) -> None:
        """
        Updates the resistance field based on the current slab deformation depths.

        Cells where the slab has accumulated heavy deformation (i.e., the agent
        has been hurt there many times) get HIGHER resistance. This creates a
        feedback loop: pain → resistance → avoidance.

        WHY this feedback loop matters for faster convergence:
        Without dynamic_resistance(), the danger gradient is STATIC — it is stamped
        once at construction time and never changes. The agent must rely entirely on
        accumulated reward signal to learn that certain cells are bad.
        With dynamic_resistance(), the danger gradient is ADAPTIVE:
          1. Early training: agent wanders and hits a trap. Slab pain grows.
          2. dynamic_resistance() reads slab → nearby cells get higher resistance.
          3. Next time the agent approaches that region, it pays a higher movement
             cost even before touching the trap. The reward signal is now stronger
             and earlier, pointing away from the danger zone.
          4. The policy converges faster because more timesteps carry useful gradient.
        This is analogous to WOUND HYPERALGESIA in biology: tissue around an injury
        becomes more sensitive after the initial damage, making the organism protect
        the area before it is fully re-injured. The pain landscape literally GROWS
        around the wound site, pushing the organism away earlier on future approaches.
        This is not just "more negative reward near traps" — the magnitude of that
        negative reward is proportional to HOW MUCH the agent has already suffered
        there, creating a dynamic curriculum: the more the agent has been punished,
        the stronger the avoidance signal becomes.

        Args:
            slab:  A SlabNetwork instance from slab.py. We call its
                   deformation_depth(x, y) method to read per-cell pain scores.
            scale: Controls how strongly slab pain maps to additional resistance.
                   scale=0.5 means one unit of slab pain adds 0.5 to resistance.
                   Clamped so the total resistance stays in [0.0, 1.0].
        """
        # WHY loop over every cell (not just traps)?
        # Slab deformation can accumulate in any cell the agent visits, not just
        # trap cells — repeated wall bumps, repeated safe-cell traversals, etc.
        # Dynamic resistance should reflect the FULL pain landscape, not just
        # the pre-set trap topology. Cells with zero deformation add 0.0 anyway,
        # so the loop is safe to run everywhere without inflating safe cells.
        for row in range(self.grid_size):
            for col in range(self.grid_size):

                # WHY read deformation_depth() directly on each cell?
                # slab.deformation_depth(x, y) returns the L2 norm of the cell's
                # accumulated pain vector — a scalar ≥ 0. More visits to a cell
                # → higher L2 norm → more resistance added here.
                deformation = slab.deformation_depth(row, col)

                # WHY BASELINE + scale * deformation?
                # We always preserve the BASELINE_RESISTANCE as the floor —
                # even a perfectly healed cell still has slight jelly resistance.
                # scale * deformation adds proportional viscosity for accumulated pain.
                # The scaling means we can tune the strength of the feedback loop
                # without changing the baseline physics.
                new_resistance = self.BASELINE_RESISTANCE + scale * deformation

                # WHY clip to [0.0, 1.0]?
                # The TRAP_RESISTANCE is the maximum meaningful value in our design.
                # Extremely high deformation (e.g. after thousands of trap visits)
                # could push resistance above 1.0 without clipping, which would
                # produce costs greater than 1.0 — larger than the spike pain itself.
                # Clipping keeps the resistance within its designed physical range.
                new_resistance = float(np.clip(new_resistance, 0.0, 1.0))

                # WHY max() instead of direct assignment?
                # The static trap topology (set in _stamp_traps()) defines a minimum
                # danger level for trap cells and their neighbours. We never want
                # dynamic_resistance() to LOWER those floors — it should only raise
                # resistance further if the slab warrants it. Using max() ensures:
                #   - A trap cell stays at ≥ TRAP_RESISTANCE even if deformation is low.
                #   - A heavily-hit safe cell can rise above BASELINE.
                #   - We never accidentally zero out a statically-stamped danger zone.
                self.field[row, col] = max(self.field[row, col], new_resistance)

    def resistance_map(self) -> np.ndarray:
        """
        Returns a copy of the full 8×8 resistance field as a numpy array.

        WHY return a copy (not a view)?
        Returning self.field directly would let callers mutate the internal state
        without going through _stamp_traps() or dynamic_resistance(). A copy
        is a cheap safety guarantee: the visualiser can modify its copy freely.
        """
        # WHY .copy()? See docstring — prevents caller mutation of internal state.
        return self.field.copy()
