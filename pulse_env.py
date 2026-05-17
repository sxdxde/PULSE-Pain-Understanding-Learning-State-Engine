# =============================================================================
# pulse_env.py — PulseGridWorld Custom Gymnasium Environment
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 1 & 3
# =============================================================================
# Phase 3 additions (marked ### PHASE 3 ###):
#   - SlabNetwork is instantiated inside the environment so trap cells can
#     "spike" the agent's skin directly from within step().
#   - apply_spike() bridges the environment's event system and the Slab.
#   - Trap reward now includes a growing pain BONUS penalty on top of the
#     fixed -10 base reward, making repeated trap visits progressively worse.
#   - elastic_recovery() is called at episode boundaries so the slab heals
#     slowly between episodes rather than resetting to zero.
#   - pain_history records every spike event for later analysis and plotting.
# =============================================================================

# WHY: We import gymnasium (not the older 'gym') because gymnasium is the
# actively maintained fork of OpenAI Gym. As of 2024 it's the standard used
# by all major RL libraries. The API is nearly identical but gymnasium has
# better typing, cleaner resets, and ongoing security/bug fixes.
import gymnasium as gym

# WHY: gymnasium.spaces defines the mathematical "shape" of what the agent
# observes (observation_space) and what actions it can take (action_space).
# These aren't just annotations — SB3 and other RL libraries read them at
# runtime to configure neural network input/output dimensions automatically.
from gymnasium import spaces

# WHY: numpy is the universal language of numerical RL. Observations, rewards,
# and internal state are all numpy arrays. Without numpy, nothing plugs into
# stable-baselines3 or any other RL framework.
import numpy as np

### PHASE 3 ###
# WHY torch: The SlabNetwork is a PyTorch module, so we need torch to build
# the force_vector tensors that slab.deform() expects. We only import what we
# need — no full neural network training machinery is used in this file.
import torch

# WHY: Import SlabNetwork from the separate slab.py file.
# The Slab lives in its own file because it is an agent-side construct (the
# body/skin), not a world-side construct (the environment rules). Importing
# it here couples the two only at the boundary where they must meet: the moment
# the agent enters a trap cell. Everything else stays cleanly separated.
from slab import SlabNetwork
### END PHASE 3 ###

### PHASE 4 ###
# WHY import ResistanceField here (environment file) rather than in the agent?
# Resistance is a property of SPACE, not of the agent. The viscosity of a cell
# is determined by the world's physics (where traps are, how deformed the slab
# is at that location). The environment is the correct owner because it is the
# one that knows where traps are and where the agent is stepping. The agent
# just receives the already-computed reward that includes the resistance cost.
from resistance_field import ResistanceField
### END PHASE 4 ###


class PulseGridWorld(gym.Env):
    """
    PulseGridWorld — an 8x8 discrete grid navigation task.

    WHY this class structure:
    Subclassing gym.Env forces us to implement a standard interface:
      - reset()  → start a new episode, return initial observation
      - step()   → take one action, return (obs, reward, terminated, truncated, info)
      - render() → (optional) visualise the current state

    This interface is what stable-baselines3 expects. By following it, our
    custom environment drops directly into any SB3 training loop — no glue code.
    """

    # WHY: metadata tells gymnasium what render modes this env supports.
    # "human" means print/display to screen; "rgb_array" means return pixel data.
    # We declare "human" here to keep things simple for Phase 1.
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        """
        WHY __init__:
        We define all static properties of the environment once here so that
        reset() only has to restore *state*, not redefine structure. This mirrors
        how real environments work — the rules don't change between episodes, only
        the agent's position does.
        """
        super().__init__()

        # WHY: Grid size as a constant so every method uses the same value.
        # If we later want a 10x10 grid, we change one line, not many.
        self.grid_size = 8

        # WHY: The goal is at the bottom-right corner. In matrix convention,
        # row 0 is top and row 7 is bottom, so (7, 7) is bottom-right.
        self.goal_pos = (7, 7)

        # WHY: Traps are stored as a set (not a list) for O(1) membership checks.
        # Every step, we ask "is the new position in traps?" — a set makes
        # that check instant regardless of how many traps we add later.
        self.trap_positions = {(3, 3), (4, 5), (2, 6)}

        # WHY: The agent always starts at top-left. This gives the agent a
        # consistent starting challenge — learn to navigate from (0,0) to (7,7)
        # while avoiding three traps scattered across the grid.
        self.start_pos = (0, 0)

        # --- ACTION SPACE ---
        # WHY: Discrete(4) tells gymnasium (and SB3) there are exactly 4 valid
        # integer actions: 0, 1, 2, 3. The RL algorithm will only ever sample
        # from this set during training. We map them to directions below in step().
        #   0 = up    (row decreases)
        #   1 = down  (row increases)
        #   2 = left  (col decreases)
        #   3 = right (col increases)
        self.action_space = spaces.Discrete(4)

        # --- OBSERVATION SPACE ---
        # WHY: We represent the agent's full observation as its (row, col) position.
        # This is the simplest possible observation — Phase 1 gives the agent
        # perfect state information (it always knows exactly where it is).
        # Later phases of PULSE could add partial observability or richer features.
        #
        # Box(low, high, shape, dtype) defines a continuous rectangular space.
        # Even though row and col are integers in practice, we use float32 because:
        #   1. SB3's PPO neural network expects float32 inputs
        #   2. Box with float32 is the most universally supported observation type
        # The network will learn that values like 3.0 mean "row 3", and that's fine.
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),   # minimum possible (row, col)
            high=np.array([7, 7], dtype=np.float32),  # maximum possible (row, col)
            dtype=np.float32
        )

        # WHY: We store render_mode so that step() and reset() can conditionally
        # call render(). This is the gymnasium-recommended pattern — the caller
        # declares their render intent at construction time.
        self.render_mode = render_mode

        # WHY: agent_pos starts as None to make it obvious the env hasn't been
        # reset yet. Attempting to step without resetting should fail loudly.
        self.agent_pos = None

        ### PHASE 3 ###
        # WHY instantiate the Slab inside the environment (not outside)?
        # The environment is the natural owner of the Slab in Phase 3 because:
        #   1. The Slab must persist across episodes — it accumulates damage over
        #      the agent's entire lifetime, not just one episode. Owning it here
        #      ensures it survives reset() calls.
        #   2. apply_spike() needs both the trap event (known here in step()) and
        #      the Slab (owned here). Putting both in the same class avoids passing
        #      the Slab as an argument to every external training loop.
        #   3. In Phase 4, when multiple agents share one environment, the Slab
        #      can be made into a shared attribute that all agents write to.
        # grid_size=8 and vector_dim=16 mirror the environment's grid and the
        # Slab's default skin thickness from Phase 2.
        self.slab = SlabNetwork(grid_size=self.grid_size, vector_dim=16)

        # WHY pain_history as a list (not a numpy array)?
        # pain_history grows by one entry every time a spike fires — its length
        # is not known in advance. Python lists append in O(1) amortised time.
        # A pre-allocated numpy array would require knowing the total number of
        # trap visits ahead of time, which we don't. We convert to numpy only at
        # plotting time, where we need it for matplotlib.
        #
        # Each entry is a dict so we can store multiple fields per spike without
        # inventing a fixed schema upfront: episode number, step within episode,
        # cell coordinates, and the pain score produced by that spike.
        self.pain_history = []

        # WHY: Track the current episode number so pain_history entries are
        # labelled correctly across resets. Starts at 0, incremented in reset().
        self.episode_count = 0

        # WHY: Track the current step within the episode for pain_history records.
        # This lets us reconstruct the timeline of spike events in a plot.
        self.step_count = 0
        ### END PHASE 3 ###

        ### PHASE 4 ###
        # WHY instantiate ResistanceField in __init__ (not in reset)?
        # The resistance topology is partly static (based on trap positions, which
        # never change between episodes) and partly dynamic (based on slab state,
        # which accumulates across episodes). Instantiating here and then calling
        # dynamic_resistance() at each step means:
        #   1. The static danger gradient (trap + neighbour stamps) is computed once.
        #   2. dynamic_resistance() can update the DYNAMIC component at any frequency
        #      without recreating the full field from scratch each time.
        # If we instantiated in reset(), we would throw away the accumulated dynamic
        # resistance every episode, defeating the feedback loop entirely.
        #
        # WHY pass self.trap_positions to ResistanceField?
        # The resistance field needs to know where traps are to stamp the static
        # danger gradient around them. We pass the same set used by step() so that
        # both the reward logic and the physics layer agree on trap locations.
        self.resistance_field = ResistanceField(
            grid_size=self.grid_size,
            trap_positions=self.trap_positions,
        )
        ### END PHASE 4 ###

    def reset(self, seed=None, options=None):
        """
        WHY reset():
        Every RL episode starts here. This method restores the environment to its
        initial state and returns the first observation that the agent will see.

        The 'seed' parameter is part of the gymnasium v26+ API — it lets you make
        episodes reproducible. Passing seed=42 means the same random sequence
        every run, which is critical for debugging and comparing experiments.

        'options' is a catch-all dict for environment-specific configuration.
        We don't use it here but must accept it to satisfy the gymnasium API contract.
        """
        # WHY: super().reset(seed=seed) initialises gymnasium's internal RNG
        # (self.np_random). Even though our env has no randomness right now,
        # calling super() is required by the gymnasium spec and future-proofs us
        # for when we add random trap placement in later phases.
        super().reset(seed=seed)

        ### PHASE 3 ###
        # WHY apply elastic_recovery HERE (in reset) rather than at the end of step()?
        # Gymnasium does not provide an explicit "episode ended" hook — the caller
        # simply calls reset() to start a new episode. reset() is therefore the
        # correct place to apply per-episode side effects like Slab healing.
        #
        # WHY elastic_recovery instead of slab.reset() (full wipe)?
        # Full reset would erase ALL accumulated damage between every episode,
        # meaning the Slab could never learn which cells are chronically dangerous
        # — it would treat every episode as if the agent had never been hurt before.
        # Elastic recovery (multiplicative decay) instead:
        #   1. PRESERVES TOPOLOGY: the most painful cells stay the most painful
        #      after healing — the relative danger map is intact.
        #   2. FADES MAGNITUDE: absolute pain scores shrink, so old damage matters
        #      less than recent damage. The Slab weights recency naturally.
        #   3. ALLOWS CHRONIC PAIN: cells visited repeatedly across many episodes
        #      accumulate pain faster than it decays, modelling sensitisation.
        # At decay=0.995, after 100 episodes with no new hits: 0.995^100 ≈ 0.606
        # → a cell retains 60% of its pain. After 1,000 episodes: ≈ 0.007 → fully healed.
        # This is slower than the per-step decay in visualise_slab.py (0.99^100 ≈ 0.37)
        # because episode boundaries are much coarser than individual timesteps.
        if self.episode_count > 0:
            # WHY only after episode 0? On the very first reset there is no prior
            # episode to heal from — the Slab is already all zeros. Applying decay
            # on zeros is a no-op mathematically, but skipping it makes the intent clear.
            self.slab.elastic_recovery(decay=0.995)

        # WHY: Increment AFTER healing so episode 0 is the first real episode,
        # not a phantom episode created by the initial reset() call.
        self.episode_count += 1

        # WHY: Reset per-episode step counter for pain_history timestamping.
        self.step_count = 0
        ### END PHASE 3 ###

        # WHY: Place the agent back at the starting cell. We use a list (mutable)
        # so that step() can modify it in-place. A tuple would require re-creating
        # it every step, which is wasteful and less readable.
        self.agent_pos = list(self.start_pos)

        # WHY: We return the observation as a numpy float32 array to exactly match
        # the observation_space dtype we declared above. Mismatches between
        # declared space and actual observation dtype cause silent bugs in SB3.
        observation = np.array(self.agent_pos, dtype=np.float32)

        # WHY: 'info' is a required second return value in gymnasium's reset().
        # It's a dict for extra debugging data. We return an empty dict because
        # there's nothing extra to report at reset time.
        info = {}

        # WHY: If the user requested "human" rendering, show the grid after reset
        # so they can see the starting state before training begins.
        if self.render_mode == "human":
            self.render()

        return observation, info

    ### PHASE 3 ###
    def apply_spike(self, x: int, y: int, intensity: float) -> float:
        """
        Fires a pain spike at cell (x, y): deforms the Slab and returns the
        resulting pain score. This is the bridge between the environment's
        event system (trap collision detected in step()) and the Slab's
        deformation API.

        WHY a dedicated method instead of inlining the logic in step()?
        Separation of concerns: step() decides WHEN a spike happens (game logic);
        apply_spike() decides WHAT a spike does (pain mechanics). If we later
        want different spike shapes (e.g. a Gaussian bump across neighbouring
        cells, or time-decaying intensity), we change only apply_spike() without
        touching the core step() flow.

        WHY the force_vector is uniform (all dimensions set to `intensity`)?
        In Phase 3 we treat pain as undifferentiated — we know the agent was
        hurt, but not which direction, how fast, or for how long. Broadcasting
        `intensity` to all 16 dimensions encodes "uniform distress" across every
        pain channel. In Phase 4+, different channels could carry richer signal
        (e.g. dim[0]=velocity_at_impact, dim[1]=proximity_to_wall, etc.).

        Args:
            x:         Row of the cell that fired the spike.
            y:         Column of the cell that fired the spike.
            intensity: Scalar pain strength. 1.0 = full trap hit.

        Returns:
            pain_score: L2 norm of the cell's slab vector AFTER this deformation.
                        This is the "how much does this cell hurt RIGHT NOW?" value.
                        It grows with every repeated visit because deform() is
                        additive — each spike ADDS to the existing deformation
                        rather than replacing it.
        """
        # WHY torch.ones(16) * intensity?
        # We need a PyTorch tensor because slab.deform() expects one (it adds it
        # to an nn.Parameter, which lives in PyTorch's tensor world).
        # torch.ones(16) creates [1., 1., ..., 1.] with 16 elements.
        # Multiplying by intensity scales every element: [intensity, ..., intensity].
        # This is our "uniform pain encoding" for Phase 3.
        force_vector = torch.ones(self.slab.vector_dim) * intensity

        # WHY: Apply the deformation to the Slab. This call ACCUMULATES — the
        # cell's vector grows a little with every spike. That accumulation is
        # what makes pain_score increase on repeated trap visits.
        # Mathematically: slab_vector[x,y] += lr * force_vector
        # = slab_vector[x,y] += 0.01 * [1.0, ..., 1.0]
        self.slab.deform(x, y, force_vector)

        # WHY: Read back the L2 norm AFTER deforming, not before.
        # We want the pain score to reflect THIS spike's contribution already
        # included. The L2 norm grows as the vector accumulates more force:
        #   1st visit: ||0.01 * ones_16||₂  = 0.01 * 4.0 = 0.04
        #   2nd visit: ||0.02 * ones_16||₂  = 0.02 * 4.0 = 0.08
        #   10th visit: ||0.10 * ones_16||₂ = 0.10 * 4.0 = 0.40
        # (between episodes, elastic_recovery shrinks these, so actual values
        # depend on the episode gap — but the trend is always upward with hits)
        pain_score = self.slab.deformation_depth(x, y)

        return pain_score
    ### END PHASE 3 ###

    def step(self, action):
        """
        WHY step():
        This is the heart of the environment. The RL agent calls this every
        timestep with its chosen action. We:
          1. Apply the action (move the agent)
          2. Compute the reward signal
          3. Determine if the episode is over
          4. Return all of this back to the agent

        The 5-tuple return (obs, reward, terminated, truncated, info) is the
        gymnasium v26 standard. The older gym returned 4 values — gymnasium split
        'done' into 'terminated' (natural end) vs 'truncated' (time limit hit)
        to give algorithms more information about WHY an episode ended.
        """
        # WHY: Map integer actions to (row_delta, col_delta) pairs.
        # Row increases downward (matrix convention), so "up" is -1 on the row axis.
        # This dict makes the mapping explicit and easy to extend (e.g. diagonal moves).
        action_map = {
            0: (-1,  0),   # up    → row decreases
            1: ( 1,  0),   # down  → row increases
            2: ( 0, -1),   # left  → col decreases
            3: ( 0,  1),   # right → col increases
        }

        # WHY: Unpack the delta so the arithmetic below is readable.
        row_delta, col_delta = action_map[action]

        # WHY: Compute the candidate new position before applying it.
        # We need to check boundary conditions *before* moving, not after.
        new_row = self.agent_pos[0] + row_delta
        new_col = self.agent_pos[1] + col_delta

        # WHY: Clamp the new position to [0, grid_size-1] on both axes.
        # This is the "wall collision" rule — hitting a wall keeps the agent in place.
        # np.clip is cleaner than writing four if-statements and is idiomatic numpy.
        new_row = int(np.clip(new_row, 0, self.grid_size - 1))
        new_col = int(np.clip(new_col, 0, self.grid_size - 1))

        # WHY: Now that we've validated the move, apply it.
        self.agent_pos = [new_row, new_col]

        # WHY: Convert position to a tuple for set membership checks.
        # Lists are not hashable, so `[3,3] in set(...)` would raise a TypeError.
        pos_tuple = (new_row, new_col)

        ### PHASE 3 ###
        # WHY: Increment step counter here, after the move is applied, so the
        # count reflects the step that caused this position (and any spike).
        self.step_count += 1
        ### END PHASE 3 ###

        ### PHASE 4 ###
        # WHY compute resistance cost BEFORE the reward if-chain?
        # The resistance cost applies to EVERY move — reaching the goal, stepping
        # into a trap, or moving through open space. By computing it before the
        # if-chain and initialising reward with it, we guarantee that no branch
        # below can accidentally omit the resistance cost. If we added it inside
        # each branch separately we would risk forgetting it in one branch.
        #
        # WHY add it to reward here (not subtract the absolute value)?
        # movement_cost() already returns a negative float (see resistance_field.py).
        # Adding a negative number is mathematically a subtraction — we do not
        # negate again here. This keeps the convention consistent: movement_cost()
        # is always ≤ 0, always reduces reward, regardless of which cell we entered.
        #
        # WHY does every move pay the resistance cost, including goal entry?
        # The goal cell has BASELINE_RESISTANCE (0.1) just like any open cell.
        # The +10 goal reward is so much larger that -0.1 from resistance is
        # negligible. What matters is that the API is consistent: resistance is a
        # universal physics cost, not a conditional one. Exceptions create bugs.
        #
        # WHY does this make trap NEIGHBOURS feel "thick" even before entry?
        # Neighbour cells have TRAP_RESISTANCE = 0.7, so moving into them costs
        # -0.7 instead of the baseline -0.1. The agent receives this cost every
        # step it spends in the neighbourhood — it does not need to hit the trap
        # to feel that the surrounding space is expensive. This gives the policy
        # a spatial gradient to follow: step away from high-cost cells toward
        # low-cost ones, which points naturally away from traps.
        resistance_cost = self.resistance_field.movement_cost(new_row, new_col)

        # WHY also update the dynamic resistance after every move?
        # The slab accumulates deformation as the agent takes hits. Calling
        # dynamic_resistance() here means the resistance field stays synchronised
        # with the slab's current pain state after every step. This means the
        # NEXT step into a freshly-hit cell will already see its raised resistance.
        # The feedback loop (pain → resistance → avoidance) is tight: it operates
        # at one-step latency rather than lagging by a full episode.
        self.resistance_field.dynamic_resistance(self.slab, scale=0.5)
        ### END PHASE 4 ###

        # --- REWARD AND TERMINATION LOGIC ---
        # WHY: We initialise reward and terminated before the if-chain so that
        # the code reads as: "start neutral, then check special cases."
        # This avoids missing a return path in complex future reward structures.
        terminated = False
        # WHY initialise reward to resistance_cost (not 0.0)?
        # Every cell costs something to enter. Starting from the resistance cost
        # means all branches below ADD their domain-specific signal (goal bonus,
        # trap penalty, step penalty) on top of the universal physics cost.
        # This is cleaner than adding resistance_cost at the end of every branch.
        reward = resistance_cost

        if pos_tuple == self.goal_pos:
            # WHY: +10 is a strong positive signal. The agent must clearly learn
            # that reaching the goal is the primary objective. If the reward were
            # too small, the -0.1 step penalty might dominate and the agent could
            # learn to stand still (minimising negative steps) rather than exploring.
            reward = 10.0
            # WHY: terminated=True ends the episode naturally. The RL algorithm
            # will not bootstrap a value estimate beyond this point — it knows
            # the episode is genuinely over, not just truncated by a time limit.
            terminated = True

        elif pos_tuple in self.trap_positions:
            # --- BASE PENALTY ---
            # WHY -10 as a FIXED base penalty (unchanged from Phase 1)?
            # The base penalty represents the immediate, objective consequence of
            # entering a trap — the "ouch" that every agent, naive or experienced,
            # receives on first contact. It is a property of the WORLD (the trap
            # is dangerous) not of the agent's history with that trap.
            # Keeping it fixed ensures the environment's rules don't change: a
            # trap is always worth -10 regardless of whether the agent has been
            # there before.
            base_trap_reward = -10.0

            ### PHASE 3 ###
            # --- PAIN BONUS PENALTY ---
            # WHY subtract the pain_score ON TOP of the base -10?
            # The pain_score represents the agent's ACCUMULATED SENSITISATION
            # to this specific cell. A cell visited for the first time has a
            # pain_score near 0 (barely deformed slab) → bonus penalty ≈ 0.
            # A cell visited 20+ times has a pain_score of ~1-2 → bonus penalty
            # of -1 to -2 on top of the -10 base.
            #
            # WHY does this matter for learning?
            # This creates a NON-STATIONARY REWARD SIGNAL that grows worse
            # for cells the agent has already been punished for. The RL
            # algorithm sees:
            #   - First trap visit at (3,3):  reward ≈ -10.04  (base + tiny pain)
            #   - 5th trap visit at (3,3):    reward ≈ -10.20  (base + growing pain)
            #   - 20th trap visit at (3,3):   reward ≈ -10.80  (base + chronic pain)
            #
            # The DIFFERENCE between base and bonus penalty:
            #   Base (-10):  Encodes the objective danger of the trap. Constant.
            #                "This cell kills you."
            #   Bonus (−pain_score): Encodes the subjective HISTORY of suffering
            #                at this cell. Grows with experience. "And you KNOW it."
            #
            # Together they form a two-component reward signal: the world's
            # punishment + the agent's body's memory of that punishment.
            # This is analogous to how biological pain works: stubbing the same
            # toe twice in quick succession hurts MORE the second time due to
            # peripheral sensitisation — the nociceptors are already primed.
            #
            # WHY is this useful for RL beyond just making traps "more negative"?
            # A purely stronger fixed penalty (-15 instead of -10) would penalise
            # all traps equally regardless of history. The pain bonus instead
            # creates a GRADIENT: traps the agent has never visited are slightly
            # less aversive than traps it has chronic experience with. This could
            # guide exploration — making the agent more likely to probe NEW danger
            # zones than to re-enter known painful ones. That's the Phase 4 hypothesis.
            pain_score = self.apply_spike(new_row, new_col, intensity=1.0)

            # WHY: Store the spike event in pain_history with full context.
            # We record: which episode, which step, which cell, and what score.
            # This lets us reconstruct the agent's complete pain biography later.
            self.pain_history.append({
                "episode":    self.episode_count,
                "step":       self.step_count,
                "cell":       (new_row, new_col),
                "pain_score": pain_score,
            })

            # WHY: Total reward = base penalty + pain bonus.
            # Note the sign: pain_score is always ≥ 0 (it's an L2 norm), so
            # subtracting it always makes the total reward MORE negative.
            reward = base_trap_reward - pain_score
            ### END PHASE 3 ###

            # WHY: Traps also terminate the episode. The agent "falls in" and must
            # start over. This teaches the consequence of exploration near traps.
            terminated = True

        else:
            # WHY: -0.1 per step is a small "living penalty" that discourages
            # the agent from wandering aimlessly. Without it, the agent might learn
            # to loop around the grid forever (avoiding traps but never reaching
            # the goal) because the reward signal is identical to not having moved.
            # This is a common RL trick: make time itself have a small cost.
            reward = -0.1

        # WHY: truncated=False because we have no time limit in this env.
        # If we added a max_steps limit (e.g. 200 steps per episode), we would
        # set truncated=True when the step counter hit the limit, even if the
        # episode didn't end naturally. SB3's PPO handles terminated and truncated
        # differently — truncated episodes still receive a value bootstrap.
        truncated = False

        # WHY: Return position as float32 to match observation_space dtype.
        observation = np.array(self.agent_pos, dtype=np.float32)

        ### PHASE 3 ###
        # WHY: Expose pain-related data through the info dict.
        # SB3 doesn't use info during training, but visualise_pain.py reads it
        # to know the current pain score without re-querying the slab separately.
        # Keeping it in info avoids creating an extra public attribute on the env.
        info = {
            "pain_score":    self.slab.deformation_depth(new_row, new_col),
            "episode_count": self.episode_count,
            "step_count":    self.step_count,
            ### PHASE 4 ###
            # WHY expose resistance_cost in info?
            # Visualisers and analysis scripts can track how the resistance cost
            # evolves over time as the dynamic field adapts to accumulated pain.
            # This lets us verify that the feedback loop (pain → resistance → cost)
            # is working as intended without re-querying the ResistanceField separately.
            "resistance_cost": resistance_cost,
        }
        ### END PHASE 3 / 4 ###

        # WHY: Render after every step if the user wants to watch training live.
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        WHY render():
        A simple text-based grid printout. This lets us visually inspect the
        environment during debugging without needing matplotlib. For Phase 1,
        "human" render just means printing to stdout.

        In later phases, we'd add an "rgb_array" mode that returns pixel data
        so we could record video of training runs.
        """
        # WHY: Build the grid as a 2D list of characters first, then print all at once.
        # Printing row-by-row with multiple print() calls is slower and harder to
        # extend (e.g. adding colour codes).
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # WHY: Mark each trap with "T" so it's visually distinct.
        for trap_r, trap_c in self.trap_positions:
            grid[trap_r][trap_c] = "T"

        # WHY: Mark the goal with "G".
        goal_r, goal_c = self.goal_pos
        grid[goal_r][goal_c] = "G"

        # WHY: Mark the agent last so it overwrites "." (and even "G" or "T" if
        # we want to see agent-on-goal collisions during debugging).
        if self.agent_pos is not None:
            grid[self.agent_pos[0]][self.agent_pos[1]] = "A"

        # WHY: Print a separator so successive render() calls don't blur together.
        print("\n" + "-" * (self.grid_size * 2 - 1))
        for row in grid:
            # WHY: Join with spaces so columns are readable as a grid.
            print(" ".join(row))
