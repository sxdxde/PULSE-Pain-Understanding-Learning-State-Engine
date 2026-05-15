# =============================================================================
# pulse_env.py — PulseGridWorld Custom Gymnasium Environment
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 1
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

        # --- REWARD AND TERMINATION LOGIC ---
        # WHY: We initialise reward and terminated before the if-chain so that
        # the code reads as: "start neutral, then check special cases."
        # This avoids missing a return path in complex future reward structures.
        terminated = False
        reward = 0.0

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
            # WHY: -10 mirrors the magnitude of the goal reward, making traps
            # symmetrically dangerous. The agent has equal incentive to avoid
            # traps as it does to seek the goal. Asymmetric rewards (e.g. -1 for
            # traps vs +10 for goal) can produce risk-seeking behaviour.
            reward = -10.0
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

        # WHY: info dict can carry debugging data (e.g. "hit_wall", "step_count").
        # We leave it empty for Phase 1 but it's a useful hook for later analysis.
        info = {}

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
