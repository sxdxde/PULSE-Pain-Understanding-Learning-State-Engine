# =============================================================================
# pain_shaped_policy.py — Pre-Entry Aversion Wrapper for a Trained PPO Agent
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 5
# =============================================================================
#
# CORE IDEA:
#   A trained PPO policy suggests actions. This wrapper intercepts those
#   suggestions and vetos any action whose target cell has a high AVERSION SCORE
#   — a joint signal combining the agent's pain MEMORY (slab deformation) and
#   the physical DANGER of the space (resistance field). If the vetoed cell was
#   going to be painful, the agent deflects to the safest remaining option.
#   If every direction is high-aversion (cornered), the agent takes the least bad.
#
# WHY we apply aversion at EVALUATION, not during training:
#   At training time, PPO collects experience and computes policy gradients.
#   It assumes the policy it used to collect data is the same policy being
#   updated (the "on-policy" assumption). If we inject the aversion filter during
#   training, three things break:
#
#     1. BIASED EXPERIENCE COLLECTION: The aversion filter prevents certain
#        transitions from ever being collected. The PPO gradient never sees
#        "what happens when you enter a high-aversion cell" — it can't learn
#        whether the filter is adding real value, or whether those cells are
#        actually safe now (e.g. if the slab has healed).
#
#     2. NON-STATIONARITY: The filter changes which actions are available based
#        on the slab state, which changes across episodes. From PPO's perspective,
#        the action space is suddenly dynamic and episode-dependent — the value
#        function can't be estimated consistently.
#
#     3. CONFLATED SIGNALS: The PPO reward already encodes resistance and pain
#        (Phases 3–4). Applying the filter on top during training double-penalises
#        the same signal: once in the reward, once in the action distribution.
#        This makes the policy gradient noisy and hard to interpret.
#
#   At evaluation time, none of these matter:
#     - We're not updating any weights. No gradient, no on-policy assumption.
#     - The filter is a DEPLOYMENT-TIME guard, like a reflex arc that sits between
#       the brain's planned action and the muscle's execution. It doesn't re-train
#       the brain; it catches the worst decisions before they execute.
#     - We can compare vanilla vs. PULSE evaluation on the SAME trained model,
#       giving a clean ablation of the aversion mechanism's value.
#
# WHY aversion_score = depth × resistance (not depth + resistance, or just depth):
#   Each factor contributes a distinct dimension of evidence:
#
#   DEPTH (slab deformation): "Has THIS cell caused me pain before?"
#     This is MEMORY — it encodes the agent's personal history with this cell.
#     A cell with depth=0 has never been visited painfully, regardless of whether
#     it is a trap cell in theory. The agent has no embodied reason to dread it yet.
#     Depth alone would flag any cell the agent has touched, including safe cells
#     repeatedly traversed on the optimal path (those accumulate small deformations
#     from normal visits).
#
#   RESISTANCE (field value): "Is this cell physically dangerous right now?"
#     This is PHYSICS — the static danger topology (trap + neighbour stamps) plus
#     the dynamic component updated by slab feedback. It encodes the world's
#     structure, not the agent's history. A cell with resistance=0.1 is objectively
#     safe; resistance=0.7 flags a known danger zone.
#
#   PRODUCT (depth × resistance): "Both are high — this is a confirmed danger zone
#     that I have personally been hurt in."
#     The product acts as a logical AND:
#       - depth=0, resistance=0.7: trap zone but never experienced → score=0. OK to explore.
#       - depth=0.4, resistance=0.1: mild pain history in a safe zone → score=0.04. OK.
#       - depth=0.4, resistance=0.7: pain history IN a danger zone → score=0.28. Caution.
#       - depth=0.8, resistance=0.7: chronic pain in confirmed danger → score=0.56. BLOCK.
#     Neither factor alone is sufficient. Both must be elevated to trigger avoidance.
#     This prevents false positives (avoiding safe cells just because they were
#     traversed a lot) and false negatives (ignoring trap zones with no history yet).
#
# PHILOSOPHICAL DISTINCTION: pain as memory vs pain as reflex:
#   PAIN AS REFLEX (classical avoidance):
#     Any cell marked as a trap is avoided, always, unconditionally. This is a
#     lookup table — the environment's ground truth is baked into the policy.
#     Fast, but inflexible. Doesn't generalise to novel danger.
#
#   PAIN AS MEMORY (this module):
#     The agent avoids cells where it has ACCUMULATED SUFFERING, weighted by the
#     current physical danger level. It does not know trap locations — it infers
#     them from its own body's history. A cell that has never hurt it is not avoided,
#     even if it is "a trap" by the environment's rules.
#     Slower to trigger (requires experience), but genuinely learned. It generalises:
#     if new trap cells are added at runtime, the agent will learn to avoid them
#     through felt experience, not by being told about them. And if trap cells are
#     removed, the aversion fades as the slab heals — no manual update needed.
#     This is the PULSE hypothesis: pain memory IS the avoidance mechanism.
# =============================================================================

# WHY numpy: Used for array operations on observations and for argmax/argmin
# over action score vectors.
import numpy as np

# WHY torch: We need to convert observations to tensors to call model.policy
# methods that compute action probability distributions.
import torch

# WHY logging: We want structured, levelled output about blocked actions.
# Using the standard logging module (not bare print) means callers can silence
# or redirect PULSE messages independently of their own output.
import logging

# WHY: Configure a named logger so PULSE messages are clearly attributed.
# This is visible in console as "[PULSE] ..." and can be filtered by log level.
logger = logging.getLogger("PULSE.aversion")


# --- ACTION SPACE CONSTANTS ---
# WHY define these here (not import from pulse_env)?
# PainShapedPolicy is an evaluation-time wrapper, not an environment subclass.
# Importing the environment just for its action_map would create a circular
# dependency risk and couple this policy module tightly to one environment class.
# The action map is simple enough to replicate as a module-level constant.
# If the environment's action map ever changes, this must be updated to match.
ACTION_MAP = {
    0: (-1,  0),   # up    → row decreases
    1: ( 1,  0),   # down  → row increases
    2: ( 0, -1),   # left  → col decreases
    3: ( 0,  1),   # right → col increases
}

# WHY a human-readable name for each action?
# Log messages like "Action 2 BLOCKED" are opaque. "Action LEFT BLOCKED" is
# immediately understandable without looking up the action map.
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class PainShapedPolicy:
    """
    Evaluation-time wrapper that adds pain-memory aversion to a trained PPO policy.

    The PPO model provides action PREFERENCES (probability distribution over actions).
    This wrapper provides action VETOES (which actions are too dangerous to take).
    The final chosen action is the PPO-preferred action from the non-vetoed set.

    WHY a wrapper (not a subclass of the PPO model)?
    SB3's PPO model is a complex object with many internal components (policy net,
    value net, optimizer, rollout buffer). Subclassing it to add aversion logic
    would risk breaking its internal training mechanics. A wrapper that holds a
    reference to the model and calls its methods is safer, cleaner, and keeps
    the concerns separate: "what the PPO brain prefers" vs "what the pain body
    allows."
    """

    def __init__(
        self,
        model,               # trained SB3 PPO model
        slab,                # SlabNetwork instance (the agent's pain memory)
        resistance_field,    # ResistanceField instance (the physics danger layer)
        grid_size: int = 8,  # must match pulse_env.py
    ):
        """
        Args:
            model:            Trained SB3 PPO model. Must expose model.policy.
            slab:             SlabNetwork from slab.py. Carries pain history.
            resistance_field: ResistanceField from resistance_field.py.
            grid_size:        Side length of the square grid. Default 8.
        """
        # WHY store all three components?
        # Each component answers a different question:
        #   model           → "what does the learned policy want to do?"
        #   slab            → "where has my body been hurt?"
        #   resistance_field → "where is the world physically dangerous?"
        # Together, they constitute the full PULSE decision-making triad.
        self.model = model
        self.slab = slab
        self.resistance_field = resistance_field
        self.grid_size = grid_size

        # WHY track statistics?
        # Over many evaluation episodes, we want to quantify how often the filter
        # fires. If blocks_count stays near 0, the threshold may be too high.
        # If it matches total_actions closely, the threshold is too low and the
        # agent is over-constrained. This self-monitoring is essential for tuning.
        self.total_actions = 0      # total select_action() calls made
        self.blocks_count = 0       # total individual action vetoes applied
        self.corners_count = 0      # times ALL actions were high-aversion

    def _resolve_action(self, row: int, col: int, action: int):
        """
        Computes the resulting position after taking `action` from (row, col).

        WHY handle wall collisions here (clamp) rather than just adding deltas?
        The environment clamps the position to [0, grid_size-1] when hitting walls.
        If we didn't clamp here, we would compute out-of-bounds target cells and
        score them incorrectly (or get an index error from the slab/resistance_field).
        By mirroring the environment's clamp logic, we guarantee that the target
        cell we score is the same cell the environment would actually land on.

        Args:
            row:    Current agent row.
            col:    Current agent column.
            action: Integer action (0-3).

        Returns:
            (new_row, new_col): The position the agent would reach after this action.
        """
        # WHY look up from ACTION_MAP?
        # Mirrors pulse_env.py's action handling exactly. Any delta arithmetic
        # beyond this dict lookup should be identical to the environment's step().
        dr, dc = ACTION_MAP[action]

        # WHY np.clip (not max/min)?
        # np.clip is idiomatic numpy and reads clearly as "clamp to range."
        # We mirror pulse_env.py line for line here — same clip, same bounds.
        new_row = int(np.clip(row + dr, 0, self.grid_size - 1))
        new_col = int(np.clip(col + dc, 0, self.grid_size - 1))

        return new_row, new_col

    def aversion_score(self, x: int, y: int) -> float:
        """
        Computes the raw aversion score for cell (x, y).

        aversion_score = deformation_depth(x,y) × resistance(x,y)

        See module-level docstring for the detailed rationale behind the product
        formula. The short version: depth is memory, resistance is physics, and
        their product is "confirmed danger with personal history."

        Args:
            x: Row index.
            y: Column index.

        Returns:
            Float ≥ 0. Larger values mean "more reason to avoid this cell."
        """
        # WHY query the slab directly (not via the environment)?
        # At evaluation time, the environment's slab IS the same object we hold
        # a reference to — it was accumulated during (or prior to) evaluation.
        # Querying it directly gives the current pain state without going through
        # the environment's step() machinery.
        depth = self.slab.deformation_depth(x, y)

        # WHY get_resistance (not movement_cost)?
        # movement_cost returns a NEGATIVE float — it's designed for reward arithmetic.
        # Here we want the raw physics value in [0, 1] for the product formula.
        # Using a negative resistance in the product would give a negative
        # aversion_score, which would never exceed a positive threshold. Always
        # use the unsigned physics layer for scalar comparisons.
        resistance = self.resistance_field.get_resistance(x, y)

        # WHY the product? See module docstring — both conditions must be elevated.
        return depth * resistance

    def should_avoid(self, x: int, y: int, threshold: float = 0.5) -> bool:
        """
        Returns True if cell (x, y) should be avoided given current pain state.

        WHY threshold=0.5 as the default?
        With slab.lr=0.01 and 16-dim vectors, one trap hit gives depth ≈ 0.04.
        A trap cell has resistance=0.7. So:
          - After 1  hit: score ≈ 0.04 × 0.7 = 0.028  → not blocked
          - After 10 hits: score ≈ 0.40 × 0.7 = 0.280 → not blocked
          - After 18 hits: score ≈ 0.72 × 0.7 = 0.504 → BLOCKED (crosses 0.5)
        0.5 requires substantial personal history (~18 visits) before blocking.
        This is intentionally conservative: the agent must KNOW from experience
        that a cell is dangerous, not just theoretically. It prevents the filter
        from blocking cells the agent has barely encountered.
        The threshold is exposed as a parameter so callers can tune it:
          - threshold=0.1: aggressive early avoidance (blocks after ~4 trap hits)
          - threshold=0.5: conservative avoidance (blocks after ~18 trap hits)
          - threshold=1.0: very permissive (blocks only chronically painful cells)

        Args:
            x:         Row index of the target cell.
            y:         Column index of the target cell.
            threshold: Score above which avoidance triggers. Default 0.5.

        Returns:
            True if aversion_score(x, y) > threshold (agent should refuse).
        """
        # WHY > and not >=?
        # The threshold is a soft boundary, not a hard cliff. Using strict >
        # means a score exactly equal to threshold is treated as borderline-safe,
        # which is the more permissive (and thus less likely to over-block) choice.
        return self.aversion_score(x, y) > threshold

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        Queries the PPO policy's action probability distribution for `obs`.

        Returns a (4,) float32 array where entry i is P(action=i | obs) under
        the trained PPO stochastic policy.

        WHY use the STOCHASTIC policy (not deterministic argmax)?
        We use the full probability distribution (not just argmax) because we
        want to RANK remaining safe actions by preference, not just pick one.
        With argmax, if the top action is blocked, we'd need a second call to
        find the fallback. With probs, we can rank all safe actions in one pass:
          safe_actions.sort(key=lambda a: probs[a], reverse=True)
        This is more efficient and preserves the policy's full preference ordering.

        Args:
            obs: numpy array of shape (2,) — [row, col] as float32.

        Returns:
            numpy array of shape (4,) — probability of each action.
        """
        # WHY obs_to_tensor (not torch.FloatTensor directly)?
        # SB3's obs_to_tensor handles any pre-processing the policy expects
        # (normalization, reshaping, device placement). For our simple float32
        # observation it is a no-op in practice, but using the official method
        # future-proofs against environments with more complex pre-processing.
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)

        # WHY torch.no_grad()?
        # We are querying the policy for inference only — no gradient computation
        # is needed. no_grad() saves memory and speeds up the forward pass.
        with torch.no_grad():
            # WHY get_distribution (not predict)?
            # predict() returns the chosen action (a scalar), not probabilities.
            # get_distribution() returns the full CategoricalDistribution object,
            # from which we extract .probs — the softmax output over all 4 actions.
            distribution = self.model.policy.get_distribution(obs_tensor)

            # WHY .squeeze(0)?
            # obs_tensor has shape (1, 2) — batch dimension of 1. The distribution's
            # .probs also has shape (1, 4). squeeze(0) removes the batch dimension,
            # giving us a plain (4,) tensor for indexing by action int.
            probs = distribution.distribution.probs.squeeze(0).cpu().numpy()

        return probs  # shape: (4,), sums to 1.0

    def select_action(self, obs: np.ndarray, candidate_actions=None, threshold: float = 0.5):
        """
        Main entry point: returns the best action from the non-avoided candidates.

        Algorithm:
          1. For each candidate action, compute the target cell.
          2. Score each target cell with aversion_score().
          3. Separate actions into SAFE (score ≤ threshold) and BLOCKED (score > threshold).
          4. If safe actions exist: return the safe action with highest PPO probability.
          5. If all actions are blocked (cornered): return the action with lowest aversion.
          6. Log all blocked actions regardless of which case applies.

        WHY the agent should NEVER be hard-blocked (always has a fallback):
          Hard-blocking (refusing to move at all) would freeze the agent permanently
          in any situation where all neighbours are high-aversion. This breaks the
          episode and produces a trivially bad evaluation result that teaches us
          nothing. More importantly, it is biologically wrong: a biological organism
          in extreme pain does NOT become paralysed — it becomes HYPERMOTIVATED to
          escape, even if every direction hurts. The "least bad" fallback models
          this: when every exit hurts, the organism chooses the exit that hurts least.
          In RL terms: an agent with a fixed maximum-step limit MUST take an action
          every step. Hard-blocking violates the environment API. The least-bad
          fallback keeps the agent moving while still being informed by pain.

        Args:
            obs:               numpy array of shape (2,) — [row, col] as float32.
            candidate_actions: list of int action codes to consider. Default: all 4.
            threshold:         aversion threshold for the should_avoid check.

        Returns:
            chosen_action (int): The action the agent should take.
            info (dict): Diagnostic information with keys:
                - "aversion_scores": {action: score} for every candidate action
                - "target_cells":    {action: (row, col)} resolved destinations
                - "blocked":         list of action ints that were vetoed
                - "safe":            list of action ints that passed the filter
                - "cornered":        True if fallback (all blocked) was used
                - "ppo_probs":       raw PPO probability for each action
        """
        # WHY default to all four actions?
        # The normal case is full freedom of movement. Specific callers can restrict
        # candidate_actions (e.g. for wall-adjacent cells where some moves are no-ops)
        # but the default assumes the full action set.
        if candidate_actions is None:
            candidate_actions = [0, 1, 2, 3]

        # WHY int() here?
        # obs comes from the environment as float32 (e.g. [3.0, 4.0]).
        # We need integer indices to look up cells in the slab and resistance field.
        row = int(obs[0])
        col = int(obs[1])

        # WHY: Track call count for statistics.
        self.total_actions += 1

        # --- STEP 1: Resolve target cells and compute aversion scores ---
        target_cells = {}       # action → (new_row, new_col)
        aversion_scores = {}    # action → float score

        for action in candidate_actions:
            # WHY _resolve_action? Mirrors the environment's wall-clamping logic.
            nr, nc = self._resolve_action(row, col, action)
            target_cells[action] = (nr, nc)
            aversion_scores[action] = self.aversion_score(nr, nc)

        # --- STEP 2: Get PPO action probabilities ---
        # WHY here (before filtering)? We need probs for ALL candidates,
        # including blocked ones, for logging and the least-bad fallback ranking.
        ppo_probs = self.get_action_probs(obs)

        # --- STEP 3: Separate safe from blocked actions ---
        safe_actions = [
            a for a in candidate_actions
            if aversion_scores[a] <= threshold   # WHY <=? See should_avoid() rationale.
        ]
        blocked_actions = [
            a for a in candidate_actions
            if aversion_scores[a] > threshold
        ]

        # --- STEP 4: Log every blocked action ---
        # WHY log at INFO level (not DEBUG)?
        # Blocked actions are the MECHANISM we want to observe — they are not
        # debugging noise, they are the signal. INFO means they appear in the
        # default logging configuration without requiring --debug flags.
        for action in blocked_actions:
            nr, nc = target_cells[action]
            score = aversion_scores[action]
            self.blocks_count += 1
            logger.info(
                f"[PULSE BLOCK] pos=({row},{col}) action={ACTION_NAMES[action]}({action}) "
                f"→ cell=({nr},{nc}) aversion={score:.4f} > threshold={threshold:.4f}"
            )

        # --- STEP 5: Choose the final action ---
        if safe_actions:
            # WHY argmax of PPO probs over SAFE actions?
            # We want the PPO policy's genuine preference, but only among safe choices.
            # Argmax gives the safe action the PPO model is MOST confident in —
            # the action it would have picked if those were its only options.
            chosen_action = max(safe_actions, key=lambda a: ppo_probs[a])
            cornered = False

        else:
            # WHY least-aversion fallback (not random, not freeze)?
            # See docstring: hard-blocking is biologically and technically wrong.
            # Least-aversion picks the "least terrible" option — the cell where
            # pain memory × physics risk is minimised. This mirrors fight-or-flight:
            # when all exits are painful, take the one that hurts least.
            chosen_action = min(candidate_actions, key=lambda a: aversion_scores[a])
            cornered = True
            self.corners_count += 1
            logger.warning(
                f"[PULSE CORNERED] pos=({row},{col}) — all {len(candidate_actions)} actions "
                f"blocked. Least-bad: action={ACTION_NAMES[chosen_action]}({chosen_action}) "
                f"aversion={aversion_scores[chosen_action]:.4f}"
            )

        # WHY return an info dict alongside the action?
        # The visualiser and the comparison script need the per-step diagnostic data
        # to draw blocked arrows and compute statistics. Returning it from select_action()
        # means callers don't need to re-query the slab and resistance_field themselves
        # after every step — they get everything in one call.
        info = {
            "aversion_scores": aversion_scores,    # raw scores for all candidates
            "target_cells":    target_cells,        # resolved (row, col) per action
            "blocked":         blocked_actions,     # vetoed action ints
            "safe":            safe_actions,        # passed action ints
            "cornered":        cornered,            # True if fallback was used
            "ppo_probs":       ppo_probs,           # raw PPO preference for all 4 actions
            "chosen":          chosen_action,       # the final selected action
        }

        return chosen_action, info

    def print_stats(self) -> None:
        """
        Prints aggregate statistics about the aversion filter's activity.
        Call this after a batch of evaluation episodes to see how often the
        filter fired, which helps tune the threshold.
        """
        # WHY guard against zero-division? If print_stats() is called before
        # any episodes run, total_actions=0 and the rate calculation would crash.
        rate = (self.blocks_count / self.total_actions) if self.total_actions > 0 else 0.0

        print("\n[PULSE Aversion Statistics]")
        print(f"  Total action decisions:   {self.total_actions}")
        print(f"  Actions blocked (vetoed): {self.blocks_count} ({rate*100:.1f}%)")
        print(f"  Cornered episodes:        {self.corners_count}")

    def reset_stats(self) -> None:
        """
        Resets aggregate statistics. Call between evaluation batches to avoid
        mixing statistics from different experimental conditions.
        """
        # WHY reset (not delete)? Keeping the attribute alive avoids AttributeError
        # in any code that reads these stats between reset and the first action.
        self.total_actions = 0
        self.blocks_count = 0
        self.corners_count = 0
