# =============================================================================
# slab.py — The Vector Slab: The Agent's "Skin"
# Part of PULSE: Pain, Understanding, and Learning State Engine — Phase 2
# =============================================================================
#
# WHY this file is separate from pulse_env.py:
# The Slab is NOT a property of the environment — it's a property of the AGENT.
# The environment defines what's out there (grid, traps, goal). The Slab defines
# what the agent feels about what's out there. Keeping them separate means:
#   1. You can swap environments (e.g. a 16x16 grid) without touching the Slab.
#   2. You can give multiple agents different Slabs in the same environment.
#   3. The Slab can be analysed, saved, and visualised independently.
# This mirrors biological reality: the nervous system is architecturally separate
# from the external world — it represents the world, it isn't the world.
#
# CONCEPTUAL MODEL:
# Imagine the 8x8 grid is the agent's body surface — its skin. Each cell on the
# grid corresponds to a patch of skin. When the agent visits a cell (especially a
# trap), that patch of skin "deforms" — a learnable vector gets pushed out of its
# resting state (zero). The magnitude of that displacement is the pain score.
# Cells that have never been activated have zero vectors (no pain). Trap cells,
# visited repeatedly, accumulate large deformations (chronic pain).
# =============================================================================

# WHY: PyTorch is the foundation of the Slab. We use it because:
#   1. Tensors give us GPU support for free if we ever scale up.
#   2. nn.Module gives us save/load, parameter tracking, and gradient support.
#   3. Future phases will run backpropagation THROUGH the Slab, so starting
#      with PyTorch now means zero refactoring cost later.
import torch

# WHY: torch.nn provides the building blocks for neural network modules.
# nn.Parameter wraps a tensor and registers it as a trainable parameter —
# this means it shows up in model.parameters(), can be optimised by Adam, etc.
import torch.nn as nn

# WHY: numpy is used only for the visualisation bridge (matplotlib speaks numpy).
# The Slab's internal computations stay in PyTorch throughout.
import numpy as np


class SlabNetwork(nn.Module):
    """
    SlabNetwork — a grid of learnable vectors representing the agent's "skin."

    WHY subclass nn.Module (not just a plain Python class)?
    nn.Module gives us three things for free that we will eventually need:
      1. model.save_state_dict() / model.load_state_dict() — persist the Slab
         between training runs without writing custom serialisation code.
      2. model.parameters() — if we later want to train the Slab end-to-end
         via backpropagation (e.g. using a pain prediction loss), the optimiser
         needs to find all Parameters automatically.
      3. model.to(device) — move the entire Slab to GPU with one line.

    A plain Python class with raw numpy arrays would need all of this built by hand.
    """

    def __init__(self, grid_size: int = 8, vector_dim: int = 16):
        """
        WHY vector_dim=16 as the default?
        16 is a deliberate middle ground:
          - Too small (e.g. dim=2): the vector can only encode pain magnitude and
            one directional component. It can't capture nuanced "texture" of pain
            (e.g. difference between falling in a trap vs. repeatedly bumping a wall).
          - Too large (e.g. dim=256): the slab becomes overparameterised relative to
            the 8x8 grid. With 64 cells × 256 dims = 16,384 parameters, most will
            never receive enough deformations to carry meaningful signal. The pain
            heatmap becomes noisy and hard to interpret.
          - dim=16 means 64 × 16 = 1,024 parameters. Compact, interpretable, but
            rich enough for each cell to encode multiple pain "channels" — intensity,
            duration, directionality, recency, etc. — when training evolves.

        In neuroscience terms, think of vector_dim as the number of distinct
        nociceptor (pain receptor) types per skin patch. More dimensions = more
        fine-grained pain discrimination. 16 is our Phase 2 starting hypothesis.

        Args:
            grid_size:  Side length of the square grid (must match pulse_env.py).
            vector_dim: Dimensionality of each cell's pain vector ("skin thickness").
        """
        # WHY: Always call super().__init__() first when subclassing nn.Module.
        # This initialises PyTorch's internal bookkeeping (parameter registry,
        # hook system, device tracking). Forgetting this causes cryptic errors.
        super().__init__()

        # WHY: Store grid_size and vector_dim as instance attributes so all
        # methods can access them without hardcoding 8 or 16 anywhere else.
        # Changing one number here propagates to every method automatically.
        self.grid_size = grid_size
        self.vector_dim = vector_dim

        # --- THE CORE DATA STRUCTURE ---
        # WHY nn.Parameter?
        # nn.Parameter wraps a tensor and marks it as "learnable" in PyTorch's
        # graph. This means:
        #   - It appears in self.parameters() → gradient-based optimisers find it.
        #   - It's saved and loaded with state_dict().
        #   - It receives gradients during backpropagation.
        #
        # Shape: (grid_size, grid_size, vector_dim) = (8, 8, 16)
        # Indexing: slab_vectors[row, col, :] gives the 16-dim vector for cell (row, col).
        #
        # WHY initialise to zeros?
        # Zero is the "resting state" of the skin — no deformation, no pain.
        # Any non-zero vector represents accumulated damage. Initialising to small
        # random values (like default nn.Linear) would mean every cell starts
        # with phantom pain, which is biologically wrong for our model.
        # torch.zeros returns a float32 tensor by default, which is what we want.
        self.slab_vectors = nn.Parameter(
            torch.zeros(grid_size, grid_size, vector_dim)
        )

        # WHY: The learning rate controls how much a single deformation event
        # shifts the slab vector. 0.01 is intentionally small:
        #   - Too large (e.g. 1.0): one trap visit would dominate the vector,
        #     erasing history from all prior deformations (catastrophic overwrite).
        #   - Too small (e.g. 1e-6): deformations are imperceptibly small and
        #     the pain score never rises meaningfully.
        #   - 0.01 means after ~100 trap visits, a cell's vector has unit magnitude,
        #     which is a reasonable "chronic pain threshold" to visualise.
        self.lr = 0.01

    # ==========================================================================
    # CORE SLAB OPERATIONS
    # ==========================================================================

    def get_vector(self, x: int, y: int) -> torch.Tensor:
        """
        Returns the current pain vector for cell (x, y).

        WHY return a tensor (not numpy array)?
        Keeping the return type as a tensor means callers can chain PyTorch
        operations (e.g. compute cosine similarity between two cell vectors)
        without a conversion step. If a caller needs numpy, they call .numpy()
        themselves — we don't force the conversion.

        Args:
            x: Row index   (0 = top row)
            y: Column index (0 = leftmost column)

        Returns:
            A 1D tensor of shape (vector_dim,) containing the pain vector.
        """
        # WHY .detach()?
        # slab_vectors is an nn.Parameter, meaning it's part of the computation
        # graph. Returning it directly would allow callers to accidentally trigger
        # gradient flows through the get_vector() call. .detach() creates a view
        # that is disconnected from the graph — safe for inspection and logging.
        # When we WANT gradients (future backprop training), we'll have a separate
        # method that skips detach().
        return self.slab_vectors[x, y].detach()

    def deform(self, x: int, y: int, force_vector: torch.Tensor) -> None:
        """
        Applies a deformation (pain stimulus) to cell (x, y).

        WHY "deform" instead of "update" or "pain_signal"?
        The word "deform" comes from mechanics: applying force to an elastic material
        changes its shape. We are modelling the skin as an elastic material — the
        force_vector is the stimulus (e.g. trap penalty encoded as a vector), and
        the slab vector is the resulting physical deformation. The name makes the
        physical metaphor explicit.

        WHY manual update instead of backpropagation?
        In Phase 2 we're in a "Hebbian" regime — the slab updates based on direct
        experience, not gradient signals from a loss function. This mirrors how
        sensitisation works in nociception: repeated painful stimuli at a location
        lower the pain threshold at that location, independent of any global
        learning signal. In Phase 3, we may switch to gradient-based deformation,
        but the manual update is the correct biological model for Phase 2.

        Args:
            x:            Row index of the cell to deform.
            y:            Column index of the cell to deform.
            force_vector: 1D tensor of shape (vector_dim,) — the pain stimulus.
                          Conventionally, this is the trap/reward signal encoded
                          into vector_dim dimensions (e.g. broadcast scalar × ones).
        """
        # WHY torch.no_grad()?
        # nn.Parameter tracks gradients by default. If we modify it inside a
        # gradient context, PyTorch may try to track this operation as part of
        # the computation graph, causing memory leaks or incorrect gradients.
        # torch.no_grad() says "this is a manual update — ignore it for autograd."
        # This is the same pattern used by PyTorch optimiser step() internals.
        with torch.no_grad():
            # WHY += instead of = ?
            # Accumulation is the key to memory. Each deformation ADDS to the
            # cell's existing state rather than overwriting it. This means:
            #   - A cell hit once has a small deformation.
            #   - A cell hit 100 times has a large deformation (chronic pain).
            # If we used =, every deformation would erase prior experience,
            # making the Slab memoryless — it would forget all past stimuli.
            self.slab_vectors[x, y] += self.lr * force_vector

    def deformation_depth(self, x: int, y: int) -> float:
        """
        Returns the L2 norm (Euclidean magnitude) of the slab vector at (x, y).
        This scalar is the cell's PAIN SCORE — how deformed this patch of skin is.

        WHY L2 norm as the pain score?
        The L2 norm ||v|| = sqrt(v₁² + v₂² + ... + v₁₆²) measures the total
        displacement of the vector from the origin (resting state). It answers:
        "How far has this cell's vector moved from zero?"

        Why not alternatives?
        - L1 norm (sum of abs values): More sensitive to small activations across
          many dimensions, less to concentrated large values. Would make pain
          "smeared" — every tiny stimulus would contribute equally to the score
          regardless of which dimension it activates. L1 is harder to interpret
          geometrically.
        - L∞ norm (max absolute value): Only reads the single loudest dimension.
          This ignores accumulation across dimensions — 100 small stimuli across
          all 16 dims would score the same as one large stimulus in one dim.
          We want a HOLISTIC pain score, not a max-channel score.
        - Mean: Would DECREASE if we add more dimensions (the same total pain
          spread across more dims looks smaller). This creates a paradox where
          "richer representations hurt less."
        - L2: Treats each dimension symmetrically, grows with both magnitude and
          spread of activation, and has a clear geometric meaning (distance from
          origin). It is the standard metric for "how much has this vector moved?"

        In nociception: L2 is analogous to the total afferent signal strength —
        the aggregate of all nociceptor channels firing from a skin patch.

        Returns:
            A Python float representing the pain score (≥ 0). Zero = no pain.
        """
        # WHY torch.norm()? It computes ||v||₂ = sqrt(sum(v_i²)).
        # .item() converts the 0-dim tensor to a plain Python float, which is
        # more convenient for logging, sorting, and matplotlib colour mapping.
        return torch.norm(self.slab_vectors[x, y]).item()

    def reset_cell(self, x: int, y: int) -> None:
        """
        Zeroes out the slab vector at (x, y), simulating complete local healing.

        WHY "reset" vs "decay" vs "heal"?
        This is a HARD reset — the cell forgets all accumulated pain instantly.
        This models acute, complete healing (e.g. removing the source of damage
        and resting). Contrast with elastic_recovery() below, which models
        gradual, continuous healing without losing all memory.

        Biologically: reset_cell() ≈ resolution of acute inflammation after an
        injury site is protected and rested.
        elastic_recovery() ≈ slow reduction in central sensitisation over days.

        Both are biologically real; which to use depends on the simulation context.
        """
        with torch.no_grad():
            # WHY zero_() (in-place) instead of assignment?
            # In-place operations on nn.Parameters must use in-place variants
            # (methods ending in _) inside torch.no_grad(). Direct assignment
            # (self.slab_vectors[x, y] = torch.zeros(self.vector_dim)) would
            # break the Parameter's registration in PyTorch's module system.
            self.slab_vectors[x, y].zero_()

    def elastic_recovery(self, decay: float = 0.99) -> None:
        """
        Multiplies ALL slab vectors by the decay factor, simulating gradual healing.

        WHY decay=0.99 as the default?
        At decay=0.99, a cell retains 99% of its deformation each timestep.
        After 100 timesteps with no new stimuli: 0.99^100 ≈ 0.366 → ~63% healed.
        After 1,000 timesteps: 0.99^1000 ≈ 0.000043 → essentially fully healed.
        This creates a biologically plausible healing timescale: pain fades slowly
        but never vanishes in a single step.

        Tuning guide:
          decay=0.999 → very slow healing (chronic pain model)
          decay=0.99  → moderate healing (default: days-to-weeks metaphor)
          decay=0.9   → fast healing (acute model: heals in ~20 steps)
          decay=0.0   → immediate full reset (equivalent to reset_cell for all cells)

        ---
        WHY ELASTIC vs PLASTIC memory matters for learning:

        ELASTIC memory: The slab naturally returns toward zero when stimuli stop.
        Think of a rubber band — pull it, release it, and it snaps back. The
        pain trace is temporary. This is essential for the agent to:
          1. Distinguish recent pain from old pain (recency signal).
          2. "Forget" benign past experiences as new ones arrive.
          3. Avoid runaway pain accumulation where every cell becomes maximally
             painful just from random exploration early in training.

        PLASTIC memory: The slab would KEEP deformations permanently (like a
        dent in clay). This is more like long-term potentiation in synapses —
        once encoded, the memory persists. Plastic memory is useful for:
          1. Encoding invariant danger zones that should NEVER be forgotten.
          2. Building up chronic pain models over thousands of episodes.
          3. Representing scar tissue — permanently altered tissue response.

        In PULSE, we use elastic memory by default because we want the agent's
        pain state to reflect its RECENT experience, not just its full history.
        A cell that was a trap but is now safe should eventually stop hurting.

        The interplay between deform() (stimulus → scar accumulation) and
        elastic_recovery() (time → healing) is what creates a DYNAMIC pain
        landscape. This is Phase 2's core contribution: the Slab models pain as
        a TIME-VARYING property, not a static label.
        ---

        Args:
            decay: Multiplicative decay factor. Must be in [0, 1].
                   0.99 is the default (slow, gradual healing).
        """
        with torch.no_grad():
            # WHY multiply in-place (*=) instead of self.slab_vectors = self.slab_vectors * decay?
            # Reassignment would create a new tensor and break the Parameter registration,
            # causing PyTorch to lose track of it for gradient computation.
            # In-place multiplication (*=) modifies the existing tensor's data directly,
            # keeping the Parameter registration intact.
            self.slab_vectors *= decay

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def pain_map(self) -> np.ndarray:
        """
        Returns a (grid_size, grid_size) numpy array of L2 pain scores for
        every cell. Used by visualise_slab.py to build the heatmap.

        WHY a dedicated pain_map method?
        Repeatedly calling deformation_depth(x, y) in nested loops in the
        visualisation script would be verbose and slow for large grids. This
        method vectorises the L2 norm computation across all cells at once
        using PyTorch's native batched operations, then hands numpy the result.

        Returns:
            A (grid_size, grid_size) float32 numpy array where each entry is
            the L2 norm of the corresponding cell's slab vector.
        """
        # WHY dim=-1 in torch.norm()?
        # slab_vectors has shape (grid_size, grid_size, vector_dim).
        # dim=-1 means "take the norm along the last axis" — i.e., compute
        # the L2 norm of the 16-dim vector at each (row, col) position.
        # The result is shape (grid_size, grid_size) — one scalar per cell.
        norms = torch.norm(self.slab_vectors, dim=-1)

        # WHY .detach().numpy()?
        # .detach() removes the tensor from the computation graph.
        # .numpy() converts to a numpy array for matplotlib compatibility.
        # These two calls together are the standard PyTorch→matplotlib bridge.
        return norms.detach().numpy()

    def summary(self) -> None:
        """
        Prints a human-readable summary of the current slab state.
        Useful for debugging and quick inspection during training.
        """
        pain = self.pain_map()
        print(f"\nSlabNetwork Summary")
        print(f"  Grid size:    {self.grid_size}×{self.grid_size}")
        print(f"  Vector dim:   {self.vector_dim}")
        print(f"  Total params: {self.grid_size**2 * self.vector_dim:,}")
        print(f"  Pain scores:")
        print(f"    Min:  {pain.min():.4f} at {np.unravel_index(pain.argmin(), pain.shape)}")
        print(f"    Max:  {pain.max():.4f} at {np.unravel_index(pain.argmax(), pain.shape)}")
        print(f"    Mean: {pain.mean():.4f}")
        print(f"    Cells with pain > 0.01: {(pain > 0.01).sum()}/{self.grid_size**2}")
