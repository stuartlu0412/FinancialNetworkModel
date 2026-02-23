import math
import numpy as np
import pandas as pd
import networkx as nx
from numba import njit
from src.model.basemodel import BaseMarketModel
from src.utils.plot_mixin import LivePlotMixin


# ---------------------------------------------------------------------------
# Numba-compiled inner loop (module-level so it is JIT-compiled once)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _mc_sweep(
    spins:      np.ndarray,   # int8  (N,)
    strategies: np.ndarray,   # int8  (N,)
    adj_ptr:    np.ndarray,   # int32 (N+1,)  CSR row pointers
    adj_idx:    np.ndarray,   # int32 (E,)    CSR neighbour indices
    node_seq:   np.ndarray,   # int32 (N,)    pre-generated node indices
    rand_vals:  np.ndarray,   # float64 (N,)  pre-generated U[0,1)
    M_t:   int,
    F_t:   int,
    NB_t:  int,
    alpha: float,
    beta:  float,
    N:     int,
) -> tuple:
    """One full MC sweep compiled by Numba — avoids all Python-level overhead."""
    alpha_over_N = alpha / N

    for idx in range(N):
        node  = node_seq[idx]
        s_i   = int(spins[node])    # cast int8 → int64 for arithmetic
        c_i   = int(strategies[node])

        # ── Local field: sum of neighbour spins ───────────────────────
        start = adj_ptr[node]
        end   = adj_ptr[node + 1]
        local = 0
        for j in range(start, end):
            local += int(spins[adj_idx[j]])

        # ── Flip probability ──────────────────────────────────────────
        h      = local - c_i * alpha_over_N * M_t
        p_flip = 1.0 / (1.0 + math.exp(-beta * h))

        # ── Update strategy C ─────────────────────────────────────────
        if s_i * c_i * M_t < 0:
            c_i = -c_i
            strategies[node] = c_i
            F_t += c_i          # +1 if now fundamentalist, -1 otherwise

        # ── Determine new spin ────────────────────────────────────────
        if rand_vals[idx] < p_flip:
            new_spin = 1
        else:
            new_spin = -1

        if new_spin != s_i:
            spins[node] = new_spin
            M_t += 2 * new_spin   # +2 (−1→+1) or −2 (+1→−1)

            # ── Incremental NB_t update ───────────────────────────────
            for j in range(start, end):
                ns = int(spins[adj_idx[j]])
                was_disorder = (ns != s_i)
                is_disorder  = (ns != new_spin)
                if is_disorder and not was_disorder:
                    NB_t += 1
                elif was_disorder and not is_disorder:
                    NB_t -= 1

    return M_t, F_t, NB_t


class ErdosRenyiModel(LivePlotMixin, BaseMarketModel):
    """
    Erdős-Rényi Bornholdt spin model — optimised implementation.

    Performance improvements over the original:
    - Spins & strategies stored as int8 numpy arrays (no dict lookups).
    - Adjacency stored in CSR (Compressed Sparse Row) format:
        adj_ptr[i] : adj_ptr[i+1]  →  neighbours of node i in adj_idx.
    - Inner MC sweep compiled by Numba (@njit, cached) — avoids all
      Python-level loop overhead, yielding ~10–50× speed-up.
    - Random node indices and flip thresholds pre-generated in bulk
      (two numpy calls per sweep instead of N Python calls).
    """

    def __init__(
        self,
        alpha: float = 20,
        beta: float = 2,
        N: int = 50000,
        k: int = 4,
        p: float = 0.5,
        seed: int = 123,
    ) -> None:
        super().__init__(seed)
        self.alpha = alpha
        self.beta = beta
        self.N = N
        self.k = k
        self.p = p

        self.G:           nx.Graph | None = None
        # CSR adjacency (built in initialize)
        self._adj_ptr:    np.ndarray | None = None  # int32 (N+1,)
        self._adj_idx:    np.ndarray | None = None  # int32 (total_edges,)
        self._spins:      np.ndarray | None = None  # int8  (N,)
        self._strategies: np.ndarray | None = None  # int8  (N,)

        self.M_t  = 0
        self.F_t  = 0
        self.NB_t = 0
        self.M_t_values:  list[int] = []
        self.F_t_values:  list[int] = []
        self.NB_t_values: list[int] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the graph, build CSR adjacency, seed arrays, warm up JIT."""
        N = self.N
        print(f"Initializing Erdős-Rényi model with N={N}, k={self.k}...")

        # ── 1. Build graph ────────────────────────────────────────────
        edge_prob = 2 * self.k / (N - 1)
        print(f"  Creating random graph (edge_prob={edge_prob:.6f})...")
        self.G = nx.gnp_random_graph(N, edge_prob, seed=self.seed)
        print(f"  Graph: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

        # ── 2. Build CSR adjacency ────────────────────────────────────
        print("  Building CSR adjacency list...")
        adj_lists = [sorted(self.G.neighbors(i)) for i in range(N)]
        degrees   = np.array([len(a) for a in adj_lists], dtype=np.int32)
        ptr       = np.zeros(N + 1, dtype=np.int32)
        ptr[1:]   = np.cumsum(degrees)
        idx_flat  = np.empty(int(ptr[N]), dtype=np.int32)
        for i, nbrs in enumerate(adj_lists):
            idx_flat[ptr[i]: ptr[i + 1]] = nbrs
        self._adj_ptr = ptr
        self._adj_idx = idx_flat

        # ── 3. Initialise spins & strategies ─────────────────────────
        print("  Assigning random spins and strategies...")
        rng = np.random.default_rng(self.seed)
        self._spins      = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
        self._strategies = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)

        # ── 4. Compute aggregate statistics ──────────────────────────
        self.M_t  = int(self._spins.sum())
        self.F_t  = int((self._strategies == 1).sum())
        self.NB_t = self._compute_disorder_bonds()

        self.M_t_values  = [self.M_t]
        self.F_t_values  = [self.F_t]
        self.NB_t_values = [self.NB_t]

        print(f"  Initial state: M_t={self.M_t}, F_t={self.F_t}, NB_t={self.NB_t}")

        # ── 5. Warm up the Numba JIT (compile with dummy 1-node data) ──
        print("  Warming up Numba JIT (first-call compilation) …")
        _dummy_ptr = np.array([0, 0], dtype=np.int32)   # 1 node, 0 neighbours
        _dummy_idx = np.empty(0, dtype=np.int32)
        _mc_sweep(
            self._spins[:1].copy(), self._strategies[:1].copy(),
            _dummy_ptr, _dummy_idx,
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.float64),
            0, 0, 0,
            self.alpha, self.beta, 1,
        )
        print("Initialization complete!\n")

    # ------------------------------------------------------------------
    # Monte-Carlo sweep
    # ------------------------------------------------------------------

    def step(self) -> None:
        """One Monte Carlo sweep — delegates hot loop to the Numba kernel."""
        N = self.N
        node_seq  = np.random.randint(0, N, size=N).astype(np.int32)
        rand_vals = np.random.random(size=N)

        M_t, F_t, NB_t = _mc_sweep(
            self._spins, self._strategies,
            self._adj_ptr, self._adj_idx,
            node_seq, rand_vals,
            self.M_t, self.F_t, self.NB_t,
            self.alpha, self.beta, N,
        )
        self.M_t  = M_t
        self.F_t  = F_t
        self.NB_t = NB_t

        self.M_t_values.append(M_t)
        self.F_t_values.append(F_t)
        self.NB_t_values.append(NB_t)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _compute_disorder_bonds(self) -> int:
        """Vectorised count of edges whose endpoints have opposite spins."""
        edges = np.array(list(self.G.edges()), dtype=np.int32)
        if len(edges) == 0:
            return 0
        return int((self._spins[edges[:, 0]] != self._spins[edges[:, 1]]).sum())

    def _update_NB(self, node: int, old_spin: int) -> None:
        """Kept for API compatibility — NB_t is now updated inside _mc_sweep."""
        start = self._adj_ptr[node]
        end   = self._adj_ptr[node + 1]
        nbr_spins    = self._spins[self._adj_idx[start:end]]
        new_spin     = int(self._spins[node])
        was_disorder = nbr_spins != old_spin
        is_disorder  = nbr_spins != new_spin
        self.NB_t += int((is_disorder & ~was_disorder).sum())
        self.NB_t -= int((was_disorder & ~is_disorder).sum())

    # ─── Hooks for LivePlotMixin (optional - network plotting is complex) ──────
    def _render(self, ax):
        """Render the network state (optional, can be complex for large N)"""
        # For large networks (N=50000), live plotting is not practical
        # This is a placeholder - consider implementing for small N only
        if not hasattr(self, '_warned_plotting'):
            print(f"Warning: Live plotting not implemented for large networks (N={self.N})")
            self._warned_plotting = True
        return None

    def _get_frame_data(self):
        """Return current state for animation"""
        # Return magnetization as proxy for state
        return self.M_t

    # ─── Simulation API ───────────────────────────────────────────────────────
    def simulate(
        self,
        frames: int = 3000,
        output_dir: str = None,
        do_plot: bool = False
    ) -> pd.DataFrame:
        """
        Run simulation for `frames` steps.
        
        Parameters:
        -----------
        frames : int
            Number of Monte Carlo sweeps to run
        output_dir : str, optional
            Directory to save results (params.json and timeseries.csv)
        do_plot : bool
            Whether to show live plotting (not recommended for large N)
        
        Returns:
        --------
        pd.DataFrame
            Time series data with columns M_t, F_t
        """
        print(f"\n{'='*70}")
        print(f"Starting Erdős-Rényi simulation")
        print(f"Parameters: α={self.alpha}, β={self.beta}, N={self.N}, k={self.k}")
        print(f"Running {frames} Monte Carlo sweeps...")
        print(f"{'='*70}\n")
        
        # Note: For network models with large N, live plotting is not practical
        if do_plot and self.N > 1000:
            print(f"Warning: Live plotting disabled for large networks (N={self.N})")
            do_plot = False
        
        if do_plot:
            self.animate(frames, interval=1)

        # Delegate to BaseMarketModel.run(), which handles:
        # - Seeding RNGs
        # - Calling initialize()
        # - Running step() loop
        # - Saving params and timeseries if output_dir is set
        df = self.run(frames, output_dir)
        
        print(f"\n{'='*70}")
        print(f"Simulation complete!")
        if output_dir:
            print(f"Results saved to: {output_dir}/")
        print(f"{'='*70}\n")
        
        return df