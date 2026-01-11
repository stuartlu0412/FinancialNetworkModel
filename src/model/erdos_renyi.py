import numpy as np
import pandas as pd
import networkx as nx
import random
from src.model.basemodel import BaseMarketModel
from src.utils.plot_mixin import LivePlotMixin

class ErdosRenyiModel(LivePlotMixin, BaseMarketModel):

    def __init__(
        self,
        alpha: float = 20,
        beta: float = 2,
        N: int = 50000,
        k: int = 4,
        p: float = 0.5,
        seed: int = 123
    ) -> None:
        
        super().__init__(seed)
        self.alpha = alpha
        self.beta = beta
        self.N = N
        self.k = k  # average degree
        self.p = p  # initial probability for spin/strategy assignment
        
        # Initialize graph structure (deterministic based on seed)
        # Note: Graph creation will use the seed from run() method
        self.G = None
        
        # Time series will be initialized in initialize()
        self.M_t = 0
        self.F_t = 0
        self.NB_t = 0
        self.M_t_values = []
        self.F_t_values = []
        self.NB_t_values = []

    def initialize(self) -> None:
        """Set up initial state: create graph and assign spins/strategies"""
        print(f"Initializing Erdős-Rényi model with N={self.N}, k={self.k}...")
        
        # Create Erdős-Rényi random graph
        # Probability for edge creation: p = 2k/(N-1) to get average degree k
        edge_prob = 2 * self.k / (self.N - 1)
        print(f"  Creating random graph (edge probability = {edge_prob:.6f})...")
        self.G = nx.gnp_random_graph(self.N, edge_prob, seed=self.seed)
        print(f"  Graph created: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Initialize spins S and strategies C randomly
        print(f"  Assigning random spins and strategies...")
        for node in self.G.nodes():
            self.G.nodes[node]['S'] = np.random.choice([-1, 1])
            self.G.nodes[node]['C'] = np.random.choice([-1, 1])

        # Calculate initial magnetization M_t
        self.M_t = sum(nx.get_node_attributes(self.G, 'S').values())
        
        # Calculate initial number of fundamentalists (C == 1)
        self.F_t = sum(
            1 for node in self.G.nodes() 
            if self.G.nodes[node]['C'] == 1
        )
        
        # Calculate initial number of disorder bonds (edges with opposite spins)
        self.NB_t = self._compute_disorder_bonds()

        # Initialize time series
        self.M_t_values = [self.M_t]
        self.F_t_values = [self.F_t]
        self.NB_t_values = [self.NB_t]
        
        print(f"  Initial state: M_t={self.M_t}, F_t={self.F_t}, NB_t={self.NB_t}")
        print("Initialization complete!\n")

    def step(self) -> None:
        """Perform one Monte Carlo sweep (N random node updates)"""
        # One full Monte-Carlo sweep = N random updates
        for _ in range(self.N):
            # Randomly choose a node to update
            node = random.randrange(self.N)
            self._update_node(node)

        # Record after the full sweep
        self.M_t_values.append(self.M_t)
        self.F_t_values.append(self.F_t)
        self.NB_t_values.append(self.NB_t)
        
        # Print progress every 100 steps
        # current_step = len(self.M_t_values) - 1
        # if current_step % 100 == 0:
        #     print(f"Step {current_step}: M_t={self.M_t}, F_t={self.F_t}, NB_t={self.NB_t}")

    def _update_node(self, node: int) -> None:
        """Update a single node's spin and strategy"""
        # Calculate local field from neighbors
        local = sum(
            self.G.nodes[neighbor]['S'] 
            for neighbor in self.G.neighbors(node)
        )
        
        # Total field h
        h = local - self.G.nodes[node]['C'] * self.alpha * self.M_t / self.N
        p_flip = 1 / (1 + np.exp(-self.beta * h))

        # Update strategy C if misaligned with global magnetization
        if self.G.nodes[node]['S'] * self.G.nodes[node]['C'] * self.M_t < 0:
            self.G.nodes[node]['C'] = -self.G.nodes[node]['C']
            # Update F_t count
            if self.G.nodes[node]['C'] == 1:
                self.F_t += 1
            else:
                self.F_t -= 1

        # Store old spin for NB_t update
        old_spin = self.G.nodes[node]['S']
        
        # Update spin S with probability p_flip
        if random.random() < p_flip:
            if self.G.nodes[node]['S'] == -1:
                self.G.nodes[node]['S'] = 1
                self.M_t += 2
        else:
            if self.G.nodes[node]['S'] == 1:
                self.G.nodes[node]['S'] = -1
                self.M_t -= 2
        
        # Update NB_t if spin changed
        if self.G.nodes[node]['S'] != old_spin:
            self._update_NB(node, old_spin)

        # Sanity check
        if abs(self.M_t) > self.N:
            raise ValueError(f"M_t ({self.M_t}) exceeds bounds [-{self.N}, {self.N}]")

    def _compute_disorder_bonds(self) -> int:
        """Count the number of edges connecting nodes with opposite spins"""
        disorder_count = 0
        for u, v in self.G.edges():
            if self.G.nodes[u]['S'] != self.G.nodes[v]['S']:
                disorder_count += 1
        return disorder_count
    
    def _update_NB(self, node: int, old_spin: int) -> None:
        """
        Incrementally update NB_t when a node's spin changes.
        
        For each neighbor:
        - If old_spin != neighbor_spin: was a disorder bond, now might not be
        - If new_spin != neighbor_spin: is now a disorder bond, might not have been before
        """
        new_spin = self.G.nodes[node]['S']
        for neighbor in self.G.neighbors(node):
            neighbor_spin = self.G.nodes[neighbor]['S']
            # Check if this edge was a disorder bond before the flip
            was_disorder = (old_spin != neighbor_spin)
            # Check if this edge is a disorder bond after the flip
            is_disorder = (new_spin != neighbor_spin)
            
            if was_disorder and not is_disorder:
                # Edge changed from disorder to order
                self.NB_t -= 1
            elif not was_disorder and is_disorder:
                # Edge changed from order to disorder
                self.NB_t += 1

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