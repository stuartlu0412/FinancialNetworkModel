import random
import numpy as np
import pandas as pd
from src.model.basemodel import BaseMarketModel
from src.utils.plot_mixin import LivePlotMixin

class BornholdtModel(LivePlotMixin, BaseMarketModel):
    
    def __init__(
        self,
        alpha: float = 20,
        beta: float = 2,
        L: int = 20,
        p: float = 0.5,
        use_abs_M: bool = False,
        seed: int = 123
    ) -> None:
    
        super().__init__(seed)
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.p = p
        self.use_abs_M = use_abs_M
        self.F_t = 0
        self.F_t_values = []

    def initialize(self) -> None:
        # initial spins S and strategies C
        self.S = np.where(np.random.random((self.L, self.L)) < self.p, 1, -1)
        self.C = np.where(np.random.random((self.L, self.L)) < self.p, 1, -1)

        # magnetization
        self.M_t = int(self.S.sum())
        self.M_t_values = [self.M_t]

        # number of fundamentalists
        self.F_t = int((self.C == 1).sum())
        self.F_t_values = [self.F_t]

        # initial disorder count NB and its time series
        self.NB = self._compute_initial_NB()
        self.NB_t = int(self.NB.sum())
        self.NB_t_values = [self.NB_t]

    def _compute_initial_NB(self) -> np.ndarray:
        NB = np.zeros((self.L, self.L), dtype=int)
        for i in range(self.L):
            for j in range(self.L):
                count = 0
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = (i+dx) % self.L, (j+dy) % self.L
                    if self.S[ni, nj] != self.S[i, j]:
                        count += 1
                NB[i, j] = count
        return NB

    def step(self) -> None:
        # one full Monte-Carlo sweep = L*L random updates
        for _ in range(self.L * self.L):
            i, j = random.randrange(self.L), random.randrange(self.L)
            # local field sum of 4 neighbors
            local = (
                self.S[(i-1)%self.L, j] + self.S[(i+1)%self.L, j] +
                self.S[i, (j-1)%self.L] + self.S[i, (j+1)%self.L]
            )
            # total field h (optionally use |M_t| for comparison experiments)
            M_eff = abs(self.M_t) if self.use_abs_M else self.M_t
            h = local - self.C[i, j] * self.alpha * M_eff / (self.L**2)
            p_flip = 1 / (1 + np.exp(-self.beta * h))

            # update strategy C if misaligned with global magnetization
            if self.S[i, j] * self.C[i, j] * self.M_t < 0:
                self.C[i, j] = -self.C[i, j]

            # update spin S with prob p_flip
            if random.random() < p_flip:
                if self.S[i, j] == -1:
                    self.S[i, j] = 1
                    self.M_t += 2
                    self._update_NB(i, j)
            else:
                if self.S[i, j] == 1:
                    self.S[i, j] = -1
                    self.M_t -= 2
                    self._update_NB(i, j)

        # record after the full sweep
        self.M_t_values.append(self.M_t)
        self.NB_t_values.append(self.NB_t)
        self.F_t = int((self.C == 1).sum())
        self.F_t_values.append(self.F_t)

    def _update_NB(self, i: int, j: int) -> None:
        # adjust local NB and NB_t for spin change at (i,j)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = (i+dx) % self.L, (j+dy) % self.L
            if self.S[ni, nj] == self.S[i, j]:
                self.NB[i, j]    -= 1
                self.NB[ni, nj]  -= 1
                self.NB_t        -= 2
            else:
                self.NB[i, j]    += 1
                self.NB[ni, nj]  += 1
                self.NB_t        += 2

    # ─── Hooks for LivePlotMixin ──────────────────────────────────────────────
    def _render(self, ax):
        self.img = ax.imshow(self.S, animated=True)
        return self.img

    def _get_frame_data(self):
        return self.S

    # ─── Simulation API ───────────────────────────────────────────────────────
    def simulate(
        self,
        frames: int = 3000,
        output_dir: str = None,
        do_plot: bool = True
    ) -> pd.DataFrame:
        """
        1) Optionally launch live animation for `frames` steps.
        2) Then delegate to BaseMarketModel.run(), which
           seeds RNGs, re-initializes, runs `step()` `frames` times,
           and dumps params + timeseries.csv if `output_dir` is set.
        """
        '''
        # 1) seed RNGs
        import random as _rand, numpy as _np
        _rand.seed(self.seed)
        _np.random.seed(self.seed)
        # 2) build initial state
        self.initialize()
        '''
        #  3) now live‐plot that state as you step
        if do_plot:
           self.animate(frames, interval=1)

        # run() will re-seed, call initialize() & step() loop, and return DataFrame
        return self.run(frames, output_dir)

if __name__ == "__main__":
    # 1) construct with whatever parameters you like
    model = BornholdtModel(L=50, p=0.5, alpha=20, beta=2, seed=42)

    # 2) run with live plotting (do_plot=True) or headless (do_plot=False)
    model.initialize()
    df = model.simulate(
        frames=3000,
        output_dir="results/bornholdt",  # will be created if needed
        do_plot=False
    )

    # 3) df is a pandas.DataFrame of your time series (M_t, NB_t)
    print(df.head())