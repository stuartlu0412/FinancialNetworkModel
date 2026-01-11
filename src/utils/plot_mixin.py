import matplotlib.pyplot as plt
from matplotlib import animation

class LivePlotMixin:
    """
    Mix-in for live animation.  Assumes subclass implements:
      • _render(ax)            → Artist to be drawn once  
      • _get_frame_data()      → array/data to update Artist each frame  
      • step()                 → advance model by one Monte Carlo sweep  
    """

    def animate(self, n_steps: int, interval: int = 50):
        fig, ax = plt.subplots()
        artist = self._render(ax)

        def _update(frame):
            self.step()
            data = self._get_frame_data()
            # for image artists:
            if hasattr(artist, 'set_data'):
                artist.set_data(data)
            return artist,

        ani = animation.FuncAnimation(
            fig, _update, frames=n_steps, interval=interval, blit=True, repeat=False
        )
        plt.show()