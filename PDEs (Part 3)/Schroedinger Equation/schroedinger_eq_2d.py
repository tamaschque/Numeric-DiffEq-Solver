from typing import Callable, Sequence
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors

matplotlib.rcParams["animation.ffmpeg_path"] = r"C:\Users\Tamas\ffmpeg\bin\ffmpeg.exe"

from numeric_de_solver.caching import matching_cache_options, cache, load_cached_result

class LaplacianEvolution_CNM_2D:
    """
    Solves the 2 dimensional differential equation du/dt = `a` * ∇²u + `V(x,y)` u using the Crank-Nicolson-Method. See https://en.wikipedia.org/wiki/Crank–Nicolson_method
        * Spacial interval is [0, x_end, dx]
        * Temporal interval is [0, t_end, dt]

    Parameters
    ----------
    laplace_fac : float
        Factor `a` in the differential equations
    y0 : Sequence[float]
        Inital condition for y. The array should be of shape the same shape as the discretized spacial interval x.
    x : Sequence[float]
        Discretized spacial interval.
    t_end : float
        Boundry of temporal interval starting at t = 0.
    dt : float
        Time step.
    """
    def __init__(
        self,
        laplace_fac: float,
        potential: Sequence[Sequence[float]],
        u0: Sequence[float],
        x: Sequence[float],
        y: Sequence[float],
        t_end: float = 5,
        dt: float = 1e-3
    ):

        potential = np.array(potential)
        u0 = np.array(u0)
        self.u = u0.copy().flatten()

        self.cache_options = {
            "laplace_fac": laplace_fac,
            "potential": potential,
            "u0": u0,
            "x": x,
            "y": y,
            "t_end": t_end,
            "dt": dt
        }

        # ---------------
        # Coordinates
        # ---------------
        # Spacial
        self.x = x
        self.y = y
        dx = x[1] - x[0]
        Nx = len(x)
        Ny = len(y)

        # Temporal
        self.t_end = t_end
        dt = dt
        t = np.arange(0, t_end+dt, dt)
        self.Nt = len(t)

        # ---------------
        # Matricies
        # ---------------
        # Kinetic
        Idx = scipy.sparse.identity(Nx)
        Idy = scipy.sparse.identity(Ny)
        Dx = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx), dtype=float)
        Dy = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny), dtype=float)

        Laplacian = 1/dx**2 * ( scipy.sparse.kron(Idy, Dx) + scipy.sparse.kron(Dy, Idx) )

        # Potential
        pot_flat = potential.flatten()
        pot = scipy.sparse.spdiags([pot_flat], [0])

        fac_a = laplace_fac * dt / 2
        fac_b = dt / 2
        self.A = scipy.sparse.identity(Nx*Ny) - fac_a * Laplacian + fac_b * pot
        self.B = scipy.sparse.identity(Nx*Ny) + fac_a * Laplacian - fac_b * pot
        self.LU = scipy.sparse.linalg.splu(self.A.tocsc()) 

    def compute(
        self,
        intermediate_steps = 10,
        abs_square = True,
        cache_location = None
        ):

        # Check Cache
        if cache_location and matching_cache_options(cache_location, **self.cache_options):
            print("Found Matching Cache.")
            _, self.data = load_cached_result(cache_location)
            return

        self.data = []
        Nx = len(self.x)
        Ny = len(self.y)
        for n in range(self.Nt):
            # CNM - Step
            b = self.B.dot(self.u)
            self.u = self.LU.solve(b)
            # Apply Boundry Conditions
            u_grid = self.u.reshape((Ny, Nx))
            u_grid[0,:] = 0
            u_grid[-1,:] = 0
            u_grid[:, 0] = 0
            u_grid[:, -1] = 0

            self.u = u_grid.flatten()

            # Note down Data
            if n % intermediate_steps == 0:
                if abs_square:
                    self.data.append(np.abs(u_grid)**2)
                else:
                    self.data.append(u_grid)

        if cache_location:
            cache(cache_location, [], self.data, **self.cache_options)

    def save_anim(self, filename, renormalize_cmap=True, renorm_fac=0.8):
        plt.style.use("dark_background")
        fig, ax = plt.subplots()
        plt.xticks([])
        plt.yticks([])

        im = ax.imshow(self.data[0], cmap="inferno",
        norm=matplotlib.colors.Normalize(0,self.data[0].max() * renorm_fac)
        )

        def update(frame):

            im.set_data(self.data[frame])
            if renormalize_cmap:
                im.set_clim(vmin=0, vmax=self.data[frame].max() * renorm_fac)

            return [im]

        anim = animation.FuncAnimation(fig, update, frames=len(self.data), interval=200)
        anim.save(filename, writer="ffmpeg", fps=30, dpi=300)
        plt.close()

class Preset:
    """Variables that need to be set by the classmethod presets"""
    filename: str = "src\\numeric_de_solver\\SG\\"
    cache_location: str = "src\\numeric_de_solver\\cached_values\\"

    laplace_fac : float
    V : np.ndarray
    psi0 : np.ndarray
    x : np.ndarray
    y : np.ndarray
    t_end : float

    @classmethod
    def coulomb_potential(cls):
        cls.filename += "coulomb_scatter.mp4"
        cls.cache_location += "coulomb_scatter.json"

        hbar = 1
        m = 5
        cls.laplace_fac = 1j*hbar/(2*m)

        # Space
        X_MAX = 1.5
        Y_MAX = 1.0
        dx = 0.0025
        cls.x = np.arange(0,X_MAX+dx,dx)
        cls.y = np.arange(0,Y_MAX+dx, dx)
        X, Y = np.meshgrid(cls.x,cls.y)

        # Time
        cls.t_end = 0.2
        cls.dt = 1e-4

        # Potential
        xpot = X_MAX/2 - 0.1
        ypot = Y_MAX/2
        cls.V =  10 / np.sqrt( (X-xpot)**2 + (Y-ypot)**2 + 1e-10)

        # Inital Condition
        x0 = 0.3   # Normal
        y0 = 0.5
        sigma = 0.15
        k0x = 50
        k0y = 0

        cls.psi0 = 50 * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * np.exp(1j * (k0x * X + k0y * Y)) * 100
        cls.psi0[0,:] = 0
        cls.psi0[-1,:] = 0
        cls.psi0[:,0] = 0
        cls.psi0[:,-1] = 0

    @classmethod
    def potential_barrier(cls):
        cls.filename += "pot_barrier.mp4"
        cls.cache_location += "pot_barrier.json"
    
        hbar = 1
        m = 5
        cls.laplace_fac = 1j*hbar/(2*m)

        # Space
        X_MAX = 2.0
        Y_MAX = 1
        dx = 0.005
        cls.x = np.arange(0,X_MAX+dx,dx)
        cls.y = np.arange(0,Y_MAX+dx, dx)
        X, Y = np.meshgrid(cls.x,cls.y)

        # Time
        cls.t_end = 0.3
        cls.dt = 1e-4

        # Potential
        V0 = 25
        width = 0.025

        cls.V = np.zeros_like(X)
        cls.V[
            (X > X_MAX/2 - width/2 * X_MAX) &
            (X < X_MAX/2 + width/2 * X_MAX)
        ] = V0

        # Inital Condition
        x0 = 0.5    # Close
        x0 = 0.3   # Normal
        y0 = Y_MAX/2
        sigma = 0.12
        k0x = 30
        k0y = 0

        cls.psi0 = 40 * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * np.exp(1j * (k0x * X + k0y * Y)) * 100
        cls.psi0[0,:] = 0
        cls.psi0[-1,:] = 0
        cls.psi0[:,0] = 0
        cls.psi0[:,-1] = 0

    @classmethod
    def empty_box(cls):
        cls.filename += "empty_box.mp4"
        cls.cache_location += "empty_box.json"
    
        hbar = 1
        m = 3
        cls.laplace_fac = 1j*hbar/(2*m)

        # Space
        X_MAX = 1
        Y_MAX = 1
        dx = 0.005
        cls.x = np.arange(0,X_MAX+dx,dx)
        cls.y = np.arange(0,Y_MAX+dx, dx)
        X, Y = np.meshgrid(cls.x,cls.y)

        # Time
        cls.t_end = 0.6
        cls.dt = 1e-4

        # Potential
        cls.V = np.zeros_like(X)

        # Inital Condition
        x0 = X_MAX/2
        y0 = Y_MAX/2
        sigma = 0.1
        k0x = 0
        k0y = 0

        cls.psi0 = 40 * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * np.exp(1j * (k0x * X + k0y * Y)) * 100
        cls.psi0[0,:] = 0
        cls.psi0[-1,:] = 0
        cls.psi0[:,0] = 0
        cls.psi0[:,-1] = 0

if __name__ == "__main__":

    Preset.coulomb_potential()

    sim = LaplacianEvolution_CNM_2D(
        Preset.laplace_fac,
        Preset.V,
        Preset.psi0,
        Preset.x,
        Preset.y,
        Preset.t_end,
        Preset.dt
    )
    sim.compute(cache_location=Preset.cache_location)

    sim.save_anim(Preset.filename, True, 0.8)
