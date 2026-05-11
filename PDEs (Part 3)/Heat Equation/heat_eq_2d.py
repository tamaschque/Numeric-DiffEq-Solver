import numpy as np
import pyglet as pyg
import scipy
from matplotlib._cm_listed import _turbo_data as Turbo

from numeric_de_solver.solver_steps import matrix_rk4_step

# -------------------------------
# Constants
# -------------------------------
# region

# Window
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
WIDTH = 1900
HEIGHT = 1200 
ASPECT = HEIGHT / WIDTH

# Simulation Window
N_SIDES = 0.95
N_TOP = 0.75
RECT_WIDTH = 10
FPS = 60

SIM_ASPECT = (N_TOP + N_SIDES) / (2*N_SIDES)

# Simulation Parameters
X_MIN, X_MAX = 0, 1
Y_MIN, Y_MAX = 0, 1*ASPECT
DX = 0.005
HEAT_DISSIP_FAC = 1
T_MIN = -1
T_MAX = 1

# Colors
COLOR_SCHEME = Turbo
WHITE = (255,255,255,255)
GRAY = (150,150,150,255)

# endregion

# Use colors to show diffusion
# RED = np.array((0.75,0.0,0.0))
# GREEN = np.array((0.75,0.6,0.0))
# COLOR_SCHEME = []
# for t in np.linspace(0,1,256):
#     color = np.array(((RED - GREEN) * t + GREEN), dtype=float)
#     COLOR_SCHEME.append(color)

# -------------------------------
# Shader sources
# -------------------------------
# region

vertex_source = """
#version 330

in vec2 position;
in float value;

out float vert_value;

void main()
{
    gl_Position = vec4(position, 0.0f, 1.0f);
    vert_value = value;
}
"""

fragment_source = """
#version 330

in float vert_value;

out vec4 outColor;

uniform sampler1D colormap;

void main()
{
    outColor = texture(colormap, vert_value);
}
"""
# endregion


def create_rect(width, height, color):
    pattern = pyg.image.SolidColorImagePattern(color)
    return pattern.create_image(width, height)

def mouse_in_sim_window(x, y):
    nx = np.interp(x, [0, WIDTH], [-1.0, 1.0])
    ny = np.interp(y, [0, HEIGHT], [-1.0, 1.0])

    if -N_SIDES <= nx <= N_SIDES and -N_SIDES <= ny <= N_TOP:
        return nx, ny
    else:
        return None

def gaussian_profile(X, Y, x0, y0, sigma):
    sx = sigma
    sy = sigma / SIM_ASPECT

    exponent = (X - x0)**2 / sx**2 + (Y - y0)**2 /sy**2

    return np.exp(-0.5 * exponent)


class TempGrid:
    def __init__(self, X, Y, init_temp):
        """Class to store the temperature values of the grid and generate the correct format for display with shaders."""
        self.temp = np.array(init_temp)   
        self.ny, self.nx = init_temp.shape
        self.X = X
        self.Y = Y

    def create_grid_verts(self):
        """Create the grid points as normalized vertex coordinates."""
        xs = np.linspace(-N_SIDES, N_SIDES, self.nx)
        ys = np.linspace(-N_SIDES, N_TOP, self.ny)

        X, Y = np.meshgrid(xs, ys)

        positions = np.column_stack((X.ravel(), Y.ravel()))

        return positions.flatten()
    
    def create_indices(self):
        """Create the indices of which vertices the triangles use."""
        indices = []

        for i in range(self.nx - 1):
            for j in range(self.ny - 1):

                v0 = j * self.nx + i
                v1 = j * self.nx + (i+1)
                v2 = (j+1) * self.nx + i
                v3 = (j+1) * self.nx + (i+1)

                indices.extend([v0, v1, v2])    # Bot left Triagle ◺
                indices.extend([v1, v3, v2])    # Top right Triagle ◹

        return indices
    
    def normalized_values(self):
        vals_flat = np.flip(self.temp, 0).flatten()
        alphas = np.interp(vals_flat, [T_MIN, T_MAX], [0,1])
        return alphas

    def add_heat(self, nx, ny, sign, sigma):
        x0 = np.interp(nx, [-N_SIDES, N_SIDES], [X_MIN, X_MAX])
        y0 = np.interp(ny, [-N_SIDES, N_TOP], [Y_MAX, Y_MIN])

        self.temp += 0.1 * sign * gaussian_profile(self.X, self.Y, x0, y0, sigma)
        self.temp = np.clip(self.temp, T_MIN, T_MAX)


class HeatEq2DSim(pyg.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #  ----  General Setup  ----
        self.set_location(
            (SCREEN_WIDTH  - WIDTH)  // 2,
            (SCREEN_HEIGHT - HEIGHT) // 2
        )

        self.mouse_state = pyg.window.mouse.MouseStateHandler()
        self.push_handlers(self.mouse_state)
        pyg.clock.schedule_interval(self.update, 1/FPS)

        self.draw_radius = 0.02
        self.sim_running = False

        # Bounding Rectangle for Simulation Window
        x_off = WIDTH/2 * (1-N_SIDES) - RECT_WIDTH
        y_bot_off = HEIGHT/2 * (1-N_SIDES) - RECT_WIDTH
        y_top_off = HEIGHT/2 * (1-N_TOP) - RECT_WIDTH
        self.bbox = pyg.shapes.Rectangle(
            x_off,
            y_bot_off,
            WIDTH-2*x_off ,
            HEIGHT-y_bot_off-y_top_off,
            color=(43,43,43)
        )

        #  ----  Simulation Variables ----
        x = np.arange(X_MIN,X_MAX+DX,DX)
        y = np.arange(Y_MIN,Y_MAX+DX,DX)

        Nx, Ny = len(x), len(y)
        X, Y = np.meshgrid(x,y)

        # Matricies
        Idx = scipy.sparse.identity(Nx)
        Idy = scipy.sparse.identity(Ny)
        Dx = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx), dtype=float)
        Dy = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny), dtype=float)

        self.Laplacian = 1/DX**2 * HEAT_DISSIP_FAC * ( scipy.sparse.kron(Idy, Dx) + scipy.sparse.kron(Dy, Idx) )

        init_temp = np.full_like(X, -1.0)
        self.grid = TempGrid(X, Y, init_temp)

        self.left_boundry = -1.0
        self.right_boundry = -1.0
        self.top_boundry = -1.0
        self.bot_boundry = -1.0

        #  ----  Sliders  ----
        self.slider_batch = pyg.graphics.Batch()

        # Left
        self.label_left = pyg.text.Label(
            f"Left Temperature: {self.left_boundry:.2f}",
            0.1*WIDTH, HEIGHT-50,
            anchor_y="bottom",
            font_name="JetBrains Mono", font_size=15,
            batch=self.slider_batch
        )
        self.slider_left = pyg.gui.Slider(
            0.1*WIDTH, HEIGHT-100,
            create_rect(300, 20, GRAY),
            create_rect(20,60, WHITE),
            batch=self.slider_batch,
        )
        self.slider_left.push_handlers(on_change=self.update_left_boundry)
        self.push_handlers(self.slider_left)
        self.slider_left.value = 0

        # Right
        self.label_right = pyg.text.Label(
            f"Right Temperature: {self.right_boundry:.2f}",
            0.3*WIDTH, HEIGHT-50,
            anchor_y="bottom",
            font_name="JetBrains Mono", font_size=15,
            batch=self.slider_batch
        )
        self.slider_right = pyg.gui.Slider(
            0.3*WIDTH, HEIGHT-100,
            create_rect(300, 20, GRAY),
            create_rect(20,60, WHITE),
            batch=self.slider_batch
        )
        self.slider_right.push_handlers(on_change=self.update_right_boundry)
        self.push_handlers(self.slider_right)
        self.slider_right.value = 0

        # Top
        self.label_top = pyg.text.Label(
            f"Top Temperature: {self.top_boundry:.2f}",
            0.7*WIDTH - 300, HEIGHT-50,
            anchor_y="bottom",
            font_name="JetBrains Mono", font_size=15,
            batch=self.slider_batch
        )
        self.slider_top = pyg.gui.Slider(
            0.7*WIDTH - 300, HEIGHT-100,
            create_rect(300, 20, GRAY),
            create_rect(20,60, WHITE),
            batch=self.slider_batch
        )
        self.slider_top.push_handlers(on_change=self.update_top_boundry)
        self.push_handlers(self.slider_top)
        self.slider_top.value = 0

        # Bot
        self.label_bot = pyg.text.Label(
            f"Bottom Temperature: {self.bot_boundry:.2f}",
            0.9*WIDTH - 300, HEIGHT-50,
            anchor_y="bottom",
            font_name="JetBrains Mono", font_size=15,
            batch=self.slider_batch
        )
        self.slider_bot = pyg.gui.Slider(
            0.9*WIDTH - 300, HEIGHT-100,
            create_rect(300, 20, GRAY),
            create_rect(20,60, WHITE),
            batch=self.slider_batch
        )
        self.slider_bot.push_handlers(on_change=self.update_bot_boundry)
        self.push_handlers(self.slider_bot)
        self.slider_bot.value = 0

        #  ----  Shader  ----
        self.init_texture()

        self.shader_batch = pyg.graphics.Batch()
        vert_shader = pyg.graphics.shader.Shader(vertex_source, "vertex")
        frag_shader = pyg.graphics.shader.Shader(fragment_source, "fragment")
        self.program = pyg.graphics.shader.ShaderProgram(vert_shader, frag_shader)

        grid_indices = self.grid.create_indices()
        grid_verts = self.grid.create_grid_verts()
        grid_norm_values = self.grid.normalized_values()

        self.vert_list = self.program.vertex_list_indexed(
            self.grid.nx * self.grid.ny,
            pyg.gl.GL_TRIANGLES,
            grid_indices,
            batch=self.shader_batch,
            position=("f", grid_verts),
            value=("f", grid_norm_values)
        )

    def on_draw(self):
        self.clear()

        self.bbox.draw()

        self.program.use()
        pyg.gl.glBindTexture(pyg.gl.GL_TEXTURE_1D, self.texture_id)
        pyg.gl.glEnable(pyg.gl.GL_BLEND)
        
        self.vert_list.value = self.grid.normalized_values()
        self.shader_batch.draw()

        self.slider_batch.draw()
 
    def init_texture(self):
        # Get Color Scheme as bytearray
        color_map_bytes = bytearray()
        for r, g, b in COLOR_SCHEME:
            color = bytes([
                int(r*255),
                int(g*255),
                int(b*255),
                255
            ])
            color_map_bytes += color

        width = len(color_map_bytes)

        # Setup Texture within OpenGL
        self.texture_id = pyg.gl.GLuint()
        self.texture = pyg.gl.glGenTextures(1, self.texture_id)
        pyg.gl.glBindTexture(pyg.gl.GL_TEXTURE_1D, self.texture_id)
        
        pyg.gl.glTexParameteri(pyg.gl.GL_TEXTURE_1D, pyg.gl.GL_TEXTURE_MIN_FILTER, pyg.gl.GL_LINEAR)
        pyg.gl.glTexParameteri(pyg.gl.GL_TEXTURE_1D, pyg.gl.GL_TEXTURE_MAG_FILTER, pyg.gl.GL_LINEAR)
        pyg.gl.glTexParameteri(pyg.gl.GL_TEXTURE_1D, pyg.gl.GL_TEXTURE_WRAP_S, pyg.gl.GL_CLAMP_TO_EDGE)

        pyg.gl.glTexImage1D(
            pyg.gl.GL_TEXTURE_1D,
            0,
            pyg.gl.GL_RGBA8,
            width // 4,
            0,
            pyg.gl.GL_RGBA,
            pyg.gl.GL_UNSIGNED_BYTE,
            (pyg.gl.GLubyte * width).from_buffer(color_map_bytes)
        )

    def simulation_step(self):
        values_flat = self.grid.temp.flatten()    # Flatten Matrix to a Vector to perform computation
        next_values = matrix_rk4_step(self.Laplacian, values_flat, 1e-6)
        next_values_grid = np.reshape(next_values, (self.grid.ny, self.grid.nx))

        # Apply Boundry Conditions
        next_values_grid[0, :] = self.top_boundry
        next_values_grid[-1, :] = self.bot_boundry
        next_values_grid[:, 0] = self.left_boundry
        next_values_grid[:, -1] = self.right_boundry

        self.grid.temp = next_values_grid
        
    def update(self, dt):
        # Simulation
        if self.sim_running:
            # Each Frame run multiple simulation steps
            for _ in range(10):     
                self.simulation_step()
    
        # Interactivity
        if npos := mouse_in_sim_window(self.mouse_state.x, self.mouse_state.y):
            if self.mouse_state[pyg.window.mouse.LEFT]:
                self.grid.add_heat(*npos, sign=1, sigma=self.draw_radius)
            if self.mouse_state[pyg.window.mouse.RIGHT]:
                self.grid.add_heat(*npos, sign=-1, sigma=self.draw_radius)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            self.draw_radius = np.clip(self.draw_radius+0.005, 0.01, 0.06)
        if scroll_y < 0:
            self.draw_radius = np.clip(self.draw_radius-0.005, 0.01, 0.06)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyg.window.key.SPACE:
            self.sim_running = not self.sim_running

    def update_left_boundry(self, slider, value):
        self.left_boundry = float(np.interp(value, [0,100], [-1,1]))
        self.label_left.text = f"Left Temperature: {self.left_boundry:.2f}"
    
    def update_right_boundry(self, slider, value):
        self.right_boundry = float(np.interp(value, [0,100], [-1,1]))
        self.label_right.text = f"Right Temperature: {self.right_boundry:.2f}"

    def update_top_boundry(self, slider, value):
        self.top_boundry = float(np.interp(value, [0,100], [-1,1]))
        self.label_top.text = f"Top Temperature: {self.top_boundry:.2f}"

    def update_bot_boundry(self, slider, value):
        self.bot_boundry = float(np.interp(value, [0,100], [-1,1]))
        self.label_bot.text = f"Bottom Temperature: {self.bot_boundry:.2f}"


if __name__ == "__main__":
    sim = HeatEq2DSim(
        width=WIDTH, height=HEIGHT,
        caption="2D Heat Equation Simulation - (Method of Lines)"
    )
    pyg.app.run()
