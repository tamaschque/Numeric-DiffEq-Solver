import numpy as np
import pyglet as pyg
import scipy.sparse

from matplotlib._cm_listed import _turbo_data as Turbo

# -------------------
#  Constants
# -------------------
# region

# General
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
FPS = 60
# Window
WIDTH = 1900
HEIGHT = 1200 
ASPECT = WIDTH / HEIGHT
# Simulation Window
N_SIDES = 0.95
N_TOP = 0.75
RECT_WIDTH = 10
# Simulation Variables
T_MIN = -1
T_MAX = 1

X_MIN = 0
X_MAX = 10
NX = 100
DX = (X_MAX - X_MIN) / NX

DT = 1e-4

HEAT_DISSIPATION_FAC = 1

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

# -------------------
#  Shaders
# -------------------
# region

vertex_source = """
    #version 330

    in vec2 position;
    in float value;
    
    out float vertex_value;
    out float vertex_opacity;
    
    void main() {
        vertex_value = value;
        gl_Position = vec4(position, 0.0, 1.0);
    }
"""

fragment_source = """
    #version 330

    in float vertex_value;

    out vec4 frag_color;

    uniform sampler1D colormap;

    void main() {
        vec4 color = texture(colormap, vertex_value);
        frag_color = color;
    }
"""

# endregion

def create_rect(width, height, color):
    pattern = pyg.image.SolidColorImagePattern(color)
    return pattern.create_image(width, height)

def gaussian_profile(xs, x0, sigma):
    return np.exp(-0.5 * (xs - x0)**2 / sigma**2 )

def mouse_in_sim_window(x, y):
    nx = np.interp(x, [0, WIDTH], [-1.0, 1.0])
    ny = np.interp(y, [0, HEIGHT], [-1.0, 1.0])

    if -N_SIDES <= nx <= N_SIDES and -N_SIDES <= ny <= N_TOP:
        return nx, ny
    else:
        return None


class TempGrid:
    def __init__(self, x_values, init_temp):
        """Class to hold all the information of the temperature distribution."""
        self.x_values = x_values
        self.temp = np.array(init_temp)
        
        self.plot_line = pyg.shapes.MultiLine([0,0])
        
    def get_normalized_coords(self):
        """Calculate (normalized) coordinates of the corners of
        the rectangles."""
        nc_cords = np.interp(
            self.x_values,
            [X_MIN, X_MAX],
            [-N_SIDES, N_SIDES] # Normalized Coords of Screen
        )

        tops = np.full(NX, N_TOP, dtype=float)
        bots = np.full(NX, -N_SIDES, dtype=float)

        verts = np.column_stack((nc_cords, tops, nc_cords, bots))

        return verts.flatten()

    def get_indices(self):
        """Calculate indices in which order to travers the corners
        to construct the triangle (sub)mesh."""
        indices = np.repeat(range(0,2*NX-2,2), 6)

        offsets = np.tile([0,1,3,0,3,2], NX-1)
        indices += offsets

        return indices

    def get_values(self):
        norm_vals =  np.interp(self.temp, [T_MIN, T_MAX], [0, 1])
        return np.repeat(norm_vals, 2)

    def add_heat(self, nx, ny, sign):
        x0 = np.interp(nx, [-N_SIDES, N_SIDES], [X_MIN, X_MAX])
        amount = np.interp(ny, [-N_SIDES, N_TOP], [0.001, 0.1])
        
        self.temp += sign * amount * gaussian_profile(self.x_values, x0, 0.25)
        self.temp = np.clip(self.temp, T_MIN, T_MAX)

class HeatEqSim(pyg.window.Window):
    def __init__(self, *args, **kwargs):
        config = pyg.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
        super().__init__(*args, **kwargs, config=config)

        #  ----  General Setup  ----
        pyg.gl.glEnable(pyg.gl.GL_BLEND)

        pyg.gl.glBlendFunc(pyg.gl.GL_SRC_ALPHA, pyg.gl.GL_ONE_MINUS_SRC_ALPHA)
        self.set_location(
            (SCREEN_WIDTH  - WIDTH)  // 2,
            (SCREEN_HEIGHT - HEIGHT) // 2
        )

        pyg.clock.schedule_interval(self.update, 1/FPS)
        self.mouse_state = pyg.window.mouse.MouseStateHandler()
        self.push_handlers(self.mouse_state)

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

        #  ----  Simulation Variables  ----
        x_values = np.linspace(X_MIN, X_MAX, NX)
        init_temp = np.full(NX, -1.0, dtype=float)

        r = HEAT_DISSIPATION_FAC * DT / (2 * DX**2)

        A = scipy.sparse.diags([-r,1+2*r,-r], [-1,0,1], (NX,NX), dtype=float)
        self.A = scipy.sparse.linalg.splu(A.tocsc())        
        self.B = scipy.sparse.diags([r,1-2*r,r], [-1,0,1], (NX,NX), dtype=float)

        self.grid = TempGrid(x_values, init_temp)

        self.left_boundry = -1.0
        self.right_boundry = -1.0

        self.simulating = False

        #  ----  Sliders  ----
        self.slider_batch = pyg.graphics.Batch()
        # Left
        self.label_left = pyg.text.Label(
            f"Left Boundry Temperature: {self.left_boundry:.2f}",
            100, HEIGHT-50,
            anchor_y="bottom",
            font_name="JetBrains Mono", font_size=15,
            batch=self.slider_batch
        )
        self.slider_left = pyg.gui.Slider(
            100, HEIGHT-100,
            create_rect(400, 20, GRAY),
            create_rect(20,60, WHITE),
            batch=self.slider_batch,
        )
        self.slider_left.push_handlers(on_change=self.update_left_boundry)
        self.push_handlers(self.slider_left)
        self.slider_left.value = 0
        # Right
        self.label_right = pyg.text.Label(
            f"Right Boundry Temperature: {self.right_boundry:.2f}",
            WIDTH-500, HEIGHT-50,
            anchor_y="bottom",
            font_name="JetBrains Mono", font_size=15,
            batch=self.slider_batch
        )
        self.slider_right = pyg.gui.Slider(
            WIDTH-500, HEIGHT-100,
            create_rect(400, 20, GRAY),
            create_rect(20,60, WHITE),
            batch=self.slider_batch
        )
        self.slider_right.push_handlers(on_change=self.update_right_boundry)
        self.push_handlers(self.slider_right)
        self.slider_right.value = 0

        #  ----  Shaders  ----
        self.init_texture()

        vertex_shader = pyg.graphics.shader.Shader(vertex_source, "vertex")
        fragment_shader = pyg.graphics.shader.Shader(fragment_source, "fragment")
        self.program = pyg.graphics.shader.ShaderProgram(vertex_shader, fragment_shader)

        self.batch = pyg.graphics.Batch()

        norm_coords = self.grid.get_normalized_coords()
        values = self.grid.get_values()

        self.vert_list = self.program.vertex_list_indexed(
            2*NX,
            pyg.gl.GL_TRIANGLES,
            self.grid.get_indices(),
            batch=self.batch,
            position=("f", norm_coords),
            value=("f", values)
        )

    def on_draw(self):
        self.clear()

        self.bbox.draw()

        self.vert_list.value = self.grid.get_values()
        pyg.gl.glBindTexture(pyg.gl.GL_TEXTURE_1D, self.texture_id)
        pyg.gl.glEnable(pyg.gl.GL_BLEND)
        self.batch.draw()

        self.slider_batch.draw()

    def update(self, dt):
        # Simulation
        if self.simulating:
            for _ in range(20):
                b = self.B.dot(self.grid.temp)

                # Account for boundry conditions in implicit equations
                r = HEAT_DISSIPATION_FAC * DT / (2 * DX**2)
                b[0] += r * self.left_boundry
                b[-1] += r * self.right_boundry

                self.grid.temp = self.A.solve(b)

                # Enforce Boundry Conditions
                self.grid.temp[0] = self.left_boundry
                self.grid.temp[-1] = self.right_boundry
            
        # Interactivity
        if npos := mouse_in_sim_window(self.mouse_state.x, self.mouse_state.y):
            if self.mouse_state[pyg.window.mouse.LEFT]:
                self.grid.add_heat(*npos, 1)
            if self.mouse_state[pyg.window.mouse.RIGHT]:
                self.grid.add_heat(*npos, -1)

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

    def update_left_boundry(self, slider, value):
        self.left_boundry = float(np.interp(value, [0,100], [-1,1]))
        self.label_left.text = f"Left Boundry Temperature: {self.left_boundry:.2f}"
    
    def update_right_boundry(self, slider, value):
        self.right_boundry = float(np.interp(value, [0,100], [-1,1]))
        self.label_right.text = f"Right Boundry Temperature: {self.right_boundry:.2f}"

    def on_key_press(self, symbol, modifiers):
        if symbol == pyg.window.key.SPACE:
            self.simulating = not self.simulating

if __name__ == "__main__":
    """
    Simulation of the 1D heat equation using the Crank-Nicolson Algorithm.
    """
    sim = HeatEqSim(
        width=WIDTH, height=HEIGHT,
        caption="Heat Equation Simulation (Diffusion View)"
    )
    pyg.app.run()