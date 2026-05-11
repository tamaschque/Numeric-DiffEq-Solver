import numpy as np
import pyglet as pyg
import scipy.sparse

from matplotlib._cm_listed import _inferno_data as Inferno
from matplotlib._cm_listed import _plasma_data as Plasma

# TODO: see if we can improove it with sparse arrays

# -------------------
#  Constants
# -------------------
# region

# Dimensions
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
FPS = 60
WIDTH = 2200
HEIGHT = 1200 
ASPECT = WIDTH / HEIGHT

# Simulation
X_MIN = -5
X_MAX = 15
NX = 200
DX = (X_MAX - X_MIN) / NX
REL_POT_POS = 0.8
DT = 1e-4
EVOLUTION_FAC = 1j

MAX_POT_STRENGTH = 0.0002
MAX_POT_WIDTH = 3
# Draw
YD_MAX = 2
YD_MIN = -0.1
XD_MAX = 10
XD_MIN = 0

# Colors
COLOR_SCHEME = Plasma
RED = (200, 0, 0)
WHITE = (255, 255, 255, 255)
GRAY = (100, 100, 100, 255)
POT_COLOR = (15,214,158)

# endregion

# -------------------
#  Shaders
# -------------------
# region

vertex_source = """
    #version 330

    in vec2 position;
    in float opacity;
    
    out float vertex_value;
    out float vertex_opacity;
    
    void main() {
        vertex_value = 0.5 * (position.y + 1.0);
        vertex_opacity = opacity; 

        gl_Position = vec4(position, 0.0, 1.0);
    }
"""

fragment_source = """
    #version 330

    in float vertex_value;
    in float vertex_opacity;

    out vec4 frag_color;

    uniform sampler1D colormap;

    void main() {
        vec4 color = texture(colormap, vertex_value);
        color.a *= vertex_opacity;
        frag_color = color;
    }
"""


# endregion

def normalize(x, x_min, x_max):
    """Normalize x to the interval [-1,1]"""
    return float(np.interp(x, [x_min,x_max], [-1,1]))

def gaussian_wavepacket(xs, x0=5, sigma=1, k0=1):
    return np.exp( -(xs - x0)**2 / (2 * sigma**2) ) * np.exp(1j * k0 * xs)

def create_rect(width, height, color):
    pattern = pyg.image.SolidColorImagePattern(color)
    return pattern.create_image(width, height)

def const_pot(xs, width, height):
    
    V = np.zeros_like(xs, dtype=complex)

    V[(xs > 5.0 - width/2) & (xs < 5.0 + width/2)] = height
    V *= 1j

    V = np.diag(V)

    return V

class WaveFunctionGrid:
    def __init__(self, x_values, psi0):
        """Class to hold all the information of the temperature distribution."""
        self.x_values = x_values
        self.psi = np.array(psi0, dtype=complex)
        
        self.plot_line = pyg.shapes.MultiLine([0,0])

    def absval(self):
        return np.abs(self.psi)**2

    def get_normalized_coords(self):
        coords = np.column_stack((self.x_values, self.absval()))
        npoints = self.normalize_coords(coords)
        npoints[:,0] *= ASPECT      # Account for aspect ratio of screen
        scale = 0.015

        nverts = []

        #  ---  Left  ---
        curr, next = npoints[0], npoints[1]
        b = next - curr
        b /= np.linalg.norm(b)

        normal = np.array([-b[1], b[0]])

        halfing = np.array([0,1])
        cosphi = np.dot(halfing, normal)

        upper = curr + scale/cosphi * halfing
        upper /= ASPECT
        lower = curr - scale/cosphi * halfing
        lower /= ASPECT
        
        nverts.append(upper)
        nverts.append(lower)

        # ---  Middle  ---
        prev = npoints[:-2]
        curr = npoints[1:-1]
        next = npoints[2:]

        a = prev - curr
        a /= np.linalg.norm(a, axis=1)[:,None]

        b = next - curr
        b /= np.linalg.norm(b, axis=1)[:,None]

        halfing = a + b
        halfing_norms = np.linalg.norm(halfing, axis=1)

        normal = np.column_stack((-b[:,1], b[:,0]))

        # Handle near zero cases
        mask = halfing_norms < 1e-10
        halfing[mask] = normal[mask]

        # Normalize
        halfing[~mask] /= halfing_norms[~mask][:,None]

        # Flip upwards
        mask = halfing[:,1] < 0
        halfing[mask] *= -1
        
        cosphi = np.sum(halfing * normal, axis=1)

        uppers = curr + (scale / cosphi)[:, None] * halfing
        lowers = curr - (scale / cosphi)[:, None] * halfing

        uppers[:,0] /= ASPECT
        lowers[:,0] /= ASPECT

        interleaved = np.ravel(np.column_stack((uppers,lowers))).reshape(-1,2)
        nverts.extend(interleaved)
        
        # ---  Rigth  ---
        prev, curr = npoints[-2], npoints[-1]
        a = prev - curr
        a /= np.linalg.norm(a)

        normal = np.array([a[1], -a[0]])
        halfing = np.array([0,1])
        cosphi = np.dot(halfing, normal)

        upper = curr + scale/cosphi * halfing
        upper /= ASPECT
        lower = curr - scale/cosphi * halfing
        lower /= ASPECT

        nverts.append(upper)
        nverts.append(lower)

        nverts = np.array(nverts).flatten()
        return nverts

    def normalize_coords(self, coords):
        coords = np.array(coords)

        xs = coords[:,0]
        ys = coords[:,1]

        xs = (1.52 + 1.52) * (xs - X_MIN)/(X_MAX - X_MIN) - 1.52
        ys = (1.0 + 1.0) * (ys - YD_MIN)/(YD_MAX - YD_MIN) - 1.0

        nc = np.column_stack((xs, ys))

        return nc

    def get_indices(self):
        indices = np.repeat(range(0,2*NX-2,2), 6)

        offsets = np.tile([0,3,2,0,1,3], NX-1)
        indices += offsets

        return indices

    def get_trap_coords(self):
        trap_verts = []
        for x, x_next, psi, psi_next in zip(self.x_values[:-1], self.x_values[1:], self.absval()[:-1], self.absval()[1:]):

            x1 = np.interp(x, [X_MIN, X_MAX], [-1.52, 1.52])
            y1 = normalize(psi, YD_MIN, YD_MAX)

            x2 = np.interp(x_next, [X_MIN, X_MAX], [-1.52, 1.52])
            y2 = normalize(psi_next, YD_MIN, YD_MAX)

            trap_verts.extend([
                x1, -1,
                x2, -1,
                x2, y2
                ])
            trap_verts.extend([
                x1, -1,
                x2, y2,
                x1, y1
                ])

        return trap_verts


class SchroedingerSim(pyg.window.Window):
    def __init__(self, *args, **kwargs):
        config = pyg.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
        super().__init__(*args, **kwargs, config=config)

        #  ----  General Setup ----
        pyg.gl.glEnable(pyg.gl.GL_BLEND)

        pyg.gl.glBlendFunc(pyg.gl.GL_SRC_ALPHA, pyg.gl.GL_ONE_MINUS_SRC_ALPHA)
        self.set_location(
            (SCREEN_WIDTH  - WIDTH)  // 2,
            (SCREEN_HEIGHT - HEIGHT) // 2
        )

        pyg.clock.schedule_interval(self.update, 1/FPS)
        self.mouse_state = pyg.window.mouse.MouseStateHandler()
        self.push_handlers(self.mouse_state)

        self.simulation_running = False

        #  ----  Parameters  ----
        self.param_batch = pyg.graphics.Batch()
        self.x0 = 5
        self.sigma = 1
        self.k0 = 1
        self._scale = 1
        self.pot_height = 1
        self.pot_width = 0.5

        # Inital Position
        self.label_x0 = pyg.text.Label("Initial Position", 200, HEIGHT-50, font_size=20, batch=self.param_batch)
        self.slider_x0 = pyg.gui.Slider(
            200,HEIGHT-100,
            create_rect(250, 20, GRAY),
            create_rect(20, 50, WHITE),
            batch=self.param_batch
            )
        self.slider_x0.push_handlers(on_change=self.update_x0)
        self.push_handlers(self.slider_x0)
        
        # Inital Spread
        self.label_sigma = pyg.text.Label("Initial Spread", 500, HEIGHT-50, font_size=20, batch=self.param_batch)
        self.slider_sigma = pyg.gui.Slider(
            500,HEIGHT-100,
            create_rect(250, 20, GRAY),
            create_rect(20, 50, WHITE),
            batch=self.param_batch
            )
        self.slider_sigma.push_handlers(on_change=self.update_sigma)
        self.push_handlers(self.slider_sigma)
        
        # Inital Velocity
        self.label_k0 = pyg.text.Label("Initial Velocity", 800, HEIGHT-50, font_size=20, batch=self.param_batch)
        self.slider_k0 = pyg.gui.Slider(
            800,HEIGHT-100,
            create_rect(250, 20, GRAY),
            create_rect(20, 50, WHITE),
            batch=self.param_batch
            )
        self.slider_k0.push_handlers(on_change=self.update_k0)
        self.push_handlers(self.slider_k0)

        # Inital Height
        self.label_scale = pyg.text.Label("Initial Height", 1100, HEIGHT-50, font_size=20, batch=self.param_batch)
        self.slider_scale = pyg.gui.Slider(
            1100,HEIGHT-100,
            create_rect(250, 20, GRAY),
            create_rect(20, 50, WHITE),
            batch=self.param_batch
            )
        self.slider_scale.push_handlers(on_change=self.update_scale)
        self.push_handlers(self.slider_scale)

        # Potential Height
        self.label_pot_height = pyg.text.Label("Potential Height", 1400, HEIGHT-50, font_size=20, batch=self.param_batch)
        self.slider_pot_height = pyg.gui.Slider(
            1400,HEIGHT-100,
            create_rect(250, 20, GRAY),
            create_rect(20, 50, WHITE),
            batch=self.param_batch
            )
        self.slider_pot_height.push_handlers(on_change=self.update_pot_height)
        self.push_handlers(self.slider_pot_height)

        # Potential Width
        self.label_pot_width = pyg.text.Label("Potential Width", 1700, HEIGHT-50, font_size=20, batch=self.param_batch)
        self.slider_pot_width = pyg.gui.Slider(
            1700,HEIGHT-100,
            create_rect(250, 20, GRAY),
            create_rect(20, 50, WHITE),
            batch=self.param_batch
            )
        self.slider_pot_width.push_handlers(on_change=self.update_pot_width)
        self.push_handlers(self.slider_pot_width)

        #  ----  Pausing Shapes  ----
        self.pause_batch = pyg.graphics.Batch()

        self.red_rect = pyg.shapes.MultiLine(
            [0,0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT],
            closed=True,
            thickness=10,
            color=RED,
            batch=self.pause_batch
        )

        self.red_circ = pyg.shapes.Circle(
            50, HEIGHT-50,
            25,
            color=RED,
            batch=self.pause_batch
        )

        #  ----  Simulation Variables  ----
        xs = np.linspace(X_MIN, X_MAX, NX)
        psi0 = gaussian_wavepacket(xs, 5, 1, 1)

        self.pot = const_pot(xs, 0, 0)

        r = EVOLUTION_FAC * DT / (2 * DX**2)

        self.A = scipy.sparse.diags([-r,1+2*r,-r], [-1,0,1], (NX,NX), dtype=complex).toarray()
        
        self.B = scipy.sparse.diags([r,1-2*r,r], [-1,0,1], (NX,NX), dtype=complex).toarray()
        self.grid = WaveFunctionGrid(xs, psi0)

        #  ----  Shaders  ----
        self.init_texture()

        vertex_shader = pyg.graphics.shader.Shader(vertex_source, "vertex")
        fragment_shader = pyg.graphics.shader.Shader(fragment_source, "fragment")
        self.program = pyg.graphics.shader.ShaderProgram(vertex_shader, fragment_shader)

        self.batch = pyg.graphics.Batch()

        # Line
        pyg.gl.glLineWidth(15.0)
        norm_coords = self.grid.get_normalized_coords()
        amount = len(norm_coords) // 2
        self.vert_list = self.program.vertex_list_indexed(
            2*NX,
            pyg.gl.GL_TRIANGLES,
            self.grid.get_indices(),
            batch=self.batch,
            position=("f", norm_coords),
            opacity=("f", [1.0] * amount)
        )
        # Shading Trapezoids
        trap_coords = self.grid.get_trap_coords()
        amount = len(trap_coords) // 2
        self.traps = self.program.vertex_list(
            amount,
            pyg.gl.GL_TRIANGLES,
            self.batch,
            position=("f", trap_coords),
            opacity=("f", [0.5] * amount)
        )

        #  ----  Potential ----
        self.potential_batch = pyg.graphics.Batch()

        self.pot_rect = pyg.shapes.Rectangle(
            0.5*WIDTH-50,0,
            100,0,
            (*POT_COLOR,100),
            batch=self.potential_batch
        )
        self.pot_outline = pyg.shapes.MultiLine(
            (0,0), (0,0), (0,0), (0,0),
            closed=True,
            thickness=15,
            color=(*POT_COLOR,255),
            batch=self.potential_batch
        )

    def on_draw(self):
        self.clear()

        # Draw Shader
        self.vert_list.position = self.grid.get_normalized_coords()
        self.traps.position = self.grid.get_trap_coords()

        self.program.use()
        pyg.gl.glBindTexture(pyg.gl.GL_TEXTURE_1D, self.texture_id)
        pyg.gl.glEnable(pyg.gl.GL_BLEND)
        self.batch.draw()

        if self.pot_height != 0:
            self.potential_batch.draw()

        # Draw Red Border is Simulation is Paused
        if not self.simulation_running:
            self.pause_batch.draw()
        
        # Draw Parameter Handles
        self.param_batch.draw()

    def on_key_press(self, symbol, modifiers):
        # Pause/Play Simulation
        if symbol == pyg.window.key.SPACE:
            self.simulation_running = not self.simulation_running
        # Reset Simulation
        if symbol == pyg.window.key.R:
            self.simulation_running = False
            self.grid.psi = gaussian_wavepacket(self.grid.x_values)

            self.slider_x0.value = 50
            self.slider_sigma.value = 50
            self.slider_k0.value = 50
            self.slider_scale.value = 50

    def update(self, dt):
        if self.simulation_running:
            # Simulation
            for _ in range(50):
                B = self.B - self.pot
                b = B.dot(self.grid.psi)
                # print(f"{b.shape=}")

                A = self.A + self.pot
                # LU = scipy.sparse.linalg.splu(A.tocsc())
                
                self.grid.psi = np.linalg.solve(A,b)
                

                
                # raise NotImplementedError
                
                # Set Boundry Conditions
                self.grid.psi[0] = 0.0
                self.grid.psi[-1] = 0.0

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

    def update_x0(self, slider, value):
        self.x0  = np.interp(value, [0,100], [X_MIN,X_MAX])
        self.grid.psi = self._scale * gaussian_wavepacket(self.grid.x_values, self.x0, self.sigma, self.k0)

    def update_k0(self, slider, value):
        self.k0  = np.interp(value, [0,100], [-1,1])
        self.grid.psi = self._scale * gaussian_wavepacket(self.grid.x_values, self.x0, self.sigma, self.k0)

    def update_sigma(self, slider, value):
        self.sigma  = np.interp(value, [0,100], [0.5,3])
        self.grid.psi = self._scale * gaussian_wavepacket(self.grid.x_values, self.x0, self.sigma, self.k0)

    def update_scale(self, slider, value):
        self._scale  = np.interp(value, [0,100], [0.1,2])
        self.grid.psi = self._scale * gaussian_wavepacket(self.grid.x_values, self.x0, self.sigma, self.k0)

    def update_pot_height(self, slider, value):
        # Update Simulation
        self.pot_height = MAX_POT_STRENGTH * value/100
        self.pot = const_pot(self.grid.x_values, self.pot_width, self.pot_height)

        draw_height = float(np.interp(value, [0,100],[0,HEIGHT]))
        self.pot_rect.height = draw_height

        self.pot_outline.delete()
        self.pot_outline = pyg.shapes.MultiLine(
            (self.pot_rect.x                      , -20),
            (self.pot_rect.x + self.pot_rect.width, -20),
            (self.pot_rect.x + self.pot_rect.width, self.pot_rect.height),
            (self.pot_rect.x                      , self.pot_rect.height),
            closed=True,
            thickness=15,
            color=(*POT_COLOR,255),
            batch=self.potential_batch
        )

    def update_pot_width(self, slider, value):
        # Update Simulation
        self.pot_width = MAX_POT_WIDTH * value/100
        self.pot = const_pot(self.grid.x_values, self.pot_width, self.pot_height)

        draw_width = self.pot_width/(XD_MAX - XD_MIN) * WIDTH

        self.pot_rect.x = 0.5*WIDTH - draw_width/2
        self.pot_rect.width = draw_width

        self.pot_outline.delete()
        self.pot_outline = pyg.shapes.MultiLine(
            (self.pot_rect.x                      , -20),
            (self.pot_rect.x + self.pot_rect.width, -20),
            (self.pot_rect.x + self.pot_rect.width, self.pot_rect.height),
            (self.pot_rect.x                      , self.pot_rect.height),
            closed=True,
            thickness=15,
            color=(*POT_COLOR,255),
            batch=self.potential_batch
        )



if __name__ == "__main__":
    sim = SchroedingerSim(
        width=WIDTH, height=HEIGHT,
        caption="Schroedinger Equation Simulation"
    )
    pyg.app.run()