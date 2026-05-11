import numpy as np
import pyglet as pyg
import scipy.sparse

from matplotlib._cm_listed import _turbo_data as Turbo

# -------------------
#  Constants
# -------------------
# region

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
FPS = 60

WIDTH = 1900
HEIGHT = 1200 
ASPECT = WIDTH / HEIGHT

T_MIN = -1
T_MAX = 1

X_MIN = 0
X_MAX = 10
NX = 200
DX = (X_MAX - X_MIN) / NX

DT = 1e-4

HEAT_DISSIPATION_FAC = 1

COLOR_SCHEME = Turbo
WHITE = (255,255,255,255)
GRAY = (150,150,150,255)

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

def create_rect(width, height, color):
    pattern = pyg.image.SolidColorImagePattern(color)
    return pattern.create_image(width, height)


class TempGrid:
    def __init__(self, x_values, init_temp):
        """Class to hold all the information of the temperature distribution."""
        self.x_values = x_values
        self.temp = np.array(init_temp)
        
        self.plot_line = pyg.shapes.MultiLine([0,0])
        
    def get_normalized_coords(self):
        coords = np.column_stack((self.x_values, self.temp))
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

        xs = (1.02 + 1.02) * (xs - X_MIN)/(X_MAX - X_MIN) - 1.02
        ys = (1.0 + 1.0) * (ys - T_MIN)/(T_MAX - T_MIN) - 1.0

        nc = np.column_stack((xs, ys))

        return nc

    def get_indices(self):
        indices = np.repeat(range(0,2*NX-2,2), 6)

        offsets = np.tile([0,3,2,0,1,3], NX-1)
        indices += offsets

        return indices

    def get_trap_coords(self):
        trap_verts = []
        for i, (temp, temp_next) in enumerate(zip(self.temp[:-1], self.temp[1:])):

            x1 = np.interp(i, [0, len(self.temp)-1], [-1.02, 1.02])
            y1 = normalize(temp, T_MIN, T_MAX)

            x2 = np.interp(i+1, [0, len(self.temp)-1], [-1.02, 1.02])
            y2 = normalize(temp_next, T_MIN, T_MAX)

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

    def add_heat(self, x, y, sign):
        idx = int(np.interp(x, [0, WIDTH], [1, len(self.temp)-2]))
        amount = np.interp(y, [0, HEIGHT], [0.01, 0.1])
        
        self.temp[idx-5:idx+5] += sign * amount
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

        #  ----  Simulation Variables  ----
        x_values = np.linspace(X_MIN, X_MAX, NX)
        init_temp = 0.5 * np.sin(x_values/X_MAX * 4*np.pi)

        r = HEAT_DISSIPATION_FAC * DT / (2 * DX**2)

        A = scipy.sparse.diags([-r,1+2*r,-r], [-1,0,1], (NX,NX), dtype=float)
        self.A = scipy.sparse.linalg.splu(A.tocsc())        
        self.B = scipy.sparse.diags([r,1-2*r,r], [-1,0,1], (NX,NX), dtype=float)

        self.grid = TempGrid(x_values, init_temp)

        self.left_boundry = 0.0
        self.right_boundry = 0.0

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
        self.slider_left.value = 50
        # Right
        self.label_right = pyg.text.Label(
            f"Right Boundry Temperature: {self.right_boundry:.2f}",
            600, HEIGHT-50,
            anchor_y="bottom",
            font_name="JetBrains Mono", font_size=15,
            batch=self.slider_batch
        )
        self.slider_right = pyg.gui.Slider(
            600, HEIGHT-100,
            create_rect(400, 20, GRAY),
            create_rect(20,60, WHITE),
            batch=self.slider_batch
        )
        self.slider_right.push_handlers(on_change=self.update_right_boundry)
        self.push_handlers(self.slider_right)
        self.slider_right.value = 50

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

    def on_draw(self):
        self.clear()

        self.vert_list.position = self.grid.get_normalized_coords()
        self.traps.position = self.grid.get_trap_coords()
        pyg.gl.glBindTexture(pyg.gl.GL_TEXTURE_1D, self.texture_id)
        pyg.gl.glEnable(pyg.gl.GL_BLEND)
        self.batch.draw()

        self.slider_batch.draw()

    def update(self, dt):
        # Simulation
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
        if self.mouse_state.y < HEIGHT-150:
            if self.mouse_state[pyg.window.mouse.LEFT]:
                self.grid.add_heat(self.mouse_state.x, self.mouse_state.y, 1)
            if self.mouse_state[pyg.window.mouse.RIGHT]:
                self.grid.add_heat(self.mouse_state.x, self.mouse_state.y, -1)

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


if __name__ == "__main__":
    """
    Simulation of the 1D heat equation using the Crank-Nicolson Algorithm.
    """
    sim = HeatEqSim(
        width=WIDTH, height=HEIGHT,
        caption="Heat Equation Simulation (Crank-Nicolson)"
    )
    pyg.app.run()