import numpy as np
import pyglet as pyg
from pyglet.math import Vec2

from interactive_ivp_solver import solve_2nd_order_ivp_interact

# -------------------------------
# Constants
# -------------------------------
# region

FPS = 60

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440

WIDTH = 1200
HEIGHT = 900

CENTER = Vec2(WIDTH//2, HEIGHT//2)
ATTACH_POS = CENTER + Vec2(0,300)

DRAW_SCALE = 70
MAX_ACCEPTED_DIST = 700
PUSH_STRENGTH = 3

WHITE = (255,255,255)
RED = (255,50,50)
YELLOW = (255,255,50)

# Inital Conditions
T10 = np.pi/4 * 0
T20 = np.pi/8 * 0

T1p0 = -2 * 0
T2p0 = 2 * 0

# Parameters

m = 1
g = 9.81
k = 5
l = 3

q_mid = 0
q_bot = 1

# endregion

# -------------------------------
# Shaders
# -------------------------------
# region

vertex_source = """
#version 330 core
in vec2 position;
out vec2 local_pos;

uniform vec2 center;
uniform float radius;
uniform float aspect;

void main() {
    local_pos = vec2(
        (position.x - center.x) / radius * aspect,
        (position.y - center.y) / radius
    );
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_source = """
#version 330 core
in vec2 local_pos;
out vec4 FragColor;

uniform vec3 color;

void main() {
    float dist = length(local_pos);
    float alpha = clamp(1.0 - dist, 0.0, 1.0);
    FragColor = vec4(color, alpha);
}
"""

# endregion

# -------------------------------
# Functions
# -------------------------------
# region

def map_range(x, x_min, x_max, target_min, target_max):
    interval = x_max - x_min
    percentage = (x - x_min) / interval

    target_x = (target_max - target_min) * percentage + target_min

    return target_x

# endregion

class DoublePend:
    def __init__(
        self,
        attach_point,
        T1=0,
        T2=0,
        l=l
    ):
        # Attributes
        self.attach_point = Vec2(*attach_point)
        self.T1 = T1
        self.T2 = T2
        self.l = l

        self.joint_radius = 12
        self.stroke_width = 7
        self.circ_radius = 40
        
        self.length = l*DRAW_SCALE

        self.mid_point = self.attach_point + Vec2.from_polar(self.T1 - np.pi/2, self.length)
        self.end_point = self.mid_point + Vec2.from_polar(self.T2 - np.pi/2, self.length)

        self.batch = pyg.graphics.Batch()
        self.generate_shapes()

    def generate_shapes(self):
        self.attach_dot = pyg.shapes.Circle(
            *self.attach_point,
            radius=self.joint_radius,
            batch=self.batch
        )

        self.mid_dot = pyg.shapes.Circle(
            *self.mid_point,
            radius=self.joint_radius,
            batch=self.batch
        )

        self.line = pyg.shapes.MultiLine(
            self.attach_point, self.mid_point, self.end_point,
            thickness=self.stroke_width,
            color=WHITE,
            batch=self.batch
        )

        self.circ_outline = pyg.shapes.Circle(
            *self.end_point,
            radius=self.circ_radius + 5,
            color=WHITE,
            batch=self.batch
        )
        self.circ = pyg.shapes.Circle(
            *self.end_point,
            radius=self.circ_radius,
            color=RED,
            batch=self.batch
        )

    def update(self, T1, T2):
        self.T1 = T1
        self.T2 = T2

        self.mid_point = self.attach_point + Vec2.from_polar(self.T1 - np.pi/2, self.length)
        self.end_point = self.mid_point + Vec2.from_polar(self.T2 - np.pi/2, self.length)

        # Update Shapes
        self.mid_dot.position = self.mid_point

        self.line.delete()
        self.line = pyg.shapes.MultiLine(
            self.attach_point, self.mid_point, self.end_point,
            thickness=self.stroke_width,
            color=WHITE,
            batch=self.batch
        )

        self.circ_outline.position = self.end_point
        self.circ.position = self.end_point

    @staticmethod
    def func(x0, y0, q, t, T1, T2, T1p, T2p):
        """Differential Equation of Charged Double Pendulum."""
        dT = T1 - T2
        D = 1 - 1/2 * np.cos(dT)**2

        dx1 = l*np.cos(T1) - x0
        dy1 = l*np.sin(T1) - y0

        dx2 = l*np.cos(T1) + l*np.cos(T2) - x0
        dy2 = l*np.sin(T1) + l*np.sin(T2) - y0

        r1 = max( np.sqrt( dx1**2 + dy1**2 ), 1e-10)
        r2 = max( np.sqrt( dx2**2 + dy2**2 ), 1e-10)

        k = 200
        q1 = q_mid
        q2 = q_bot

        V1_1 = -1/2 * 1/r1**3 * ( -2*dx1*l*np.sin(T1) + 2*dy1*l*np.cos(T1) )
        V2_1 = -1/2 * 1/r2**3 * ( -2*dx2*l*np.sin(T1) + 2*dy2*l*np.cos(T1) )
        V2_2 = -1/2 * 1/r2**3 * ( -2*dx2*l*np.sin(T2) + 2*dy2*l*np.cos(T2) )

        F1 = k*q1*q*V1_1 + k*q2*q*V2_1
        F2 = k*q2*q*V2_2

        R1 = - 1/2 * T2p**2 * np.sin(dT) - g/l * np.sin(T1) - F1/(2*m*l**2)
        R2 = T1p**2 * np.sin(dT) - g/l * np.sin(T2) - F2/(m*l**2)

        return [
            (R1 - 1/2 * R2 * np.cos(dT)) / D,
            (R2 - R1 * np.cos(dT)) / D
        ]

class DoublePendSimulation(pyg.window.Window):
    def __init__(
        self,
        T10 = 0,
        T20 = 0,
        T1p0 = 0,
        T2p0 = 0,
        *args,
        **kwargs
        ):
        config = pyg.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
        super().__init__(*args, **kwargs, config=config)
        
        # Screen Settings
        r, g, b = [0x1f] * 3
        pyg.gl.glClearColor(r/255, g/255, b/255, 1)

        self.set_location(
            (SCREEN_WIDTH  - WIDTH)  // 2,
            (SCREEN_HEIGHT - HEIGHT) // 2
        )

        # Charge at Mouse Cursor
        self.program = pyg.graphics.shader.ShaderProgram(
            pyg.graphics.shader.Shader(vertex_source, 'vertex'),
            pyg.graphics.shader.Shader(fragment_source, 'fragment')
        )

        verts = [-1, -1, 1, -1, 1, 1, -1, 1]
        self.vert_list = self.program.vertex_list(
            4,
            pyg.gl.GL_TRIANGLE_FAN,
            position=('f', verts)
        )

        self.pend = DoublePend(ATTACH_POS, T10, T20)

        self.mouse_state = pyg.window.mouse.MouseStateHandler()
        self.push_handlers(self.mouse_state)

        # Simulation Variables
        self.T1 = T10
        self.T2 = T20
        self.T1p = T1p0
        self.T2p = T2p0

        self.vars_next = [0, 0, 0, 0]

        self.charge = 0
        self.charge_color = (0.2, 0.2, 0.2)
        self.charge_strength = 1

    def simulation_step(self, dt):
        mouse_x = self.mouse_state.x - ATTACH_POS[0]
        mouse_y = self.mouse_state.y - ATTACH_POS[1]

        # Remap mouse pos to simulation coordinates

        x = mouse_x / DRAW_SCALE
        y = - mouse_y / DRAW_SCALE

        _, self.vars_next = solve_2nd_order_ivp_interact(
            lambda *args: self.pend.func(y, x, self.charge, *args),
            [self.T1, self.T2],
            [self.T1p, self.T2p],
            dt=dt
        )

        self.vars_next = self.vars_next.tolist()

    def on_draw(self):
        # Prep Simulation for next Frame
        self.T1, self.T2, self.T1p, self.T2p = self.vars_next.copy()
        self.pend.update(self.T1, self.T2)

        self.T1p *= 0.999
        self.T2p *= 0.999

        self.clear()
        # Draw Charge
        ndc_coords = [
            2 * (self.mouse_state.x / WIDTH) - 1,
            2 * (self.mouse_state.y / HEIGHT) - 1
        ]
        aspect_ratio = WIDTH / HEIGHT

        self.program["center"] = ndc_coords
        self.program["radius"] = 0.2 * self.charge_strength
        self.program["color"]  = self.charge_color
        self.program['aspect'] = aspect_ratio
        
        self.program.use()
        pyg.gl.glEnable(pyg.gl.GL_BLEND)
        pyg.gl.glBlendFunc(pyg.gl.GL_SRC_ALPHA, pyg.gl.GL_ONE_MINUS_SRC_ALPHA)
        self.vert_list.draw(pyg.gl.GL_TRIANGLE_FAN)

        # Draw Pendulum
        self.pend.batch.draw()

    def update(self, dt):
        self.simulation_step(dt)

        # Exert Force of Charge
        if self.mouse_state[pyg.window.mouse.LEFT]:
            self.charge = 1 * self.charge_strength
            self.charge_color = (1.0, 0.2, 0.2)
        elif self.mouse_state[pyg.window.mouse.RIGHT]:
            self.charge = -1 * self.charge_strength
            self.charge_color = (0.2, 0.2, 1.0)
        else:
            self.charge = 0
            self.charge_color = (0.2, 0.2, 0.2)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            offset = 0.25
        elif scroll_y < 0:
            offset = -0.25

        self.charge_strength = np.clip(self.charge_strength + offset, 0.1, 3)

if __name__ == "__main__":
    """
    Controlls:
        * By holding the left/right mouse button you excert a positive/negative charge on the pendulum. Initially the middle joint is uncharged (q_mid = 0) and the mass at the end is positively charged (q_bot > 0).
        * By scrolling you can increase/decrease the effect of the charge. 
    """
    sim = DoublePendSimulation(
        T10, T20, T1p0, T2p0,
        width=WIDTH,
        height=HEIGHT,
        caption="Charged Double Pendulum Simulation")
    pyg.clock.schedule_interval(sim.update, 1/FPS)
    pyg.app.run()
