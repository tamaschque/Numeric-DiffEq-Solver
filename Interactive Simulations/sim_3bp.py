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
ASPECT = WIDTH / HEIGHT

CENTER = Vec2(WIDTH//2, HEIGHT//2)

BODY_RADIUS = 15
TRAIL_LENGTH = 200

WHITE = (255,255,255)

# Parameters

G = 10
m1 = 1
m2 = 2
m3 = 2

# endregion

# -------------------------------
# Functions
# -------------------------------
# region

def dist(r1, r2):
    # Cap min distance to avoid unreasonable accerlations
    return max(np.linalg.norm(r1-r2), 0.04)

f1 = lambda r1, r2, r3: -G * m2 * (r1-r2) / dist(r1,r2)**3 - G * m3 * (r1-r3) / dist(r1,r3)**3

f2 = lambda r1, r2, r3: -G * m1 * (r2-r1) / dist(r2,r1)**3 - G * m3 * (r2-r3) / dist(r2,r3)**3

f3 = lambda r1, r2, r3: -G * m1 * (r3-r1) / dist(r3,r1)**3 - G * m2 * (r3-r2) / dist(r3,r2)**3

def threebodyfunc(t, r1x , r1y , r2x , r2y , r3x , r3y, r1px, r1py, r2px, r2py, r3px, r3py):
    
    r1 = np.array([r1x,r1y])
    r2 = np.array([r2x,r2y])
    r3 = np.array([r3x,r3y])
                
    return [
        *f1(r1,r2, r3),
        *f2(r1,r2, r3),
        *f3(r1,r2, r3),
    ]


def nc2sc(nc_x, nc_y):
    tx = (nc_x + 1) / 2 
    ty = (nc_y + 1) / 2 

    x = WIDTH * tx
    y = HEIGHT * ty

    return x, y

def sc2nc(x, y):
    tx = x / WIDTH
    ty = y / HEIGHT

    nc_x = 2 * tx - 1
    nc_y = 2 * ty - 1

    return nc_x, nc_y

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
    float r = length(local_pos);

    float core = smoothstep(1.0, 0.0, r);

    float glow = exp(-6.0 * r * r);

    float intensity = core * 0.4 + glow;

    FragColor = vec4(color * intensity, intensity);

}
"""

# endregion

class Body:
    """Class that hold information of a body. Uses normalized coordinates."""
    def __init__(self, x, y, xp=0, yp=0, batch=None, radius=BODY_RADIUS, color=WHITE, program=None, add_trail=True):
        self.x = x
        self.y = y
        self.xp = xp
        self.yp = yp
        self.color = color
        self.batch = batch
        self.program = program
        self.add_trail = add_trail

        self.shape = pyg.shapes.Circle(
            *nc2sc(self.x, self.y),
            radius=radius,
            color=color,
            batch=self.batch
        )

        verts = [-1, -1, 1, -1, 1, 1, -1, 1]
        self.vert_list = self.program.vertex_list(
            4,
            pyg.gl.GL_TRIANGLE_FAN,
            position=('f', verts)
        )

        if self.add_trail:
            self.trail = [nc2sc(self.x, self.y)] * TRAIL_LENGTH

            self.line = pyg.shapes.MultiLine(
                *self.trail,
                color=self.color,
                batch=self.batch
            )

    @property
    def vars(self):
        return self.x, self.y
    
    @property
    def vars_p(self):
        return self.xp, self.yp

    def set_pos(self, x, y):
        self.x, self.y = sc2nc(x, y)

    def update(self):
        self.shape.position = nc2sc(self.x, self.y)

        if self.add_trail:
            self.trail.pop()
            x,y = nc2sc(self.x, self.y)
            self.trail.insert(0, (float(x), float(y)))

            self.line.delete()
            self.line = pyg.shapes.MultiLine(
                *self.trail,
                thickness=4,
                color=self.color,
                batch=self.batch
            )

    def draw(self, color, radius=0.2):
        self.program.use()
        pyg.gl.glEnable(pyg.gl.GL_BLEND)
        pyg.gl.glBlendFunc(pyg.gl.GL_ONE, pyg.gl.GL_ONE)
        self.program["center"] = self.vars
        self.program["radius"] = radius
        self.program["color"]  = color
        self.program['aspect'] = ASPECT

        self.vert_list.draw(pyg.gl.GL_TRIANGLE_FAN)

    def pull_to_pos(self, nc_x, nc_y):

        vec = Vec2(nc_x-self.x, nc_y-self.y)
        dist = vec.length()
        tangent = vec.normalize()

        self.xp += dist * tangent.x
        self.yp += dist * tangent.y

        if dist < 0.5:
            self.xp *= 0.95
            self.yp *= 0.95

class ThreeBodyProblemSimulation(pyg.window.Window):
    def __init__(
        self,
        *args,
        **kwargs
        ):
        config = pyg.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
        super().__init__(*args, **kwargs, config=config)
        
        # General Setup
        self.set_location(
            (SCREEN_WIDTH  - WIDTH)  // 2,
            (SCREEN_HEIGHT - HEIGHT) // 2
        )
        self.batch = pyg.graphics.Batch()
        self.mouse_state = pyg.window.mouse.MouseStateHandler()
        self.key_state = pyg.window.key.KeyStateHandler()
        self.push_handlers(self.mouse_state, self.key_state)

        # Shader
        self.program = pyg.graphics.shader.ShaderProgram(
            pyg.graphics.shader.Shader(vertex_source, 'vertex'),
            pyg.graphics.shader.Shader(fragment_source, 'fragment')
        )

        # Simulation Variables
        vars = [
            -0.5, 0,
            0.5 , 0,
            0   , 0
        ]
        
        # Bodies
        self.body_1 = Body(
            vars[0], vars[1],
            batch=self.batch,
            program=self.program,
            color=(220, 220, 160)
        )

        self.body_2 = Body(
            vars[2], vars[3],
            batch=self.batch,
            program=self.program,
            color=(100, 220, 100)
        )
        
        self.body_3 = Body(
            vars[4], vars[5],
            batch=self.batch,
            program=self.program,
            color=(180, 255, 180),
            add_trail=False
        )

    def simulation_step(self, dt):

        _, y_values = solve_2nd_order_ivp_interact(
            threebodyfunc,
            [
                *self.body_1.vars,
                *self.body_2.vars,
                *self.body_3.vars
                ],
            [
                *self.body_1.vars_p,
                *self.body_2.vars_p,
                *self.body_3.vars_p,
            ],
            dt=0.001
        )
        
        self.body_1.x, self.body_1.y = y_values[0], y_values[1]
        self.body_2.x, self.body_2.y = y_values[2], y_values[3]

        self.body_1.xp, self.body_1.yp = y_values[6], y_values[7]
        self.body_2.xp, self.body_2.yp = y_values[8], y_values[9]

    def update(self, dt):

        # Move Body 3 to mouse pos
        x = self.mouse_state.x
        y = self.mouse_state.y
        self.body_3.set_pos(x, y)

        self.simulation_step(dt)

        # Call planets back with Spacebar
        if self.key_state[pyg.window.key.SPACE]:
            self.body_1.pull_to_pos(-0.5, 0)
            self.body_2.pull_to_pos(0.5, 0)

    def on_draw(self):
        self.clear()

        self.body_1.update()
        self.body_2.update()
        self.body_3.update()

        self.batch.draw()
        self.body_1.draw((1, 0.9, 0), radius=0.15)
        self.body_2.draw((0.5, 1, 0.5), radius=0.15)     
        self.body_3.draw((0, 0.9, 1), radius=0.15)


if __name__ == "__main__":
    """
    Controlls:
        * Control the blue body with the mouse.
        * The other two bodies can be called back by holding down the spacebar.
    """
    sim = ThreeBodyProblemSimulation(
        width=WIDTH,
        height=HEIGHT,
        caption="3 Body Problem Simulation"
    )
    pyg.clock.schedule_interval(sim.update, 1/FPS)
    pyg.app.run()
