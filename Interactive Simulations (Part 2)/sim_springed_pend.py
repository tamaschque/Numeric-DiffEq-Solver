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

DRAW_SCALE = 60
MAX_ACCEPTED_DIST = 700
PUSH_STRENGTH = 3

WHITE = (255,255,255)
RED = (255,50,50)
YELLOW = (255,255,50)

# Parameters

m = 1
g = 9.81
k = 5
l = 3

# Inital Conditions
r0 = 1
rp0 = 0

T0 = 0
Tp0 = 0

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

class SpringedPend:
    def __init__(
        self,
        attach_point,
        r=0,
        T=0,
        l=l
    ):
        # Attributes
        self.attach_point = Vec2(*attach_point)
        self.T = T
        self.r = r
        self.l = l

        self.segments = 7
        self.joint_radius = 12
        self.stroke_width = 7
        self.spring_width = 30
        self.buff = 0.15
        self.circ_radius = 40
        
        self.length = (l+r)*DRAW_SCALE

        self.end_point = self.attach_point + Vec2.from_polar(self.T - np.pi/2, self.length)

        self.batch = pyg.graphics.Batch()
        self.back_group = pyg.graphics.Group(order=-1)  # Spring is behind Pendulum Mass
        self.generate_shapes()

    def generate_shapes(self):
        self.attach_dot = pyg.shapes.Circle(
            *self.attach_point,
            radius=15,
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

        self.spring = self.get_spring_shape()

    def get_spring_shape(self) -> pyg.shapes.MultiLine:

        self.tangent = (self.end_point - self.attach_point) / self.length
        self.normal = Vec2(*self.tangent).rotate(np.pi/2).normalize()

        spring_start = map_range(self.buff, 0, 1, self.attach_point, self.end_point)
        spring_end = map_range(1-self.buff, 0, 1, self.attach_point, self.end_point)

        # Generate Jagged Points    
        points = []
        for i in range(1, 2*self.segments):
            point = Vec2(
                *map_range(
                    i,
                    0,
                    2*self.segments,
                    spring_start,
                    spring_end
                )
            )
            points.append(point)

        for i in range(len(points[1:-1])):
            if i % 4 == 0:
                points[i] += self.normal * self.spring_width
            elif i % 4 == 2:
                points[i] -= self.normal * self.spring_width

        spring_points = [self.attach_point, spring_start, *points, spring_end, self.end_point]

        return pyg.shapes.MultiLine(
            *spring_points,
            thickness=5,
            color=WHITE,
            batch=self.batch,
            group=self.back_group
        )

    def update(self, r, T):
        self.r = r
        self.T = T

        self.length = (self.l + self.r) * DRAW_SCALE
        self.end_point = self.attach_point + Vec2.from_polar(self.T - np.pi/2, self.length)

        # Update Shapes
        self.spring.delete()
        self.spring = self.get_spring_shape()

        self.circ_outline.position = self.end_point
        self.circ.position = self.end_point

    @staticmethod
    def func(t, r, T, rp, Tp):
        """Differential Equation of the Springed Pendulum."""
        return [
            (l + r) * Tp**2 - k/m * r + g * np.cos(T),
            - g/(l+r) * np.sin(T) - 2*rp/(l+r) * Tp 
        ]

class PushArrow:
    def __init__(self):
        self.batch = pyg.graphics.Batch()

        # Set as arbitrary values at initialization
        self.pend_pos = None
        self.mouse_pos = None

        # Shapes
        self.line = pyg.shapes.Line(
            0, 0,
            100, 100,
            7,
            (255,0,0),
            batch=self.batch
        )
        self.triag = pyg.shapes.Triangle(
            0, 0,
            0, 0,
            0, 0,
            (255,0 ,0),
            batch=self.batch
        )

    def calc_arrow(self):
        dist = (self.pend_pos - self.mouse_pos).length()
        dist = np.clip(dist, 0, MAX_ACCEPTED_DIST)
        tangent = (self.pend_pos - self.mouse_pos).normalize()
        normal = tangent.rotate(np.pi/2).normalize()

        arrow_len = map_range(
            dist,
            0, MAX_ACCEPTED_DIST,
            25, 200
        )

        arrow_color = map_range(
            dist,
            0, MAX_ACCEPTED_DIST,
            np.array(YELLOW), np.array(RED)
        ).astype(np.uint8)

        start = self.pend_pos + 70 * tangent
        end = start + arrow_len * tangent

        triag_points = [
            end + 25 * normal - 10 * tangent,
            end + 30 * tangent,
            end - 25 * normal - 10 * tangent
        ]

        return start, end, triag_points, arrow_color

    def update(self, mouse_pos, pend_pos):
        self.mouse_pos = Vec2(*mouse_pos)
        self.pend_pos = Vec2(*pend_pos)

        start, end, triag_points, color = self.calc_arrow()

        # Update Shapes
        self.line.color = color
        self.line.position = start
        self.line.x2, self.line.y2 = end

        self.triag.color = color
        self.triag.x, self.triag.y = triag_points[0]
        self.triag.x2, self.triag.y2 = triag_points[1]
        self.triag.x3, self.triag.y3 = triag_points[2]

class SpringedPendSimulation(pyg.window.Window):
    def __init__(
        self,
        r0=0,
        T0=0,
        rp0=0,
        Tp0=0,
        *args,
        **kwargs
        ):
        config = pyg.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
        super().__init__(*args, **kwargs, config=config)
        
        # Screen Settings
        r, g, b = [0x1f] * 3
        # pyg.gl.glClearColor(r/255, g/255, b/255, 1)

        self.set_location(
            (SCREEN_WIDTH  - WIDTH)  // 2,
            (SCREEN_HEIGHT - HEIGHT) // 2
        )

        self.pend = SpringedPend(CENTER + Vec2(0,300))
        self.push_arrow = PushArrow()

        self.mouse_state = pyg.window.mouse.MouseStateHandler()
        self.push_handlers(self.mouse_state)

        # Simulation Variables
        self.r = r0
        self.rp = rp0
        self.T = T0
        self.Tp = Tp0

        self.vars_next = [0, 0, 0, 0]
        self.push = [0, 0]

    def simulation_step(self, dt):

        self.rp += self.push[0]
        self.Tp += self.push[1]

        self.push = [0,0]

        _, self.vars_next = solve_2nd_order_ivp_interact(
            self.pend.func,
            [self.r, self.T],
            [self.rp, self.Tp],
            dt=dt
        )

        self.vars_next = self.vars_next.tolist()

    def on_draw(self):
        self.clear()
        self.pend.batch.draw()

        if self.mouse_state[pyg.window.mouse.LEFT]:
            self.push_arrow.update(
                (self.mouse_state.x, self.mouse_state.y),
                self.pend.end_point
            )
            self.push_arrow.batch.draw()

        # Prep Simulation for next Frame
        self.r, self.T, self.rp, self.Tp = self.vars_next.copy()

        self.pend.update(self.r, self.T)
        self.rp *= 0.999
        self.Tp *= 0.999

    def update(self, dt):
        self.simulation_step(dt)


    def on_mouse_release(self, x, y, button, modifiers):
        dist_vec = (self.pend.end_point - Vec2(x, y))
        dist = dist_vec.length()
        dist = np.clip(dist, 0, MAX_ACCEPTED_DIST)

        tangent = dist_vec.normalize()

        phi = np.atan2(tangent.y, tangent.x)    # Angle of incoming push
        alpha = phi - self.T                    # 'relative' angle of push 

        self.push = [
            - np.sin(alpha) * PUSH_STRENGTH,
            np.cos(alpha) * PUSH_STRENGTH
        ]
        
if __name__ == "__main__":
    """
    Controlls:
        * By holding down the left mouse button you can "charge" a push. Strength is dependent on the distance to the pendulum mass.
        * By releasing the button you push the mass in the according direction.
    """
    sim = SpringedPendSimulation(
        r0,
        T0,
        rp0,
        Tp0,
        width=WIDTH,
        height=HEIGHT,
        caption="Springed Pendulum Simulation"
    )
    pyg.clock.schedule_interval(sim.update, 1/FPS)
    pyg.app.run()
