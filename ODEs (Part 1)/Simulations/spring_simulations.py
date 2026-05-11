from manim.typing import Point3DLike, Point3D
from manim.mobject.opengl.opengl_mobject import OpenGLMobject

from manim import *
import numeric_de_solver as desolv

# ---------------------------------------------
#   Helpers
# ---------------------------------------------

def pointify(mob_or_point: Mobject | Point3DLike) -> Point3D:
    if isinstance(mob_or_point, (Mobject, OpenGLMobject)):
        return mob_or_point.get_center()
    else:
        return np.array(mob_or_point)


class Spring(VMobject):
    """Simple illustration of a spring as a line with jagged edges.

    Parameters
    ----------
    start : Mobject | Point3DLike
        Starting point of the spring.
    end : Mobject | Point3DLike
        Ending point of the spring.
    segments : int
        Number of Segements the spring should have.
    buff : float
        Distance between start/end and the start of the first/last segment.
    spring_height : float
        Distance by which the points are offset. If the spring is going horizontally this would be half the height of the mobject.
    """
    def __init__(
        self,
        start: Mobject | Point3DLike = UP,
        end: Mobject | Point3DLike = DOWN,
        segments: int = 6,
        buff: float = 0.25,
        spring_height: float = 0.4,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.start = pointify(start)
        self.end = pointify(end)
        self.segments = segments
        self.buff = buff
        self.spring_height = spring_height

        points = [self.start, *self._get_subline_points(), self.end]
        self.set_points_as_corners(points)

    def _get_subline_points(self):
        tangent = normalize(self.end - self.start)
        normal = normalize(rotate_vector(tangent, PI/2))

        sub_start = self.start + tangent * self.buff
        sub_end = self.end - tangent * self.buff
        
        alphas = np.linspace(0,1,2*self.segments)
        points = [interpolate(sub_start, sub_end, alpha) for alpha in alphas]

        for i, point in enumerate(points[1:-1]):
            if i % 4 == 0:
                point += self.spring_height * normal
            elif i % 4 == 2:
                point -= self.spring_height * normal

        return points


def create_spring_sys(spring_attachment, pos, color=BLUE):
    spring = Spring(
        spring_attachment,
        pos,
        segments=7,
        buff=0.5,
        stroke_width=7,
        spring_height=0.3
    ).set_z_index(-1)

    mass = Square(0.75, color=WHITE).set_fill(opacity=1, color=color).move_to(pos)

    return VGroup(spring, mass)


# ---------------------------------------------
#   Simulations
# ---------------------------------------------

class SpringSystem(Scene):
    def construct(self):
        spring_attachment = Dot(2*UP)
        pos = Dot(DOWN).set_opacity(0)

        spring_sys = create_spring_sys(spring_attachment, pos)
        self.add(spring_attachment, spring_sys)

        dt = 1e-6
        t1 = 10
        amount = int(t1/dt)

        x = 1
        k = 5
        c = 0.5
        a = 0
        v = 0

        x_values = [x] * amount
        for i in range(amount):
            a = - k * x - c * v
            v += a * dt     # v = v + dv
            x += v * dt

            x_values[i] = x

        t = ValueTracker(0)
        t_values = np.arange(0,t1,dt)
        spring_pos = lambda t: desolv.interpolate_points(t, t_values, x_values)

        def spring_sys_updater(mob):
            new_pos = DOWN + spring_pos(t.get_value())*DOWN
            new_spring = create_spring_sys(spring_attachment, new_pos)
            mob.become(new_spring)

        spring_sys.add_updater(spring_sys_updater)
        
        t_anim = t1
        self.play(t.animate(run_time=t_anim, rate_func=linear).set_value(t_anim))


class CoupledSpringSystem(Scene):
    def construct(self):

        spring_attachment = Dot(3*UP)
        l = 2.5
        mid_point = 3*UP + l*DOWN
        end_point = 3*UP + 2*l*DOWN
        pos1 = Dot(mid_point).set_opacity(0)
        pos2 = Dot(end_point).set_opacity(0)

        spring_sys1 = create_spring_sys(spring_attachment, pos1)
        spring_sys2 = create_spring_sys(pos1, pos2, RED)

        self.add(spring_attachment, spring_sys1, spring_sys2)

        dt = 1e-6
        t1 = 3
        amount = int(t1/dt)

        k = 5
        x1, x2 = 1, 0
        a1, a2 = 0, 0
        v1, v2 = 0, 0

        x1_values = [x1] * amount
        x2_values = [x2] * amount
        for i in range(amount):
            # Simulate x1
            a1  = -k * x1 + k * (x2 - x1)
            v1 += a1 * dt
            x1 += v1 * dt

            # Simulate x2
            a2  = -k * (x2 - x1)
            v2 += a2 * dt
            x2 += v2 * dt

            x1_values[i] = x1
            x2_values[i] = x2

        t = ValueTracker(0)
        t_values = np.arange(0,t1,dt)
        spring1_pos = lambda t: desolv.interpolate_points(t, t_values, x1_values)
        spring2_pos = lambda t: desolv.interpolate_points(t, t_values, x2_values)

        def spring_sys1_updater(mob):
            new_pos = mid_point + spring1_pos(t.get_value())*DOWN
            new_spring = create_spring_sys(spring_attachment, new_pos)
            mob.become(new_spring)

        def spring_sys2_updater(mob):
            new_attach = mid_point + spring1_pos(t.get_value())*DOWN
            new_pos = end_point + spring2_pos(t.get_value())*DOWN
            new_spring = create_spring_sys(new_attach, new_pos, RED)
            mob.become(new_spring)

        spring_sys1.add_updater(spring_sys1_updater)
        spring_sys2.add_updater(spring_sys2_updater)
        
        self.play(t.animate(run_time=t1, rate_func=linear).set_value(t1))
