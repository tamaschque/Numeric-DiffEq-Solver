# Numeric Simulations of Physics Problems

This repo contains the source code of most animations from my video on simulating physics Problems. As the series continues this repo will be updates as well. 

<p align="center">
  <img src="Animations/ThreeBodyProblem_Periodic.gif" alt="Single Pendulum" width="300">
  &nbsp;&nbsp;&nbsp;
  <img src="Animations/ChaoticLorenzAttractor.gif" alt="Single Pendulum" width="300">
</p>

## Watch the Videos here:
<p align="center">
    <a href="https://www.youtube.com/watch?v=M_OOwhA2fY8">
    <img src="https://img.youtube.com/vi/M_OOwhA2fY8/hqdefault.jpg" width="300" />
    </a>
    <a href="https://www.youtube.com/watch?v=_apHxVJU61g">
    <img src="https://img.youtube.com/vi/_apHxVJU61g/hqdefault.jpg" width="300" />
    </a>
</p>

## Solvers
The methods covered in the videos are:
 - Ordinary Differential Equations
    - Euler Method
    - Improved/Modified Euler
    - Runge-Kutta 4
    - Runge-Kutta 4(5)
 - Partial Differential Equations
    - Method of Lines
    - Crank-Nicolson

As a disclaimer: The focus of these implementations are readability over efficiency. For actual usage you should use libraries like scipy.

## Part 1 - ODEs
The first video covers the basic idea behind numerical simulations and only covers ordinary differential equations.

The visualization of the data was done using Manim Community Edition and Blender. It mostly consisted of importing the y- and t-values from the cache transforming them into an interpolated function using *interpolate_points*. This way the movement of the body could be driven by a *ValueTracker* serving as the time. See the example below:

```
class SinglePendulum(Scene):
    def construct(self):

        time = ValueTracker()

        cache_path = "single_pendulum.json"
        t_values, y_values = numeric_de_solver.load_chached_result(cache_path)

        pend_func = lambda t: numeric_de_solver.interpolate_points(t, t_values, y_values)

        pend = always_redraw(lambda: 
            Pendulum(
                attach_pos=2*UP+4*LEFT,
                radius=0.4,
                angle=pend_func(time.get_value())[0]
            )
        )

        vert_line = DashedLine(3*UP+4*LEFT, 2*DOWN+4*LEFT, color=GRAY).set_z_index(-1)

        angle_arc = always_redraw(lambda:
            Arc(
                1.5,
                3/2*PI,
                pend_func(time.get_value())[0],
                color=BLUE_C
            ).move_arc_center_to(pend.attach_pos)
        )

        self.add(vert_line, angle_arc, pend)
        t_anim = 40
        self.play(time.animate(run_time=t_anim, rate_func=linear).set_value(t_anim))
        self.wait()

```

<div align="center">
    <img src="Animations/SinglePendulum.gif" alt="Single Pendulum" width="900">
</div>

## Part 2 - Interactive Simulations
Part 2 doesn't really expand the scope of the differential equations covered but explains how to simulate in real time and make these simulations interactable.

The related folder contains the source code for the interactive simulations covered in the video:

 * Springed Double Pendulum (can be pushed)
 * Charged Double Pendulum (interacts with charged mouse cursor)
 * 3 Body Problem (mouse cursor controlls one of the bodies)
 
You will need to install the pyglet library to run these files.

## Part 3 - PDEs
Part 3 talks about partial differential equations with a big focus on the heat equation. The insights gained from simulation the heat equation are then applied to solve the quantum mechanical Schrödinger Equation

Both equations are covered in 1D and 2D with interactivity.

## Helpful Links
Here are some of the websites, videos, ... that helped me a lot during the making of this video.
 - Quick summary of numerically solving ODEs - [Link](https://www.youtube.com/watch?v=A1JnGhaVJsQ)
 - Explanation of Runge-Kutta method of order 2 - [Link](https://www.youtube.com/watch?v=bSs2Sj5Qi8I)
 - Inital values for periodic solutions of the three body problem - [Link](https://www.youtube.com/watch?v=8_RRZcqBEAc)
 - Portfolio of Alexis F. Espinoza Q. - contains slides and source code explaining the Crank-Nicolson method - [Link](https://alexisfespinozaq.github.io/aespinoza-physics-portfolio/)
 - Solving the 1D Heat Equation using Crank-Nicolson [Link](https://www.youtube.com/watch?v=y0C3ew3tk2A)

The Wikipedia links of the used Method can be found in the source code.

 ## Youtube Channels:
  - ## [tamaschque](https://www.youtube.com/@tamaschque) (Main)


  - ## [tama](https://www.youtube.com/@tamasque) (Second)
