import math

import pymunk

# robot.radius + obstacle_radius must stay under Costmap's default inflation_radius
# (0.15, see src/planning/costmap.py) - otherwise a path the planner considers safe
# could still physically collide, since the planner treats the robot as a point.
DEFAULT_ROBOT_RADIUS = 0.05
DEFAULT_OBSTACLE_RADIUS = 0.08


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class SimulatedRobot:
    """A Pymunk-simulated differential-drive robot: real inertia, a turning
    radius, and physical collision with obstacles, instead of teleporting to
    whatever position it's commanded to.
    """

    def __init__(self, radius: float = DEFAULT_ROBOT_RADIUS,
                 obstacle_radius: float = DEFAULT_OBSTACLE_RADIUS,
                 wheel_base: float = 0.15,
                 max_wheel_speed: float = 0.5,
                 heading_gain: float = 4.0):
        self.radius = radius
        self.obstacle_radius = obstacle_radius
        self.wheel_base = wheel_base
        self.max_wheel_speed = max_wheel_speed
        self.heading_gain = heading_gain

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = (0.0, 0.0)
        shape = pymunk.Circle(self.body, radius)
        shape.elasticity = 0.0
        shape.friction = 0.5
        self.space.add(self.body, shape)

        self._obstacle_shapes = []

    def set_obstacles(self, points) -> None:
        for body, shape in self._obstacle_shapes:
            self.space.remove(body, shape)
        self._obstacle_shapes = []

        for x, y in points:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x, y)
            shape = pymunk.Circle(body, self.obstacle_radius)
            shape.elasticity = 0.0
            shape.friction = 0.5
            self.space.add(body, shape)
            self._obstacle_shapes.append((body, shape))

    def command_towards(self, target_x: float, target_y: float, dt: float) -> None:
        """Proportional go-to-goal differential-drive control: turn toward the
        target, then drive forward at a speed capped by distance-remaining and
        how well-aligned the heading already is (can't strafe or reverse).

        Linear and angular speed are coupled through a shared max_wheel_speed,
        like a real differential-drive robot's left/right wheel speed limits
        (v_left = v - w*L/2, v_right = v + w*L/2, both capped) - a sharp turn
        proportionally reduces how fast it can also move forward, rather than
        treating max linear and max angular velocity as independent limits
        that could both be maxed out simultaneously (which no real
        differential-drive robot can actually do).

        Applies the linear component as a force toward the desired velocity
        rather than overwriting body.velocity directly - Chipmunk's contact
        resolution only persists a collision's velocity correction into the
        *next* step, so directly re-assigning velocity every tick silently
        erases that correction and lets the body tunnel straight through
        obstacles. Routing the command through the force solver instead means
        it's resolved together with any contact constraints every step.
        """
        dx = target_x - self.body.position.x
        dy = target_y - self.body.position.y
        distance = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)

        heading_error = _normalize_angle(desired_heading - self.body.angle)
        alignment = max(0.0, math.cos(heading_error))

        raw_forward_speed = min(self.max_wheel_speed, distance) * alignment
        raw_angular_velocity = self.heading_gain * heading_error

        left_speed = raw_forward_speed - raw_angular_velocity * self.wheel_base / 2
        right_speed = raw_forward_speed + raw_angular_velocity * self.wheel_base / 2

        max_wheel_mag = max(abs(left_speed), abs(right_speed))
        if max_wheel_mag > self.max_wheel_speed:
            scale = self.max_wheel_speed / max_wheel_mag
            left_speed *= scale
            right_speed *= scale

        forward_speed = (left_speed + right_speed) / 2
        angular_velocity = (right_speed - left_speed) / self.wheel_base

        desired_velocity = (forward_speed * math.cos(self.body.angle),
                             forward_speed * math.sin(self.body.angle))
        current_velocity = self.body.velocity
        force = (
            (desired_velocity[0] - current_velocity[0]) * self.body.mass / dt,
            (desired_velocity[1] - current_velocity[1]) * self.body.mass / dt,
        )
        self.body.apply_force_at_world_point(force, self.body.position)
        self.body.angular_velocity = angular_velocity

    def step(self, dt: float) -> None:
        self.space.step(dt)

    def get_position(self):
        return (self.body.position.x, self.body.position.y)

    def get_pose(self):
        return (self.body.position.x, self.body.position.y, self.body.angle)
