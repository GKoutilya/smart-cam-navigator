import math
import unittest

from src.action.simulated_robot import SimulatedRobot

DT = 1.0 / 30.0


class TestSimulatedRobot(unittest.TestCase):
    def test_initial_position_at_origin(self):
        robot = SimulatedRobot()
        x, y = robot.get_position()
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_drives_toward_target_ahead(self):
        robot = SimulatedRobot()
        for _ in range(60):
            robot.command_towards(1.0, 0.0, DT)
            robot.step(DT)

        x, y = robot.get_position()
        self.assertGreater(x, 0.1)  # made real forward progress
        self.assertLess(abs(y), 0.05)  # stayed on the straight line

    def test_turns_in_place_rather_than_reversing_for_target_behind(self):
        robot = SimulatedRobot()
        robot.command_towards(-1.0, 0.0, DT)  # directly behind initial heading (angle=0)
        robot.step(DT)

        x, y = robot.get_position()
        self.assertLess(math.hypot(x, y), 0.01)  # didn't drive backward/sideways
        # Pure rotation (no forward component) tops out at the wheel-coupled
        # max rate: 2 * max_wheel_speed / wheel_base.
        max_rotation_rate = 2 * robot.max_wheel_speed / robot.wheel_base
        self.assertGreater(abs(robot.body.angular_velocity), max_rotation_rate * 0.9)

    def test_sharp_turn_reduces_achievable_forward_speed(self):
        # A real differential-drive robot can't turn sharply AND drive at full
        # speed simultaneously - both draw from the same max_wheel_speed budget.
        straight = SimulatedRobot()
        straight.command_towards(1.0, 0.0, DT)  # target directly ahead: no turn needed
        straight.step(DT)

        turning = SimulatedRobot()
        turning.command_towards(1.0, 1.0, DT)  # target off to the side: needs to turn
        turning.step(DT)

        self.assertGreater(straight.body.velocity.length, turning.body.velocity.length)

    def test_decelerates_and_settles_near_target(self):
        robot = SimulatedRobot(max_wheel_speed=2.0)
        for _ in range(300):
            robot.command_towards(0.3, 0.0, DT)
            robot.step(DT)

        x, y = robot.get_position()
        self.assertAlmostEqual(x, 0.3, delta=0.01)
        self.assertAlmostEqual(y, 0.0, delta=0.01)

    def test_collides_with_obstacle_instead_of_passing_through(self):
        robot = SimulatedRobot()
        robot.set_obstacles([(0.3, 0.0)])
        for _ in range(300):
            robot.command_towards(1.0, 0.0, DT)
            robot.step(DT)

        x, y = robot.get_position()
        clearance = robot.radius + robot.obstacle_radius
        self.assertGreaterEqual(math.hypot(x - 0.3, y), clearance - 0.03)
        self.assertLess(x, 0.3)  # never made it past the obstacle


if __name__ == '__main__':
    unittest.main()
