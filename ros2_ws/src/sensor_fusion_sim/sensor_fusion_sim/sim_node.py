"""Simulated robot: integrates /cmd_vel at 100 Hz and publishes noisy IMU, wheel odometry and GPS."""
import numpy as np
import rclpy
import rclpy.executors
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from fusion_core.sim import RobotSim, SensorConfig

from .common import fill_odom


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")
        self.declare_parameter("seed", 7)
        self.declare_parameter("goal_radius", 0.75)
        self.declare_parameter("goal_timeout", 40.0)
        self.timeout = self.get_parameter("goal_timeout").value
        seed = self.get_parameter("seed").value
        self.r_goal = self.get_parameter("goal_radius").value
        self.cfg = SensorConfig()
        self.sim = RobotSim(self.cfg, seed=seed, x0=[0.0, 0.0, 0.0, 0.0])
        self.rng = np.random.default_rng(seed + 1)
        self.v_cmd, self.w_cmd = 0.0, 0.0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_imu = self.create_publisher(Imu, "imu/data", 50)
        self.pub_wheel = self.create_publisher(Odometry, "wheel/odometry", 50)
        self.pub_gps = self.create_publisher(PointStamped, "gps/fix_local", 10)
        self.pub_truth = self.create_publisher(Odometry, "ground_truth/odom", 50)
        self.pub_goal = self.create_publisher(PointStamped, "goal", latched)
        self.pub_result = self.create_publisher(Bool, "goal_result", 10)  # True = parked at goal, False = timeout
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.new_goal()
        self.create_timer(self.cfg.dt, self.tick)

    def new_goal(self):
        ang, d = self.rng.uniform(-np.pi, np.pi), self.rng.uniform(8, 20)
        self.goal = self.sim.state[:2] + d * np.array([np.cos(ang), np.sin(ang)])
        self.goal_t0 = self.sim.t
        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "odom"
        m.point.x, m.point.y = float(self.goal[0]), float(self.goal[1])
        self.pub_goal.publish(m)

    def on_cmd(self, msg):
        self.v_cmd, self.w_cmd = msg.linear.x, msg.angular.z

    def tick(self):
        a = float(np.clip(2.0 * (self.v_cmd - self.sim.state[3]), -2.0, 2.0))
        meas = self.sim.step(a, self.w_cmd)
        stamp = self.get_clock().now().to_msg()
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, "base_link"
        imu.angular_velocity.z, imu.linear_acceleration.x = meas["imu"]
        self.pub_imu.publish(imu)
        if "odom" in meas:
            w = Odometry()
            w.header.stamp, w.header.frame_id = stamp, "base_link"
            w.twist.twist.linear.x = float(meas["odom"])
            self.pub_wheel.publish(w)
        if "gps" in meas:
            g = PointStamped()
            g.header.stamp, g.header.frame_id = stamp, "odom"
            g.point.x, g.point.y = float(meas["gps"][0]), float(meas["gps"][1])
            self.pub_gps.publish(g)
        x, y, th, v = self.sim.state
        self.pub_truth.publish(fill_odom(Odometry(), stamp, x, y, th, v))
        if np.hypot(*(self.goal - self.sim.state[:2])) < self.r_goal and v < 0.3:
            self.get_logger().info(f"goal reached in {self.sim.t - self.goal_t0:.1f}s -> new goal")
            self.pub_result.publish(Bool(data=True))
            self.new_goal()
        elif self.sim.t - self.goal_t0 > self.timeout:
            self.get_logger().info("goal timeout -> new goal")
            self.pub_result.publish(Bool(data=False))
            self.new_goal()


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
