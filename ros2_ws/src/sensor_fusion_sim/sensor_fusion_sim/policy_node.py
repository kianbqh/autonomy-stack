"""Behavior-cloned go-to-goal policy. Pure-NumPy MLP inference on weights exported from PyTorch."""
import numpy as np
import rclpy
import rclpy.executors
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from fusion_core.ekf import wrap_angle
from fusion_core.numpy_policy import NumpyMLP

from .common import quat_to_yaw

V_MAX, W_MAX = 1.5, 1.5


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("model_path", "models/bc_policy.npz")
        self.policy = NumpyMLP(self.get_parameter("model_path").value)
        self.pose, self.goal = None, None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, "odometry/filtered", self.on_pose, 50)
        self.create_subscription(PointStamped, "goal", self.on_goal, latched)
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_timer(0.05, self.tick)  # 20 Hz control

    def on_pose(self, m):
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, quat_to_yaw(p.orientation), m.twist.twist.linear.x)

    def on_goal(self, m):
        self.goal = (m.point.x, m.point.y)

    def tick(self):
        if self.pose is None or self.goal is None:
            return
        x, y, th, v = self.pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        b = wrap_angle(np.arctan2(dy, dx) - th)
        obs = np.array([np.hypot(dx, dy) / 10.0, np.cos(b), np.sin(b), v / 2.0])
        a = self.policy(obs)
        cmd = Twist()
        cmd.linear.x = float((a[0] + 1) / 2 * V_MAX)
        cmd.angular.z = float(a[1] * W_MAX)
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
