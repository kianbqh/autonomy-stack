"""EKF fusion node: IMU (predict) + wheel odometry + GPS (update) -> /odometry/filtered."""
import rclpy
import rclpy.executors
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu

from fusion_core.ekf import EKF, EKFConfig

from .common import fill_odom


class EKFNode(Node):
    def __init__(self):
        super().__init__("ekf_node")
        self.declare_parameter("dt", 0.01)
        self.dt = self.get_parameter("dt").value
        self.ekf = EKF(EKFConfig(), x0=[0, 0, 0, 0, 0], P0=[[0.01 if i == j else 0 for j in range(5)] for i in range(5)])
        self.pub = self.create_publisher(Odometry, "odometry/filtered", 50)
        self.create_subscription(Imu, "imu/data", self.on_imu, 50)
        self.create_subscription(Odometry, "wheel/odometry", self.on_wheel, 50)
        self.create_subscription(PointStamped, "gps/fix_local", self.on_gps, 10)

    def on_imu(self, m):
        self.ekf.predict(m.angular_velocity.z, m.linear_acceleration.x, self.dt)
        x, y, th, v, _ = self.ekf.x
        out = fill_odom(Odometry(), m.header.stamp, x, y, th, v)
        P = self.ekf.P
        cov = [0.0] * 36
        cov[0], cov[1], cov[6], cov[7], cov[35] = P[0, 0], P[0, 1], P[1, 0], P[1, 1], P[2, 2]
        out.pose.covariance = [float(c) for c in cov]
        self.pub.publish(out)

    def on_wheel(self, m):
        self.ekf.update_odom(m.twist.twist.linear.x)

    def on_gps(self, m):
        if not self.ekf.update_gps([m.point.x, m.point.y]):
            self.get_logger().warn("GPS outlier rejected by Mahalanobis gate")


def main():
    rclpy.init()
    node = EKFNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
