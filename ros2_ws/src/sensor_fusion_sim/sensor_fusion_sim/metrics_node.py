"""Live evaluation: RMSE of fused vs raw GPS against ground truth, topic rates, goals reached."""
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class MetricsNode(Node):
    def __init__(self):
        super().__init__("metrics_node")
        self.declare_parameter("out", "results/ros2_metrics.json")
        self.out = self.get_parameter("out").value
        self.truth = self.fused = self.gps = None
        self.e_f, self.e_g = [], []
        self.counts = {"imu/data": 0, "wheel/odometry": 0, "gps/fix_local": 0, "odometry/filtered": 0, "cmd_vel": 0}
        self.goal_ok = self.goal_fail = 0
        self.t0 = time.time()
        self.create_subscription(Odometry, "ground_truth/odom", self.on_truth, 50)
        self.create_subscription(Odometry, "odometry/filtered", self.on_fused, 50)
        self.create_subscription(PointStamped, "gps/fix_local", self.on_gps, 10)
        from std_msgs.msg import Bool
        self.create_subscription(Bool, "goal_result", self.on_result, 10)
        from sensor_msgs.msg import Imu
        from geometry_msgs.msg import Twist
        self.create_subscription(Imu, "imu/data", lambda m: self._count("imu/data"), 50)
        self.create_subscription(Odometry, "wheel/odometry", lambda m: self._count("wheel/odometry"), 50)
        self.create_subscription(Twist, "cmd_vel", lambda m: self._count("cmd_vel"), 10)
        self.create_timer(0.1, self.sample)
        self.create_timer(5.0, self.dump)

    def _count(self, k):
        self.counts[k] += 1

    def on_truth(self, m):
        self.truth = np.array([m.pose.pose.position.x, m.pose.pose.position.y])

    def on_fused(self, m):
        self._count("odometry/filtered")
        self.fused = np.array([m.pose.pose.position.x, m.pose.pose.position.y])

    def on_gps(self, m):
        self._count("gps/fix_local")
        self.gps = np.array([m.point.x, m.point.y])

    def on_result(self, m):
        if m.data:
            self.goal_ok += 1
        else:
            self.goal_fail += 1

    def sample(self):
        if self.truth is None or self.fused is None or self.gps is None:
            return
        self.e_f.append(np.sum((self.fused - self.truth) ** 2))
        self.e_g.append(np.sum((self.gps - self.truth) ** 2))

    def dump(self):
        el = time.time() - self.t0
        if not self.e_f:
            return
        res = {"wall_time_s": el, "samples_10hz": len(self.e_f),
               "rmse_fused_m": float(np.sqrt(np.mean(self.e_f))), "rmse_raw_gps_m": float(np.sqrt(np.mean(self.e_g))),
               "goals_reached": self.goal_ok, "goals_timed_out": self.goal_fail, "topic_rates_hz": {k: v / el for k, v in self.counts.items()}}
        path = Path(self.out)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w") as f:
                json.dump(res, f, indent=2)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if rclpy.ok():
            self.get_logger().info(f"t={el:.0f}s fused RMSE {res['rmse_fused_m']:.2f} m | raw GPS {res['rmse_raw_gps_m']:.2f} m"
                                   f" | goals ok/timeout {self.goal_ok}/{self.goal_fail}")


def main():
    rclpy.init()
    node = MetricsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # timeout and the ROS launcher may both send SIGINT. Finish the final
        # atomic snapshot without letting the second signal interrupt it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.dump()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
