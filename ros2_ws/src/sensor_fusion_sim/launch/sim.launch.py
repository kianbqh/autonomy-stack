"""Bring up the full stack: sim -> EKF -> BC policy -> cmd_vel loop, plus live metrics."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model = LaunchConfiguration("model_path")
    out = LaunchConfiguration("metrics_out")
    seed = LaunchConfiguration("seed")
    return LaunchDescription([
        DeclareLaunchArgument("model_path", default_value="models/bc_policy.npz"),
        DeclareLaunchArgument("metrics_out", default_value="results/ros2_metrics.json"),
        DeclareLaunchArgument("seed", default_value="7"),
        Node(package="sensor_fusion_sim", executable="sim_node", parameters=[{"seed": seed}], output="screen"),
        Node(package="sensor_fusion_sim", executable="ekf_node", output="screen"),
        Node(package="sensor_fusion_sim", executable="policy_node", parameters=[{"model_path": model}], output="screen"),
        Node(package="sensor_fusion_sim", executable="metrics_node", parameters=[{"out": out}], output="screen"),
    ])
