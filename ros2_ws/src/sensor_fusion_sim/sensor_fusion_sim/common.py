"""Shared helpers. The algorithmic core lives in the ROS-free `fusion_core` package (on PYTHONPATH)."""
import math


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def fill_odom(msg, stamp, x, y, yaw, v, frame="odom", child="base_link"):
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.child_frame_id = child
    msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
    qx, qy, qz, qw = yaw_to_quat(float(yaw))
    o = msg.pose.pose.orientation
    o.x, o.y, o.z, o.w = qx, qy, qz, qw
    msg.twist.twist.linear.x = float(v)
    return msg
