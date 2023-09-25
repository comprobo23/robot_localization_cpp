#include "helper_functions.hpp"

#include <algorithm>
#include <builtin_interfaces/msg/time.hpp>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <random>
#include <tuple>
#include <vector>

#include "angle_helpers.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pf.hpp"

geometry_msgs::msg::Pose
stamped_transform_to_pose(geometry_msgs::msg::TransformStamped t) {
  auto transform = t.transform;
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation.x;
  pose.position.y = transform.translation.y;
  pose.position.z = transform.translation.z;
  pose.orientation.x = transform.rotation.x;
  pose.orientation.y = transform.rotation.y;
  pose.orientation.z = transform.rotation.z;
  pose.orientation.w = transform.rotation.w;

  return pose;
}

TFHelper::TFHelper(std::shared_ptr<rclcpp::Node> node) {
  // logger = node->get_logger();
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*node);
  translation.reset();
  rotation.reset();
}

geometry_msgs::msg::Pose
TFHelper::convert_translation_rotation_to_pose(float translation[3], float rotation[4]) {
  geometry_msgs::msg::Pose pose;

  pose.position.x = translation[0];
  pose.position.y = translation[1];
  pose.position.z = translation[2];
  pose.orientation.x = rotation[0];
  pose.orientation.y = rotation[1];
  pose.orientation.z = rotation[2];
  pose.orientation.w = rotation[3];

  return pose;
}

std::vector<float>
TFHelper::convert_pose_to_xy_theta(geometry_msgs::msg::Pose pose) {
  auto angles = euler_from_quaternion(pose.orientation);
  std::vector<float> ret;
  ret.push_back(pose.position.x);
  ret.push_back(pose.position.y);
  ret.push_back(angles[2]);
  return ret;
}

float TFHelper::angle_normalize(float z) { return atan2(sin(z), cos(z)); }

float TFHelper::angle_diff(float a, float b) {
  a = angle_normalize(a);
  b = angle_normalize(b);
  auto d1 = a - b;
  // TODO: pi
  auto d2 = 2 * 3.141592 - abs(d1);
  if (d1 > 0) {
    d2 *= -1.0;
  }
  if (abs(d1) < abs(d2)) {
    return d1;
  }
  return d2;
}

void TFHelper::fix_map_to_odom_transform(geometry_msgs::msg::Pose robot_pose,
                                         geometry_msgs::msg::Pose odom_pose) {
  Eigen::Quaterniond robot_rotation(
      robot_pose.orientation.w, robot_pose.orientation.x,
      robot_pose.orientation.y, robot_pose.orientation.z);

  Eigen::Vector3d robot_translation(
      robot_pose.position.x, robot_pose.position.y, robot_pose.position.z);
  Eigen::Matrix4d robot_transformation = Eigen::Matrix4d::Identity();
  robot_transformation.block(0, 0, 3, 3) =
      robot_rotation.normalized().toRotationMatrix();
  robot_transformation.block(0, 3, 3, 1) = robot_translation;

  Eigen::Quaterniond odom_rotation(
      odom_pose.orientation.w, odom_pose.orientation.x, odom_pose.orientation.y,
      odom_pose.orientation.z);

  Eigen::Vector3d odom_translation(odom_pose.position.x, odom_pose.position.y,
                                   odom_pose.position.z);
  Eigen::Matrix4d odom_transformation = Eigen::Matrix4d::Identity();
  odom_transformation.block(0, 0, 3, 3) =
      odom_rotation.normalized().toRotationMatrix();
  odom_transformation.block(0, 3, 3, 1) = odom_translation;

  Eigen::Matrix4d odom_to_map =
      robot_transformation * odom_transformation.inverse();
  Eigen::Matrix3d odom_to_map_rotation = odom_to_map.block<3, 3>(0, 0);
  Eigen::Quaterniond odom_to_map_eigen_quat(odom_to_map_rotation);

  geometry_msgs::msg::Quaternion odom_to_map_geom_quat;
  odom_to_map_geom_quat.x = odom_to_map_eigen_quat.x();
  odom_to_map_geom_quat.y = odom_to_map_eigen_quat.y();
  odom_to_map_geom_quat.z = odom_to_map_eigen_quat.z();
  odom_to_map_geom_quat.w = odom_to_map_eigen_quat.w();

  geometry_msgs::msg::Point odom_to_map_trans;
  odom_to_map_trans.x = odom_to_map(0, 3);
  odom_to_map_trans.y = odom_to_map(1, 3);
  odom_to_map_trans.z = odom_to_map(2, 3);

  translation = odom_to_map_trans;
  rotation = odom_to_map_geom_quat;
}

void TFHelper::send_last_map_to_odom_transform(
    std::string odom_frame, std::string map_frame,
    builtin_interfaces::msg::Time timestamp) {
  if (!translation.has_value() || !rotation.has_value()) {
    return;
  }
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = timestamp;
  transform.header.frame_id = map_frame;
  transform.child_frame_id = odom_frame;
  transform.transform.translation.x = translation.value().x;
  transform.transform.translation.y = translation.value().y;
  transform.transform.translation.z = translation.value().z;
  transform.transform.rotation.x = rotation.value().x;
  transform.transform.rotation.y = rotation.value().y;
  transform.transform.rotation.z = rotation.value().z;
  transform.transform.rotation.w = rotation.value().w;
  tf_broadcaster_->sendTransform(transform);
}

std::tuple<std::optional<geometry_msgs::msg::Pose>,
           std::optional<std::chrono::nanoseconds>>
TFHelper::get_matching_odom_pose(std::string odom_frame, std::string base_frame,
                                 builtin_interfaces::msg::Time timestamp) {
  if (tf_buffer_->canTransform(odom_frame, base_frame, timestamp)) {
    std::chrono::duration<int> zero(0);
    // we can get the pose at the exact right time
    return std::make_tuple(
        stamped_transform_to_pose(
            tf_buffer_->lookupTransform(odom_frame, base_frame, timestamp)),
        zero);
  } else if (tf_buffer_->canTransform(odom_frame, base_frame, rclcpp::Time())) {
    auto most_recent =
        tf_buffer_->lookupTransform(odom_frame, base_frame, rclcpp::Time());
    auto timestamp_tp = std::chrono::time_point<std::chrono::system_clock>(
        std::chrono::seconds(timestamp.sec) +
        std::chrono::nanoseconds(timestamp.nanosec));
    auto most_recent_tp = std::chrono::time_point<std::chrono::system_clock>(
        std::chrono::seconds(most_recent.header.stamp.sec) +
        std::chrono::nanoseconds(most_recent.header.stamp.nanosec));
    std::chrono::nanoseconds dt = timestamp_tp - most_recent_tp;
    return std::make_tuple(std::nullopt, dt);
  } else {
    return std::make_tuple(std::nullopt, std::nullopt);
  }
}

std::tuple<std::vector<float>, std::vector<float>>
TFHelper::convert_scan_to_polar_in_robot_frame(sensor_msgs::msg::LaserScan msg,
                                               std::string base_frame) {
  auto laser_pose = stamped_transform_to_pose(tf_buffer_->lookupTransform(
      base_frame, msg.header.frame_id, rclcpp::Time()));

  auto qx = laser_pose.orientation.x;
  auto qy = laser_pose.orientation.y;
  auto qz = laser_pose.orientation.z;
  auto qw = laser_pose.orientation.w;

  double yaw = std::atan2(2.0 * (qx * qy + qw * qz),
                          qw * qw + qx * qx - qy * qy - qz * qz);
  auto num_meas = msg.ranges.size();
  auto ranges = std::vector<float>();
  auto angles = std::vector<float>();

  for (unsigned int i = 0; i < num_meas; i++) {
    ranges.push_back(msg.ranges[i]);
    angles.push_back(msg.angle_min + yaw +
                     ((msg.angle_max - msg.angle_min) / num_meas * i));
  }

  return std::make_tuple(ranges, angles);
}
