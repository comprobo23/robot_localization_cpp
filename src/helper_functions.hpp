#ifndef HELPER_FUNCTIONS_HPP
#define HELPER_FUNCTIONS_HPP

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tuple>

#include "rclcpp/rclcpp.hpp"

/**
 * TFHelper Provides functionality to convert poses between various forms,
 * compare angles in a suitable way, and publish needed transforms to ROS
 */
class TFHelper {
public:
  std::optional<geometry_msgs::msg::Point> translation;
  std::optional<geometry_msgs::msg::Quaternion> rotation;

  TFHelper(std::shared_ptr<rclcpp::Node> node);

  /**
   * Convert from representation of a pose as translation and rotation
   * (Quaternion) tuples to a geometry_msgs/Pose message
   */
  geometry_msgs::msg::Pose
  convert_translation_rotation_to_pose(float translation[3], float rotation[4]);

  /**
   * Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple
   */
  std::vector<float> convert_pose_to_xy_theta(geometry_msgs::msg::Pose pose);

  /**
   * convenience function to map an angle to the range [-pi,pi]
   */
  float angle_normalize(float z);

  /**
   * Calculates the difference between angle a and angle b (both should
   * be in radians) the difference is always based on the closest
   * rotation from angle a to angle b.
   * examples:
   *      angle_diff(.1,.2) -> -.1
   *      angle_diff(.1, 2*math.pi - .1) -> .2
   *      angle_diff(.1, .2+2*math.pi) -> -.1
   */
  float angle_diff(float a, float b);

  /**
   * This method constantly updates the offset of the map and
   * odometry coordinate systems based on the latest results from
   * the localizer.
   * @param robot_pose: should be of type geometry_msgs/msg/Pose and represent
   *      the robot's position within the map
   * @param odom_pose: should be of type geometry_msgs/msg/Pose and represent
   *      the robot's position within the odometry coordinate system
   */
  void fix_map_to_odom_transform(geometry_msgs::msg::Pose robot_pose,
                                 geometry_msgs::msg::Pose odom_pose);

  void send_last_map_to_odom_transform(std::string odom_frame,
                                       std::string map_frame,
                                       builtin_interfaces::msg::Time timestamp);

  /**
   * Find the odometry position for a given timestamp.  We want to avoid
   * blocking, so if the transform is not ready, we return None.
   * @return: a tuple where the first element is the stamped transform and the
   * second element is the delta in time between the requested time and the
   * most recent transform available
   */
  std::tuple<std::optional<geometry_msgs::msg::Pose>,
             std::optional<std::chrono::nanoseconds>>
  get_matching_odom_pose(std::string odom_frame, std::string base_frame,
                         builtin_interfaces::msg::Time timestamp);

  /**
   * Convert the scan data to a polar representation in the robot frame.
   * The reason that we have to do this differently than in the warmup
   * project is that the Turtlebot4's lidar frame is not oriented the same as
   * the Neato. If you use the results in (r, theta) you will have the correct
   * angles and distances relative to the robot.
   *
   * Note: theta is in radians
   */
  std::tuple<std::vector<float>, std::vector<float>>
  convert_scan_to_polar_in_robot_frame(sensor_msgs::msg::LaserScan msg,
                                       std::string base_frame);

private:
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

#endif