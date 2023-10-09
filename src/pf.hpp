#ifndef PF_HPP
#define PF_HPP

#define _USE_MATH_DEFINES

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <rclcpp/time.hpp>
#include <string>
#include <tuple>

#include "angle_helpers.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "helper_functions.hpp"
#include "nav2_msgs/msg/particle_cloud.hpp"

#include "occupancy_field.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
using std::placeholders::_1;

/**
 * Represents a hypothesis (particle) of the robot's pose consisting of x,y and
 * theta (yaw) Attributes: x: the x-coordinate of the hypothesis relative to the
 * map frame y: the y-coordinate of the hypothesis relative ot the map frame
 *      theta: the yaw of the hypothesis relative to the map frame
 *      w: the particle weight (the class does not ensure that particle weights
 * are normalized
 */
class Particle {
public:
  float w;
  float theta;
  float x;
  float y;

  /**
   * Construct a new Particle
   * x: the x-coordinate of the hypothesis relative to the map frame
   * y: the y-coordinate of the hypothesis relative ot the map frame
   * theta: the yaw of KeyboardInterruptthe hypothesis relative to the map
   * frame w: the particle weight (the class does not ensure that particle
   * weights are normalized
   */
  Particle(float w = 1.0, float theta = 0.0, float x = 0.0, float y = 0.0);
  /**
   * A helper function to convert a particle to a geometry_msgs/Pose message
   */
  geometry_msgs::msg::Pose as_pose();
  void update_position(Eigen::Matrix3d delta, float delta_angle);
  void add_noise();
  void update_weight(float new_w);
  Eigen::Matrix3Xd transform_scan_to_map(std::vector<float> r,
                                        std::vector<float> theta);
  Eigen::Matrix3d homogeneous_pose();
};

/**
 * Return a random sample of n elements from the set choices with the specified
 * probabilities
 * @param choices: the values to sample from represented as a list
 * @param probabilities: the probability of selecting each element in choices
 * represented as a list
 * @param n: the number of samples
 */
std::vector<Particle> draw_random_sample(std::vector<Particle> choices,
                                             std::vector<float> probabilities,
                                             unsigned int n);

/**
 * The class that represents a Particle Filter ROS Node
 * Attributes list:
 *      base_frame: the name of the robot base coordinate frame (should be
 "base_footprint" for most robots)
 *      map_frame: the name of the map coordinate frame (should be "map" in most
 cases)
 *      odom_frame: the name of the odometry coordinate frame (should be "odom"
 in most cases)
 *      scan_topic: the name of the scan topic to listen to (should be "scan" in
 most cases)
 *      n_particles: the number of particles in the filter
 *      d_thresh: the amount of linear movement before triggering a filter
 update
 *      a_thresh: the amount of angular movement before triggering a filter
 update
 *      pose_listener: a subscriber that listens for new approximate pose
 estimates (i.e. generated through the rviz GUI)
 *      particle_pub: a publisher for the particle cloud
 *      last_scan_timestamp: this is used to keep track of the clock when using
 bags
 *      scan_to_process: the scan that our run_loop should process next
 *      occupancy_field: this helper class allows you to query the map for
 distance to closest obstacle
 *      ransform_helper: this helps with various transform operations
 (abstracting away the tf2 module)
 *      particle_cloud: a list of particles representing a probability
 distribution over robot poses
 *      current_odom_xy_theta: the pose of the robot in the odometry frame when
 the last filter update was performed. The pose is expressed as a list
 [x,y,theta] (where theta is the yaw) thread: this thread runs your main loop
*/
class ParticleFilter : public rclcpp::Node {
public:
  ParticleFilter();

  void setup_helpers(std::shared_ptr<ParticleFilter> nodePtr);

private:
  /**
   * This function takes care of sending out the map to odom transform
   */
  void pub_latest_transform();

  /**
   * This is the main run_loop of our particle filter.  It checks to see if
   * any scans are ready and to be processed and will call several helper
   * functions to complete the processing.
   * ou do not need to modify this function, but it is helpful to understand
   * it.
   */
  void run_loop();

  bool moved_far_enough_to_update(std::vector<float> new_odom_xy_theta);

  /**
   * Update the estimate of the robot's pose given the updated particles.
   * There are two logical methods for this:
   * (1): compute the mean pose(2): compute the most likely pose (i.e. the
   * mode of the distribution)
   */
  void update_robot_pose();

  /**
   * Update the particles using the newly given odometry pose.
   * The function computes the value delta which is a tuple (x,y,theta)
   * that indicates the change in position and angle between the odometry
   * when the particles were last updated and the current odometry.
   */
  void update_particles_with_odom();

  /**
   * Resample the particles according to the new particle weights.
   * The weights stored with each particle should define the probability that
   * a particular particle is selected in the resampling step.  You may want
   * to make use of the given helper function draw_random_sample in
   * helper_functions.py.
   */
  void resample_particles();

  /**
   * Updates the particle weights in response to the scan data
   * r: the distance readings to obstacles
   * theta: the angle relative to the robot frame for each corresponding
   * reading
   */
  void update_particles_with_laser(std::vector<float> r,
                                   std::vector<float> theta);

  /**
   * Callback function to handle re-initializing the particle filter based on
   * a pose estimate. These pose estimates could be generated by another ROS
   * Node or could come from the rviz GUI
   */
  void update_initial_pose(geometry_msgs::msg::PoseWithCovarianceStamped msg);

  /**
   * Initialize the particle cloud.
   * Arguments
   *      xy_theta: a triple consisting of the mean x, y, and theta (yaw) to
   * initialize the particle cloud around.  If this input is omitted, the
   * odometry will be used
   */
  void initialize_particle_cloud(
      std::optional<std::vector<float>> xy_theta = std::nullopt);

  void normalize_particles();

  void publish_particles(rclcpp::Time timestamp);

  void scan_received(sensor_msgs::msg::LaserScan msg);

private:
  std::string base_frame;
  std::string map_frame;
  std::string odom_frame;
  std::string scan_topic;
  unsigned int n_particles;
  float d_thresh;
  float a_thresh;
  rclcpp::Publisher<nav2_msgs::msg::ParticleCloud>::SharedPtr particle_pub;
  std::optional<builtin_interfaces::msg::Time> last_scan_timestamp;
  std::optional<sensor_msgs::msg::LaserScan> scan_to_process;
  std::vector<float> current_odom_xy_theta;
  std::vector<Particle> particle_cloud;
  std::optional<geometry_msgs::msg::Pose> odom_pose;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr
      laserscan_subscriber;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initial_pose_subscriber;
  rclcpp::TimerBase::SharedPtr timer;
  std::shared_ptr<OccupancyField> occupancy_field;
  std::shared_ptr<TFHelper> transform_helper_;
};

#endif