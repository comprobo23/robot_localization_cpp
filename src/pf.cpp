#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <random>
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
#include "pf.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
using std::placeholders::_1;

Particle::Particle(float w, float theta, float x, float y) {
  this->w = w;
  this->theta = theta;
  this->x = x;
  this->y = y;
}

/**
 * A helper function to convert a particle to a geometry_msgs/Pose message
 */
geometry_msgs::msg::Pose Particle::as_pose() {
  geometry_msgs::msg::Pose pose = geometry_msgs::msg::Pose();
  pose.position.x = this->x;
  pose.position.y = this->y;
  pose.orientation = quaternion_from_euler(0, 0, this->theta);

  return pose;
}

void Particle::update_position(Eigen::Matrix3d delta, float delta_angle) {
  auto particle_delta = this->homogeneous_pose() * delta;
  this->x = particle_delta(0, 2);
  this->y = particle_delta(1, 2);
  this->theta += delta_angle;
}
void Particle::add_noise() {
  std::normal_distribution<float> dist(0.0, 0.05);
  std::default_random_engine gen;

  this->x += dist(gen);
  this->y += dist(gen);
  this->theta += dist(gen);
}
void Particle::update_weight(float new_w) { this->w = new_w; }
Eigen::Matrix3Xd Particle::transform_scan_to_map(std::vector<float> r,
                                                 std::vector<float> theta) {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> ones;
  Eigen::Matrix3Xd scan(3, r.size());

  for (unsigned int i = 0; i < r.size(); i++) {
    scan(0, i) = r[i] * cos(theta[i]);
    scan(1, i) = r[i] * sin(theta[i]);
    scan(2, i) = 1;
  }
  return this->homogeneous_pose() * scan;
}
Eigen::Matrix3d Particle::homogeneous_pose() {
  Eigen::Matrix3d pose{{cos(this->theta), -sin(this->theta), this->x},
                       {sin(this->theta), cos(this->theta), this->y},
                       {0, 0, 1}};
  return pose;
}

std::vector<Particle> draw_random_sample(std::vector<Particle> choices,
                                         std::vector<float> probabilities,
                                         unsigned int n) {
  if (choices.empty() || probabilities.empty() || n == 0 ||
      choices.size() != probabilities.size()) {
    // Handle invalid inputs
    return {};
  }

  std::vector<Particle> samples;
  std::vector<float> bins;

  // Initialize random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(probabilities.begin(), probabilities.end());

  for (unsigned int i = 0; i < n; i++) {
    samples.push_back(choices[d(gen)]);
  }

  return samples;
}

ParticleFilter::ParticleFilter() : Node("pf") {
  base_frame = "base_footprint"; // the frame of the robot base
  map_frame = "map";             // the name of the map coordinate frame
  odom_frame = "odom";           // the name of the odometry coordinate frame
  scan_topic = "scan";           // the topic where we will get laser scans from

  n_particles = 1000; // the number of particles to use

  d_thresh = 0.2; // the amount of linear movement before performing an update
  a_thresh =
      M_PI / 6; // the amount of angular movement before performing an update

  // pose_listener responds to selection of a new approximate robot
  // location (for instance using rviz)
  auto sub1_opt = rclcpp::SubscriptionOptions();
  sub1_opt.callback_group =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  initial_pose_subscriber =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "initialpose", 10,
          std::bind(&ParticleFilter::update_initial_pose, this, _1), sub1_opt);

  // publish the current particle cloud.  This enables viewing particles
  // in rviz.
  particle_pub = this->create_publisher<nav2_msgs::msg::ParticleCloud>(
      "particle_cloud", 10);

  auto sub2_opt = rclcpp::SubscriptionOptions();
  sub2_opt.callback_group =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  // laser_subscriber listens for data from the lidar
  laserscan_subscriber = this->create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, 10, std::bind(&ParticleFilter::scan_received, this, _1),
      sub2_opt);

  // this is used to keep track of the timestamps coming from bag files
  // knowing this information helps us set the timestamp of our map ->
  // odom transform correctly
  last_scan_timestamp.reset();
  // this is the current scan that our run_loop should process
  scan_to_process.reset();

  timer = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&ParticleFilter::pub_latest_transform, this));
}

void ParticleFilter::pub_latest_transform() {
  if (last_scan_timestamp.has_value()) {
    rclcpp::Time last_scan_time(last_scan_timestamp.value());
    rclcpp::Duration offset(0, 100000000);
    auto postdated_timestamp = last_scan_time + offset;
    transform_helper_->send_last_map_to_odom_transform(map_frame, odom_frame,
                                                       postdated_timestamp);
  }
}

void ParticleFilter::run_loop() {
  if (!scan_to_process.has_value()) {
    return;
  }
  auto msg = scan_to_process.value();
  std::tuple<std::optional<geometry_msgs::msg::Pose>,
             std::optional<std::chrono::nanoseconds>>
      matching_odom_pose = transform_helper_->get_matching_odom_pose(
          odom_frame, base_frame, msg.header.stamp);
  auto new_pose = std::get<0>(matching_odom_pose);
  auto dt = std::get<1>(matching_odom_pose);
  if (!new_pose.has_value()) {
    // we were unable to get the pose of the robot corresponding to the
    // scan timestamp
    if (dt.has_value() && dt.value() < std::chrono::nanoseconds(0)) {
      //  we will never get this transform, since it is before our
      //  oldest one
      scan_to_process.reset();
    }
    return;
  }
  auto polar_coord =
      transform_helper_->convert_scan_to_polar_in_robot_frame(msg, base_frame);
  auto r = std::get<0>(polar_coord);
  auto theta = std::get<1>(polar_coord);
  // clear the current scan so that we can process the next one
  scan_to_process.reset();
  odom_pose = new_pose;
  auto new_odom_xy_theta =
      transform_helper_->convert_pose_to_xy_theta(odom_pose.value());
  if (current_odom_xy_theta.size() == 0) {
    current_odom_xy_theta = new_odom_xy_theta;
  } else if (particle_cloud.size() == 0) {
    // now that we have all of the necessary transforms we can update
    // the particle cloud
    initialize_particle_cloud();
  } else if (moved_far_enough_to_update(new_odom_xy_theta)) {
    // we have moved far enough to do an update!
    update_particles_with_odom(); // update based on odometry
    update_particles_with_laser(r,
                                theta); // update based on laser scan
    update_robot_pose();  // update robot's pose based on particles
    resample_particles(); // resample particles to focus on areas of high
                          // density
  }

  // publish particles (so things like rviz can see them)
  publish_particles(msg.header.stamp);
}

bool ParticleFilter::moved_far_enough_to_update(
    std::vector<float> new_odom_xy_theta) {
  return abs(new_odom_xy_theta[0] - current_odom_xy_theta[0] > d_thresh ||
             abs(new_odom_xy_theta[1] - current_odom_xy_theta[1]) > d_thresh ||
             abs(new_odom_xy_theta[2] - current_odom_xy_theta[2]) > a_thresh);
}

void ParticleFilter::update_robot_pose() {
  // first make sure that the particle weights are normalized
  this->normalize_particles();

  // TODO: assign the latest pose into self.robot_pose as a
  // geometry_msgs.Pose object just to get started we will fix the robot's
  // pose to always be at the origin
  Particle best_particle = this->particle_cloud[0];
  for (unsigned int i = 0; i < this->particle_cloud.size(); i++) {
    if (this->particle_cloud[i].w > best_particle.w) {
      best_particle = this->particle_cloud[i];
    }
  }

  float particle_pos[3] = {best_particle.x, best_particle.y, 0.0};
  auto particle_quat = quaternion_from_euler(0, 0, best_particle.theta);
  float particle_quat_arr[4] = {(float)particle_quat.x, (float)particle_quat.y,
                                (float)particle_quat.z, (float)particle_quat.w};
  auto robot_pose =
      this->transform_helper_->convert_translation_rotation_to_pose(
          particle_pos, particle_quat_arr);
  this->transform_helper_->fix_map_to_odom_transform(robot_pose,
                                                     this->odom_pose.value());
}

void ParticleFilter::update_particles_with_odom() {
  auto new_odom_xy_theta =
      transform_helper_->convert_pose_to_xy_theta(odom_pose.value());

  if (current_odom_xy_theta.size() != 3) {
    current_odom_xy_theta = new_odom_xy_theta;
    return;
  }

  // compute the change in x,y,theta since our last update
  auto current_odom_pose = Eigen::Matrix3d{
      {cos(this->current_odom_xy_theta[2]),
       -sin(this->current_odom_xy_theta[2]), this->current_odom_xy_theta[0]},
      {sin(this->current_odom_xy_theta[2]), cos(this->current_odom_xy_theta[2]),
       this->current_odom_xy_theta[1]},
      {0, 0, 1}};
  auto new_odom_pose =
      Eigen::Matrix3d{{cos(new_odom_xy_theta[2]), -sin(new_odom_xy_theta[2]),
                       new_odom_xy_theta[0]},
                      {sin(new_odom_xy_theta[2]), cos(new_odom_xy_theta[2]),
                       new_odom_xy_theta[1]},
                      {0, 0, 1}};
  auto delta = current_odom_pose.inverse() * new_odom_pose;
  auto delta_angle = new_odom_xy_theta[2] - this->current_odom_xy_theta[2];
  this->current_odom_xy_theta = new_odom_xy_theta;

  for (int i = this->particle_cloud.size() - 1; i >= 0; i--) {
    this->particle_cloud[i].update_position(delta, delta_angle);
    if (this->occupancy_field->get_closest_obstacle_distance(
            this->particle_cloud[i].x, this->particle_cloud[i].y) > 1000) {
      this->particle_cloud.erase(this->particle_cloud.begin() + i);
    }
  }
}

void ParticleFilter::resample_particles() {
  // make sure the distribution is normalized
  this->normalize_particles();

  std::vector<float> probabilities;
  for (unsigned int i = 0; i < this->particle_cloud.size(); i++) {
    probabilities.push_back(this->particle_cloud[i].w);
  }
  this->particle_cloud = draw_random_sample(this->particle_cloud,
  probabilities,
                                            this->n_particles);

  // for (unsigned int i = 0; i < this->particle_cloud.size(); i++) {
  //   this->particle_cloud[i].add_noise();
  // }

  this->normalize_particles();
}

void ParticleFilter::update_particles_with_laser(std::vector<float> r,
                                                 std::vector<float> theta) {
  for (unsigned int i = 0; i < this->particle_cloud.size(); i++) {
    auto laser_scan_map_frame =
        this->particle_cloud[i].transform_scan_to_map(r, theta);

    double avg_dist = 0.0;
    for (unsigned int i = 0; i < r.size(); i++) {
      avg_dist += this->occupancy_field->get_closest_obstacle_distance(
                      laser_scan_map_frame(0, i), laser_scan_map_frame(1, i)) /
                  r.size();
    }
    // Update the particle's weight based on the gaussian of the average
    // distance
    auto gaussian =
        1 / (0.2 * pow((2 * M_PI), 0.5)) * exp(-0.5 * pow((avg_dist / 0.2), 2));

    this->particle_cloud[i].update_weight(gaussian);
  }
}

void ParticleFilter::update_initial_pose(
    geometry_msgs::msg::PoseWithCovarianceStamped msg) {
  auto xy_theta = transform_helper_->convert_pose_to_xy_theta(msg.pose.pose);
  initialize_particle_cloud(xy_theta);
}

void ParticleFilter::initialize_particle_cloud(
    std::optional<std::vector<float>> xy_theta) {
  if (!xy_theta.has_value()) {
    xy_theta = transform_helper_->convert_pose_to_xy_theta(odom_pose.value());
  }

  auto map_bounds = this->occupancy_field->get_obstacle_bounding_box();
  std::default_random_engine gen;
  std::uniform_real_distribution<float> x_dist(map_bounds[0], map_bounds[1]);
  std::uniform_real_distribution<float> y_dist(map_bounds[2], map_bounds[3]);
  std::uniform_real_distribution<float> theta_dist(0, 2 * M_PI);

  while (this->particle_cloud.size() < this->n_particles) {
    Particle new_particle;
    new_particle.x = x_dist(gen);
    new_particle.y = y_dist(gen);
    new_particle.theta = theta_dist(gen);
    new_particle.w = 1.0;

    if (this->occupancy_field->get_closest_obstacle_distance(
            new_particle.x, new_particle.y) < 1000) {
      this->particle_cloud.push_back(new_particle);
    }
  }

  this->normalize_particles();
  this->update_robot_pose();
}

void ParticleFilter::normalize_particles() {
  auto weight_total = 0.0;
  for (unsigned int i = 0; i < this->particle_cloud.size(); i++) {
    weight_total += this->particle_cloud[i].w;
  }
  if (weight_total != 0.0) {
    for (unsigned int i = 0; i < this->particle_cloud.size(); i++) {
      this->particle_cloud[i].update_weight(this->particle_cloud[i].w /
                                            weight_total);
    }
  } else {
    for (unsigned int i = 0; i < this->particle_cloud.size(); i++) {
      this->particle_cloud[i].update_weight(1 / this->particle_cloud.size());
    }
  }
}

void ParticleFilter::publish_particles(rclcpp::Time timestamp) {
  this->normalize_particles();
  nav2_msgs::msg::ParticleCloud msg;
  msg.header.stamp = timestamp;
  msg.header.frame_id = map_frame;

  for (unsigned int i = 0; i < particle_cloud.size(); i++) {
    nav2_msgs::msg::Particle converted;
    converted.weight = particle_cloud[i].w;
    converted.pose = particle_cloud[i].as_pose();
    msg.particles.push_back(converted);
  }

  // actually send the message so that we can view it in rviz
  particle_pub->publish(msg);
}

void ParticleFilter::scan_received(sensor_msgs::msg::LaserScan msg) {
  last_scan_timestamp = msg.header.stamp;
  /**
   * we throw away scans until we are done processing the previous scan
   * self.scan_to_process is set to None in the run_loop
   */
  if (!scan_to_process.has_value()) {
    scan_to_process = msg;
  }
  // call run_loop to see if we need to update our filter, this will prevent
  // more scans from coming in
  run_loop();
}

void ParticleFilter::setup_helpers(std::shared_ptr<ParticleFilter> nodePtr) {
  occupancy_field = std::make_shared<OccupancyField>(OccupancyField(nodePtr));
  std::cout << "done generating occupancy field" << std::endl;
  transform_helper_ = std::make_shared<TFHelper>(TFHelper(nodePtr));
  std::cout << "done generating TFHelper" << std::endl;
}

int main(int argc, char **argv) {
  // this is useful to give time for the map server to get ready...
  // TODO: fix in some other way
  sleep(5);
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<ParticleFilter>();
  node->setup_helpers(node);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
