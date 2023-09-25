#ifndef OCCUPANCYFIELD_HPP
#define OCCUPANCYFIELD_HPP

#include <nav_msgs/msg/occupancy_grid.hpp>

#include "knncpp.h"
#include "rclcpp/rclcpp.hpp"

typedef Eigen::MatrixXd Matrix;

/**
 * Stores an occupancy field for an input map.  An occupancy field returns
 * the distance to the closest obstacle for any coordinate in the map
 * Attributes:
 *      map: the map to localize against (nav_msgs/OccupancyGrid)
 *      closest_occ: the distance for each entry in the OccupancyGrid to
 *      the closest obstacle
 */
class OccupancyField {
public:
  nav_msgs::msg::OccupancyGrid map;
  Matrix closest_occ;
  Matrix occupied_coordinates;

  OccupancyField(std::shared_ptr<rclcpp::Node> node);

  /**
   * Returns: the upper and lower bounds of x and y such that the resultant
   * bounding box contains all of the obstacles in the map.  The format of
   * the return value is ((x_lower, x_upper), (y_lower, y_upper))
   */
  std::array<double, 4> get_obstacle_bounding_box();

  /**
   * Compute the closest obstacle to the specified (x,y) coordinate in
   * the map.  If the (x,y) coordinate is out of the map boundaries, nan
   * will be returned.
   */
  double get_closest_obstacle_distance(float x, float y);
};
#endif