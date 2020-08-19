"""Plot and optimise flight tracks for satellite validation"""

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as spg
import shapely.ops as sops

def end_point(current_point, angle, distance):
    """Calculate end-pont of line"""
    angle_rad = np.radians(angle)
    return (current_point[0] + distance * np.sin(angle_rad),
            current_point[1] + distance * np.cos(angle_rad))

def arc_centre_endpoint(pos, angle, radius, angle_of_turn):
    """Calculate centre and end-point of arc given start-point and direction"""
    # Arc centre is perpendicular to current path
    clockwise = (angle_of_turn > 0.0)
    if clockwise:
        centre_direction = angle + 90.0
    else:
        centre_direction = angle - 90.0

    centre = end_point(pos, centre_direction, radius)
    endpoint = end_point(centre, centre_direction + 180 + angle_of_turn, radius)
    return (centre, endpoint)

def total_distance(leg_distances, turn_angles, radius_of_turn):
    return np.sum(leg_distances) + np.sum(np.abs(turn_angles)) / 360.0 * 2 * np.pi * radius_of_turn

def sampled_area_fraction(leg_distances,
                          turn_angles,
                          x0 = 0.0,
                          y0 = 0.0,
                          radius_of_turn=6.0,
                          satellite_footprint_radius=7.5,
                          sensor_footprint_radius=1.0,
                          true_airspeed = 140.0,
                          plot_figure=True):

    # Add footprint
    footprint = spg.Point(0,0)
    footprint = footprint.buffer(satellite_footprint_radius)

    if plot_figure:
        from matplotlib.patches import Arc
        plt.figure()
        ax = plt.axes()
        plt.plot(*footprint.exterior.xy, color="red", linestyle="--")

    pointing_angle = 0.0
    pos = (x0,y0)
    tracks = []
    for i_leg in range(len(leg_distances)):
        # Straight line segment
        new_pos = end_point(pos, pointing_angle, leg_distances[i_leg])
        if plot_figure:
            plt.plot([pos[0],new_pos[0]],[pos[1],new_pos[1]],color="black")
        line = spg.LineString([pos, new_pos])
        tracks.append(line)
        pos = new_pos
        if i_leg == (len(leg_distances) - 1):
            break
        # Turn
        centre,endpoint = arc_centre_endpoint(pos, pointing_angle, radius_of_turn, turn_angles[i_leg])
        if plot_figure:
            if turn_angles[i_leg] < 0:
                t1 = 360.0-pointing_angle
                t2 = t1 + np.abs(turn_angles[i_leg])
                print(pointing_angle, t1, t2)
            else:
                t1 = 180 - (turn_angles[i_leg]) - pointing_angle
                t2 = t1 + (turn_angles[i_leg])
            a = Arc(centre,
                    2*radius_of_turn,
                    2*radius_of_turn,
                    angle = 0.0,
                    theta1 = t1,
                    theta2 = t2)
            ax.add_artist(a)

        pointing_angle += turn_angles[i_leg]
        pos = endpoint

    distance_in_footprint = 0.0

    sampled_areas = []
    for l in tracks:
        distance_in_footprint += l.intersection(footprint).length
        lbuff = l.buffer(sensor_footprint_radius)
        if plot_figure:
            plt.plot(*lbuff.exterior.xy, color="blue", linestyle="--")
        sampled_areas.append(lbuff.intersection(footprint))
    sampled_area = sops.unary_union(sampled_areas)
    saf = sampled_area.area / footprint.area

    if plot_figure:
        if isinstance(sampled_area, spg.MultiPolygon):
            for poly in sampled_area:
                plt.plot(*poly.exterior.xy,color="green")
        else:
            plt.plot(*sampled_area.exterior.xy,color="green")

        # Set limits
        plt.xlim(-40,40)
        plt.ylim(-40,40)
        ax.set_aspect("equal","box")
    tdist = total_distance(leg_distances, turn_angles, radius_of_turn)
    total_time = tdist * 1000.0/true_airspeed/60.0
    if plot_figure:
        print(f"sampled distance (km): {distance_in_footprint:.2f}, total time (min): {total_time:.1f}, "
              f"sampled area (km^2): {sampled_area.area}, sampled area fraction: {saf}")
    return saf

def objective_function(x,
                       num_leg,
                       radius_of_turn,
                       satellite_footprint_radius,
                       sensor_footprint_radius,
                       true_airspeed):
    return -1.0* sampled_area_fraction(x[0:num_leg], x[num_leg:(2*num_leg-1)], x[-2], x[-1],
                                       radius_of_turn,
                                       satellite_footprint_radius,
                                       sensor_footprint_radius,
                                       true_airspeed,
                                       False)

def optimum_distance_angle(num_leg,
                           radius_of_turn=6.0,
                           satellite_footprint_radius=7.5,
                           sensor_footprint_radius=0.5,
                           true_airspeed = 140.0,
                           max_time = 20.0):
    from scipy.optimize import minimize, NonlinearConstraint, Bounds
    max_dist = max_time * 60.0 * true_airspeed / 1000.0

    # x vector contains num_leg distances, followed by num_leg-1 angles followed by x0 and y0
    dist_constraint = NonlinearConstraint(lambda x: total_distance(x[0:num_leg], x[num_leg:(2*num_leg-1)],
                                                                   radius_of_turn),
                                          0.1, max_dist)
    lower_bounds = np.concatenate((np.ones(num_leg), -359*np.ones(num_leg-1), [-50.0, -50.0]))
    upper_bounds = np.concatenate((np.inf*np.ones(num_leg), 359*np.ones(num_leg-1), [50.0, 50.0]))

    bounds_constraint = Bounds(lower_bounds, upper_bounds)
    x0 = np.concatenate((15.0 * np.ones(num_leg), 225.0*np.ones(num_leg-1), [0.0, -7.5]))
    res = minimize(objective_function, x0, method="trust-constr",
                   args = (num_leg,radius_of_turn,satellite_footprint_radius, sensor_footprint_radius, true_airspeed),
                   constraints=[dist_constraint],
                   bounds=bounds_constraint,
                   options={"verbose":1})
    print(res.x)
    sampled_area_fraction(res.x[0:num_leg], res.x[num_leg:(2*num_leg-1)], res.x[-2], res.x[-1],
                          radius_of_turn, satellite_footprint_radius, sensor_footprint_radius, true_airspeed,
                          True)
    plt.ylim(-20,20)
    plt.xlim(-20,20)

if __name__ == "__main__":
    plt.ioff()
    optimum_distance_angle(5, max_time=20.0, sensor_footprint_radius=0.5)
    #sampled_area_fraction([10,5,12,3],[100,-180,-120])
    plt.draw()
    plt.show()
