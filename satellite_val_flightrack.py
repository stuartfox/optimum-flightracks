"""Plot and optimise flight tracks for satellite validation"""

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as spg
import shapely.ops as sops

def circle_line_intersect(p0, p1, r):
    """Find the intersection points between a circle centred on origin and a line connecting two points"""
    # Find "k" value from roots of polynomial
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    a = dx**2 + dy**2
    b = 2*(p0[0]*dx + p0[1]*dy)
    c = (p0[0]**2 + p0[1]**2 - r**2)
    k = np.roots([a,b,c])

    if (np.imag(k[0]) != 0) or k[0] == k[1]:
        # Either no intersection or line is a tangent
        return None

    x = k*dx + p0[0]
    y = k*dy + p0[1]
    return ((x[0], y[0]), (x[1], y[1]), k)

def end_point(current_point, angle, distance):
    """Calculate end-pont of line"""
    angle_rad = np.radians(angle)
    return (current_point[0] + distance * np.sin(angle_rad),
            current_point[1] + distance * np.cos(angle_rad))

def arc_centre_endpoint(pos, angle, radius, angle_of_turn, clockwise=True):
    """Calculate centre and end-point of arc given start-point and direction"""
    # Arc centre is perpendicular to current path
    if clockwise:
        centre_direction = angle + 90.0
    else:
        centre_direction = angle - 90.0

    centre = end_point(pos, centre_direction, radius)
    endpoint = end_point(centre, centre_direction + 180 + angle_of_turn, radius)
    return (centre, endpoint)

def total_distance(distance_straight, angle_of_turn, radius_of_turn, num_leg):
    return num_leg * distance_straight + (num_leg - 1) * 2 * np.pi * radius_of_turn * angle_of_turn/360.0

def sampled_area_fraction(distance_straight,
                          angle_of_turn,
                          x0 = 0.0,
                          y0 = 0.0,
                          num_leg = 3,
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
    for leg in range(num_leg):
        # Straight line segment
        new_pos = end_point(pos, pointing_angle, distance_straight)
        if plot_figure:
            plt.plot([pos[0],new_pos[0]],[pos[1],new_pos[1]],color="black")
        line = spg.LineString([pos, new_pos])
        tracks.append(line)
        pos = new_pos
        if leg == (num_leg-1):
            break
        # Turn
        centre,endpoint = arc_centre_endpoint(pos, pointing_angle, radius_of_turn, angle_of_turn)
        if plot_figure:
            t1 = 180 - angle_of_turn - pointing_angle
            t2 = t1 + angle_of_turn
            a = Arc(centre,
                    2*radius_of_turn,
                    2*radius_of_turn,
                    angle = 0.0,
                    theta1 = t1,
                    theta2 = t2)
            ax.add_artist(a)

        pointing_angle += angle_of_turn
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
    tdist = total_distance(distance_straight, angle_of_turn, radius_of_turn, num_leg)
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
    return -1.0* sampled_area_fraction(x[0], x[1], x[2], x[3],
                                       num_leg,
                                       radius_of_turn,
                                       satellite_footprint_radius,
                                       sensor_footprint_radius,
                                       true_airspeed,
                                       False)

def optimum_distance_angle(num_leg,
                           radius_of_turn=6.0,
                           satellite_footprint_radius=7.5,
                           sensor_footprint_radius=1.0,
                           true_airspeed = 140.0,
                           max_time = 30.0):
    from scipy.optimize import minimize, LinearConstraint, Bounds
    max_dist = max_time * 60.0 * true_airspeed / 1000.0
    dist_constraint = LinearConstraint([[num_leg, (num_leg-1)*2*np.pi/360*radius_of_turn, 0.0, 0.0]],[0.0],[max_dist])
    bounds_constraint = Bounds([1.0, 89.0, -50.0, -50.0],[np.inf, 359.0, 50.0, 50.0])
    x0 = np.array([15.0, 225.0, 0.0, -7.5])
    res = minimize(objective_function, x0, method="trust-constr",
                   args = (num_leg,radius_of_turn,satellite_footprint_radius, sensor_footprint_radius, true_airspeed),
                   constraints=[dist_constraint],
                   bounds=bounds_constraint,
                   options={"verbose":1})
    print(res.x)
    sampled_area_fraction(res.x[0], res.x[1], res.x[2], res.x[3],
                          num_leg, radius_of_turn, satellite_footprint_radius, sensor_footprint_radius, true_airspeed,
                          True)
    plt.ylim(-20,20)
    plt.xlim(-20,20)

if __name__ == "__main__":
    plt.ioff()
    optimum_distance_angle(5, max_time=20.0, sensor_footprint_radius=0.5)
    plt.draw()
    plt.savefig("example.png")
    plt.show()
