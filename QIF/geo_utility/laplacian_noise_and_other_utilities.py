"""
set of functions to transform a position (lat, long) into its laplacian noisy version
"""
import random
import math
from scipy import special as scipy_special
import sys

#   earth radius in meters
earthRadius = 6378137


#   return addVectorToPos which returns a noisyPosition (noisy_lat, noisy_long), r the radius, z the radius seed,
#   angleSeed the angle seed and theta the angle
def addPolarNoise(epsilon, pos, str_ctrl):
    #   random number in [0, 2*PI)
    angleSeed = random.random()
    theta = angleSeed * math.pi * 2

    #   random variable in [0,1), seed for the radius
    z = random.random()
    r = inverseCumulativeGamma(epsilon=epsilon, z=z, str_ctrl=str_ctrl)

    noisyPosition = addVectorToPos(pos, r, theta)

    return [noisyPosition, r, z, angleSeed, theta]


#   same function as before, this one returns the noisy position and the radius given epsilon, a starting position and
#   the two seeds for the angle a and the radius b
def modified_addPolarNoise_return_radius(epsilon, pos, a, b, str_ctrl):
    #   random number in [0, 2*PI)
    angleSeed = a
    theta = angleSeed * math.pi * 2
    #   random variable in [0,1)
    distanceSeed = b
    r = inverseCumulativeGamma(epsilon, distanceSeed, str_ctrl=str_ctrl)
    noisyPosition = addVectorToPos(pos, r, theta)
    return [noisyPosition, r]


#   returns a noisy position given an initial position a distance and an angle
def addVectorToPos(pos, distance, angle):
    #   force a float division: a division between integer is forced to be integer
    ang_distance = distance / float(earthRadius)

    # print ang_distance

    lat1 = rad_of_deg(pos.latitude)
    lon1 = rad_of_deg(pos.longitude)

    lat2 = math.asin(math.sin(lat1) * math.cos(ang_distance) +
                     math.cos(lat1) * math.sin(ang_distance) * math.cos(angle))
    lon2 = lon1 + math.atan2(math.sin(angle) * math.sin(ang_distance) * math.cos(lat1),
                             math.cos(ang_distance) - math.sin(lat1) * math.sin(lat2))

    #    normalise to - 180.. + 180
    lon2 = (lon2 + 3 * math.pi) % (2 * math.pi) - math.pi
    latitude = deg_of_rad(lat2)
    longitude = deg_of_rad(lon2)
    return [latitude, longitude]


#   inverseCumulativeGamma function; str_ctrl decides which LambertW functions should be used
def inverseCumulativeGamma(epsilon, z, str_ctrl):
    x = (z - 1) / math.e;
    if str_ctrl == "appr":
        return -(LambertW(x=x, bool_approx=True) + 1) / epsilon
    elif str_ctrl == "no_appr":
        return -(LambertW(x=x, bool_approx=False) + 1) / epsilon
    elif str_ctrl == "lambertw_scipy":
        #   Possible issues
        #   The evaluation can become inaccurate very close to the branch point at -1/e. In some corner cases,
        #   lambertw might currently fail to converge, or can end up on the wrong branch. Problem when the seed
        #   is 0 becasue the value of x is -1/e.
        return -(LambertW_return_scipy_function(x=x) + 1) / epsilon
    else:
        sys.exit("Error: improper value for LambertW function selection")


#   implementation of LambertW function rounding the result
def LambertW(x, bool_approx):
    #   min_diff decides when the while loop should stop
    min_diff = 1e-10;
    if x == -1 / math.e:
        return -1

    elif x < 0 and x > -1 / math.e:
        q = math.log(-x);
        p = 1;
        while (math.fabs(p - q) > min_diff):
            p = (q * q + x / math.exp(q)) / (q + 1)
            q = (p * p + x / math.exp(p)) / (p + 1)

        #   This line decides the precision of the float number that would be returned
        if bool_approx is True:
            return round(1000000 * q) / 1000000
        else:
            return q

    elif x == 0:
        return 0
    #   TODO why do you need this if branch?

    else:
        return 0


#   call scipy LambertW function with branch=-1 since x=(z-1)/e in [-1/e, 0)
def LambertW_return_scipy_function(x):
    return scipy_special.lambertw(z=x, k=-1)


#   conversion deg to rad
def rad_of_deg(ang):
    return ang * math.pi / 180


#   conversion rad to deg
def deg_of_rad(ang):
    return ang * 180 / math.pi


#   transform a position latitude and longitude in the corrisponding x and y cartesian coordinates
def getCartesian(pos):
    x = earthRadius * rad_of_deg(pos.longitude)
    y = earthRadius * math.log(math.tan(math.pi / 4 + rad_of_deg(pos.latitude) / 2))
    return [x, y]
