import math
import numpy as np
from Camo_Worm import Camo_Worm


def boundx(x, imshape):
    # print(x)
    # print(imshape)
    return int(max(0, min(imshape[1]-1, x)))

def boundy(y, imshape):
    return int(max(0, min(imshape[0]-1, y)))

def costfn(
    clew: list[Camo_Worm], worm_idx: int, 
    imshape: tuple, image, smoothed_image, smoothed_shape: tuple, 
    w_internal: float=0.8, w_dist: float=1.5, w_env=3.0):


    worm = clew[worm_idx]
    # --------------------
    # Internal Score
    # --------------------
    # essentially, we divided by the maximum possible value allowed
    # i.e. length cannot be greater than the image size
    # i.e. theta cannot be greater than 90 degrees
    # i.e. dr which is the radius of deviation cannot be greater than the length of the worm itself
    # and we double that because we really don't like it (not sure if that's a good idea)
    length = 2 * worm.r
    length_score = length / max(imshape) 

    theta_score = worm.theta 
    dr_score = 2 * abs(worm.dr / (length))
    internal_score = (length_score + 2*theta_score + dr_score)/3

    # --------------------
    # Group Score
    distance_score = 0
    
    for i in range(0, len(clew)):
        if i is not worm_idx:
            point1 = np.array(clew[i].centre_point())
            point2 = np.array(worm.centre_point())

            eu_distance = np.linalg.norm(point1 - point2)

            distance_score += min(1 / eu_distance, 1)
    distance_score = distance_score 
    # --------------------
    # Environment Score

    # Check intensity of pixels at control points of the worm
    worm_length = 2*worm.r
    filter_width = max(int(worm.width), 1)
    exam_pts = worm.intermediate_points(max(2,int(worm_length/5)))
    worm_intensity = worm.colour * 255
    intensity_scores = []
    for pt in exam_pts:
        # if (pt[0] not in range(0, imshape[1])) or (pt[1] not in range(0, imshape[0])):
        #     continue
        # clamp values
        x = boundx(pt[0], imshape) 
        y = boundx(pt[1], imshape) 
        # get a window of intensity in region to work out mean intensity
        x0 = max(min(x - filter_width, imshape[1]-2), 0)
        x1 = min(max(x + filter_width, 1), imshape[1]-1)
        y0 = max(min(y - filter_width, imshape[0]-2), 0)
        y1 = min(max(y + filter_width, 1), imshape[0]-1)
        
        pt_intensity = abs((image[y0:y1, x0:x1]-worm_intensity).mean())
        # if math.isnan(pt_intensity):
        #     from IPython import embed 
        #     embed()
        # print(pt_intensity)
        intensity_scores.append(pt_intensity) #  abs((pt_intensity - worm_intensity) / 255)
    # intensity score between 0  and 1
    mean_intensity = np.mean(intensity_scores)
    # print(median_intensity)
    # print(worm_intensity)
    intensity_score =  ((mean_intensity) / 255)

    # check against smoothed image
    p0x = boundx(worm.p0[0], smoothed_shape)
    p0y = boundy(worm.p0[1], smoothed_shape)

    p1x = boundx(worm.p1[0], smoothed_shape)
    p1y = boundy(worm.p1[1], smoothed_shape)

    p2x = boundx(worm.p2[0], smoothed_shape)
    p2y = boundy(worm.p2[1], smoothed_shape)

    # from IPython import embed 
    # embed()

    val0 = smoothed_image[p0y][p0x]
    val1 = smoothed_image[p1y][p1x]
    val2 = smoothed_image[p2y][p2x]

    segment_score = 0
    if val0 != val1:
        segment_score += 0.3
    if val0 != val2:
        segment_score += 0.3
    if val1 != val2:
        segment_score += 0.3

    environment_score = 0.7 * intensity_score + 0.3 * segment_score
    # environment_score = intensity_score 
    # print(environment_score)

    # --------------------
    # Final Score
    # print(f"Internal: {internal_score} - Distance: {distance_score * w_dist}")
    final_score = internal_score * w_internal + distance_score * w_dist + environment_score * w_env
    # print(final_score)
    return final_score, internal_score * w_internal, distance_score * w_dist, environment_score * w_env


