import numpy as np
from Camo_Worm import Camo_Worm


def costfn(
    clew: list[Camo_Worm], worm_idx: int, 
    imshape: tuple, image, 
    w_internal: float=0.3, w_dist: float=1.0, w_env=2.0):


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

    theta_score = worm.theta / (np.pi/2)
    dr_score = 2 * abs(worm.dr / (length))
    internal_score = (length_score + theta_score + dr_score)/3

    # --------------------
    # Group Score
    distance_score = 0
    
    for i in range(0, len(clew)):
        if i is not worm_idx:
            point1 = np.array(clew[i].centre_point())
            point2 = np.array(worm.centre_point())

            eu_distance = np.linalg.norm(point1 - point2)

            distance_score += min(1 / eu_distance, 1)
    distance_score = distance_score / len(clew)
    # --------------------
    # Environment Score

    # Check intensity of pixels at control points of the worm
    worm_length = 2*worm.r
    filter_width = int(worm.width)
    exam_pts = worm.intermediate_points(max(2,int(worm_length/5)))
    worm_intensity = worm.colour * 255
    intensity_scores = []
    for pt in exam_pts:
        # if (pt[0] not in range(0, imshape[1])) or (pt[1] not in range(0, imshape[0])):
        #     continue
        # clamp values
        x = int(max(0, min(imshape[1]-1, pt[0])))
        y = int(max(0, min(imshape[0]-1, pt[1])))
        # get a window of intensity in region to work out mean intensity
        x0 = max(x - filter_width, 0)
        x1 = min(x + filter_width, imshape[1]-1)
        y0 = max(y - filter_width, 0)
        y1 = min(y + filter_width, imshape[0]-1)
        
        pt_intensity = abs((image[y0:y1, x0:x1]-worm_intensity).mean())
        # print(pt_intensity)
        intensity_scores.append(pt_intensity) #  abs((pt_intensity - worm_intensity) / 255)
    # intensity score between 0  and 1
    mean_intensity = np.mean(intensity_scores)
    # print(median_intensity)
    # print(worm_intensity)
    intensity_score = 3 * ((mean_intensity) / 255)

    environment_score = intensity_score
    # print(environment_score)

    # --------------------
    # Final Score
    # print(f"Internal: {internal_score} - Distance: {distance_score * w_dist}")
    final_score = internal_score * w_internal + distance_score * w_dist + environment_score * w_env
    # print(final_score)
    return final_score


