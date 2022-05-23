import numpy as np
from Camo_Worm import Camo_Worm


def costfn(
    clew: list[Camo_Worm], worm_idx: int, 
    imshape: tuple, image, 
    w_internal: float=0.3, w_dist: float=1.0, w_env=2.0):


    worm = clew[worm_idx]
    # --------------------
    # Internal Score
    # Penalty for curvature. Curvature <= 1/3rd image dimension is between 0 - 1,
    # curvature score is always < 3
    c_penalty = worm.dr / (max(imshape) / 6)
    desired_size = imshape[0] * imshape[1] / len(clew)
    size_score = abs((desired_size - (2 * worm.r * worm.width))/desired_size)
    # gamma should be close to 90 degrees
    # gamma_score = abs(((np.pi/4) - worm.dgamma) / (np.pi / 4))
    # theta should be close to 0 (horizontal lines)
    theta_score = worm.theta / (np.pi / 2)


    internal_score = (0.3 * size_score + c_penalty + theta_score) / 3

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
    exam_pts = worm.intermediate_points(int(worm_length/5))
    worm_intensity = worm.colour * 255
    intensity_scores = []
    for pt in exam_pts:
        # if (pt[0] not in range(0, imshape[1])) or (pt[1] not in range(0, imshape[0])):
        #     continue
        # clamp values
        x = int(max(0, min(imshape[1]-1, pt[0])))
        y = int(max(0, min(imshape[0]-1, pt[1])))
        # get a window of intensity in region to work out mean intensity
        x0 = max(x - 1, 0)
        x1 = min(x + 1, imshape[1]-1)
        y0 = max(y - 1, 0)
        y1 = min(y + 1, imshape[0]-1)
        
        pt_intensity = image[y0:y1, x0:x1].mean()
        intensity_scores.append(pt_intensity) #  abs((pt_intensity - worm_intensity) / 255)
    # intensity score between 0  and 1
    mean_intensity = np.mean(intensity_scores)
    # print(median_intensity)
    # print(worm_intensity)
    intensity_score = 2 * abs((mean_intensity - worm_intensity) / 255)

    environment_score = intensity_score
    # print(environment_score)

    # --------------------
    # Final Score
    # print(f"Internal: {internal_score} - Distance: {distance_score * w_dist}")
    final_score = internal_score * w_internal + distance_score * w_dist + environment_score * w_env
    # print(final_score)
    return final_score


