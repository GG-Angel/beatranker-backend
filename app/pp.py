import math

# We got these calculations from the source code of the BeatLeader website.
# It was originally in JavaScript, so we converted it into Python code for use in our project.

WEIGHT_COEFFICIENT = 0.965

POINT_LIST = [
    [1.0, 7.424], [0.999, 6.241], [0.9975, 5.158], [0.995, 4.01],
    [0.9925, 3.241], [0.99, 2.7], [0.9875, 2.303], [0.985, 2.007],
    [0.9825, 1.786], [0.98, 1.618], [0.9775, 1.49], [0.975, 1.392],
    [0.9725, 1.315], [0.97, 1.256], [0.965, 1.167], [0.96, 1.094],
    [0.955, 1.039], [0.95, 1.0], [0.94, 0.931], [0.93, 0.867],
    [0.92, 0.813], [0.91, 0.768], [0.9, 0.729], [0.875, 0.65],
    [0.85, 0.581], [0.825, 0.522], [0.8, 0.473], [0.75, 0.404],
    [0.7, 0.345], [0.65, 0.296], [0.6, 0.256], [0.0, 0.0],
]

WEIGHT_CURVE = [1.0, 0.965, 0.931, 0.899, 0.867, 0.837, 0.808, 0.779, 0.752, 0.726, 0.7, 0.676, 0.652, 0.629, 0.607, 
                0.586, 0.566, 0.546, 0.527, 0.508, 0.49, 0.473, 0.457, 0.441, 0.425, 0.41, 0.396, 0.382, 0.369, 0.356, 
                0.343, 0.331, 0.32, 0.309, 0.298, 0.287, 0.277, 0.268, 0.258, 0.249, 0.24, 0.232, 0.224, 0.216, 0.209, 
                0.201, 0.194, 0.187, 0.181, 0.175, 0.168, 0.163, 0.157, 0.151, 0.146, 0.141, 0.136, 0.131, 0.127, 0.122, 
                0.118, 0.114, 0.11, 0.106, 0.102, 0.099, 0.095, 0.092, 0.089, 0.086, 0.083, 0.08, 0.077, 0.074, 0.072, 
                0.069, 0.067, 0.064, 0.062, 0.06, 0.058, 0.056, 0.054, 0.052, 0.05, 0.048, 0.047, 0.045, 0.043, 0.042, 0.041, 
                0.039, 0.038, 0.036, 0.035, 0.034, 0.033, 0.032, 0.03, 0.029, 0.028, 0.027, 0.026, 0.025, 0.025, 0.024, 0.023, 
                0.022, 0.021, 0.021, 0.02, 0.019, 0.018, 0.018, 0.017, 0.017, 0.016, 0.015, 0.015, 0.014, 0.014, 0.013, 0.013, 
                0.012, 0.012, 0.012, 0.011, 0.011, 0.01, 0.01, 0.01, 0.009, 0.009, 0.009, 0.008, 0.008, 0.008, 0.008, 0.007, 0.007, 
                0.007, 0.007, 0.006, 0.006, 0.006, 0.006, 0.006, 0.005, 0.005, 0.005, 0.005, 0.005, 0.004, 0.004, 0.004, 0.004, 0.004, 
                0.004, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 
                0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0]

MOD_MULTIPLIERS = {
    "SF": 1.72, # +72%
    "FS": 1.4,  # +40%
    "GN": 1.08, # + 8%
    "NB": 0.8,  # -20%
    "NO": 0.8,  # -20%
    "SS": 0.7,  # -30%
    "NA": 0.7,  # -30%
}

modifier_list = ["SF", "FS", "GN", "NB", "NO", "SS", "NA"]

def curve(acc, points):
    """
    Calculates the performance points based on a given accuracy by interpolating
    values from the points list.

    Parameters:
        acc (float): The accuracy point to calculate the PP for as a decimal from 0-1.
        points (list of lists): List of accuracy and point pairs for interpolation.

    Returns:
        float: Interpolated performance points for the given accuracy.
    """

    i = 0
    for i in range(len(points)):
        if points[i][0] <= acc:
            break

    if i == 0:
        i = 1

    middle_dis = (acc - points[i - 1][0]) / (points[i][0] - points[i - 1][0])
    return points[i - 1][1] + middle_dis * (points[i][1] - points[i - 1][1])

def inflate(pp):
    """
    Applies a scaling factor to increase the performance points (PP) to account for 
    a more competitive score based on a predefined base value.

    Parameters:
        pp (float): The initial performance points to be inflated.

    Returns:
        float: Inflated performance points.
    """

    return (650 * (pp ** 1.3)) / (650 ** 1.3)

def calc_modified_rating(rating, ratingName, modifier_ratings, mods):
    """ Computes the modified rating for pass, tech, or acc for any mods present.

    Params:
        rating (float):           The rating value itself
        ratingName (str):         The name of the rating, which can be 'passRating', 'accRating', or 'techRating'
        modifierRatings (dict):   The dictionary of modifier ratings stored within the corresponding map; used to compute SF and FS modifiers
        mods (list):              The list of modifiers used by the player, which can be SF, FS, GN, NB, NO, SS, or NA

    Returns:
        modified_rating (float):  The resulting modified rating (or the original rating if no mods are present)
    
    """

    if len(mods) == 0:
        return rating
    
    # only keep modifiers that actually affect the ratings
    mods = list(filter(lambda m: m in modifier_list, mods))
    
    # if SF or FS is activated, set the rating to its corresponding rating in the modifers dict
    # since these modifiers change each rating independently
    for mod in ["SF", "FS"]:
        if mod in mods:
            rating = modifier_ratings[f"{mod.lower()}{ratingName[0].upper() + ratingName[1:]}"]
            break

    remaining_mods = list(filter(lambda m: m not in ["SF", "FS"], mods))

    if len(remaining_mods) == 0:
        return rating

    # the rest of the modifers are calculated by summing up their differences against the new rating
    modifiers_sum = sum((rating * MOD_MULTIPLIERS[mod]) - rating for mod in remaining_mods)

    return rating + modifiers_sum

def get_pp_from_acc(accuracy, pass_rating, acc_rating, tech_rating):
    """
    Calculates performance points earned for a specific accuracy point 
    on a BeatLeader ranked map.

    Parameters:
        accuracy (float): The player's accuracy % on the map as a decimal from 0-1.
        pass_rating (float): The pass rating of the level.
        acc_rating (float): The accuracy rating of the level.
        tech_rating (float): The technical difficulty rating of the level.

    Returns:
        list: A list containing the total inflated PP and individual contributions
              from pass, accuracy, and technical ratings after inflation.
    """

    pass_pp = 15.2 * math.exp(pass_rating ** (1 / 2.62)) - 30
    if not math.isfinite(pass_pp) or pass_pp < 0:
        pass_pp = 0

    acc_pp = curve(accuracy, POINT_LIST) * acc_rating * 34
    
    tech_pp = math.exp(1.9 * accuracy) * 1.08 * tech_rating
    total_pp = inflate(pass_pp + acc_pp + tech_pp)
    inflation = total_pp / (pass_pp + acc_pp + tech_pp)

    return [total_pp, pass_pp * inflation, acc_pp * inflation, tech_pp * inflation]

def calc_pp_from_accuracy(acc, pass_rating, acc_rating, tech_rating):
    pp_values = get_pp_from_acc(acc, pass_rating, acc_rating, tech_rating)
    
    total_pp, pass_pp, acc_pp, tech_pp = pp_values
    
    return {
        'accuracy': acc,
        'total_pp': total_pp,
        'pass_pp': pass_pp,
        'acc_pp': acc_pp,
        'tech_pp': tech_pp
    }

# accuracy_point = 0.9784
# pass_rating = 8.563406
# acc_rating = 9.8987465
# tech_rating = 3.609957

# modifiersRating = {
#     "id": 6459,
#     "ssPredictedAcc": 0.9801551,
#     "ssPassRating": 7.125044,
#     "ssAccRating": 8.900174,
#     "ssTechRating": 3.2884529,
#     "ssStars": 7.7313967,
#     "fsPredictedAcc": 0.9727765,
#     "fsPassRating": 10.450607,
#     "fsAccRating": 11.29106,
#     "fsTechRating": 3.510017,
#     "fsStars": 11.007533,
#     "sfPredictedAcc": 0.9649299,
#     "sfPassRating": 14.816195,
#     "sfAccRating": 13.037447,
#     "sfTechRating": 3.5754108,
#     "sfStars": 14.492574,
#     "bfsPredictedAcc": 0.9736861,
#     "bfsPassRating": 10.827765,
#     "bfsAccRating": 10.513508,
#     "bfsTechRating": 3.6590998,
#     "bfsStars": 10.477555,
#     "bsfPredictedAcc": 0.96795607,
#     "bsfPassRating": 14.141363,
#     "bsfAccRating": 11.907755,
#     "bsfTechRating": 3.8464062,
#     "bsfStars": 13.156608
# }

# print(compute_modified_rating(pass_rating, "PassRating", modifiersRating, ["SF", "GN", "NB", "NO"]))

# result = calculate_pp_for_accuracy(accuracy_point, pass_rating, acc_rating, tech_rating)
# print(f"Accuracy: {result['accuracy']}\nTotal PP: {result['total_pp']}\nPass PP: {result['pass_pp']}\nAcc PP: {result['acc_pp']}\nTech PP: {result['tech_pp']}")