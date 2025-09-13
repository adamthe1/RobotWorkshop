SUPPORTED_MISSIONS = {
    "pour beer": [
        "place_cup_on_bar",
        "pour_beer_bottle",
        "reset_before_new_mission"
    ],
    
    "pour mojito": [
        "place_cup_on_bar",
        "pour_mojito_bottle",
        "reset_before_new_mission"
    ],

    "placeCupOnBar": [
        "place_cup_on_bar",
        "reset_before_new_mission"
    ],

    "testC": [
        "test1",
        "test2",
        "reset_before_new_mission"
    ],
    "wave_hello": [
        "wave",
        "reset_before_new_mission"
    ]
}

SUPPORTED_MISSIONS_PER_ROBOT = {
    "FrankaPanda": ["placeCupOnBar", "testC", "pour beer", "pour mojito"],
    "SO101": ["wave_hello"]}

def mission_for_types(robot_types):
    """Get supported missions for given robot types"""
    missions = set()
    for rtype in robot_types:
        if rtype in SUPPORTED_MISSIONS_PER_ROBOT:
            missions.update(SUPPORTED_MISSIONS_PER_ROBOT[rtype])
    return list(missions)
