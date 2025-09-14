SUPPORTED_MISSIONS = {
    "pour beer": [
        "pick_up_cup",
        "place_cup_on_bar",
        "pour_beer",
        "reset_before_new_mission"
    ],
    
    "pour mojito": [
        "pick_up_cup",
        "place_cup_on_bar",
        "pick_up_mojito",
        "pour_mojito",
        "put_back_mojito",
        "reset_before_new_mission"
    ],

    "testC": [
        "test1",
        "test2",
        "reset_before_new_mission"
    ],
    "wave_hello": [
        "wave",
        "so101_reset_before_new_mission"
    ],
    "placeCupOnBar": [
        "pick_up_cup",
        "place_cup_on_bar",
        "reset_before_new_mission",
    ]
}

SUPPORTED_MISSIONS_PER_ROBOT = {
    "FrankaPanda": ["pour beer", "pour mojito", "placeCupOnBar"],
    "SO101": ["wave_hello"]}

def mission_for_types(robot_types):
    """Get supported missions for given robot types"""
    missions = set()
    for rtype in robot_types:
        if rtype in SUPPORTED_MISSIONS_PER_ROBOT:
            missions.update(SUPPORTED_MISSIONS_PER_ROBOT[rtype])
    return list(missions)
