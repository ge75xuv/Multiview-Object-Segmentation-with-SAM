from pathlib import Path

OR4D_TAKE_SPLIT = {'train': [1, 3, 5, 7, 9, 10], 'val': [4, 8], 'test': [2, 6]}
OR4D_TAKE_FOLDERS = ['export_holistic_take1_processed', 'export_holistic_take2_processed', 'export_holistic_take3_processed', 'export_holistic_take4_processed',
                     'export_holistic_take5_processed',
                     'export_holistic_take6_processed', 'export_holistic_take7_processed', 'export_holistic_take8_processed', 'export_holistic_take9_processed',
                     'export_holistic_take10_processed']
# thesee are the takes that are actually used
OR4D_TAKE_NAMES = ['001_4DOR', '002_4DOR', '003_4DOR', '004_4DOR', '005_4DOR', '006_4DOR', '007_4DOR', '008_4DOR', '009_4DOR', '010_4DOR']
OR4D_TAKE_NAME_TO_FOLDER = {'001_4DOR': 'export_holistic_take1_processed', '002_4DOR': 'export_holistic_take2_processed', '003_4DOR': 'export_holistic_take3_processed',
                            '004_4DOR': 'export_holistic_take4_processed', '005_4DOR': 'export_holistic_take5_processed', '006_4DOR': 'export_holistic_take6_processed',
                            '007_4DOR': 'export_holistic_take7_processed', '008_4DOR': 'export_holistic_take8_processed', '009_4DOR': 'export_holistic_take9_processed',
                            '010_4DOR': 'export_holistic_take10_processed'}

OR4D_SPLIT_TO_TAKES = {
    "train": ['001_4DOR', '003_4DOR', '005_4DOR', '007_4DOR', '009_4DOR', '010_4DOR'],
    "small_train": ['001_4DOR', '005_4DOR', '007_4DOR', '009_4DOR'],
    "mini_train": ['001_4DOR'],  # just for debugging
    "val": ['004_4DOR', '008_4DOR'],
    "test": ['002_4DOR', '006_4DOR']
}

OBJECT_COLOR_MAP = {
    'anesthesia_equipment': (0.96, 0.576, 0.65),
    'operating_table': (0.2, 0.83, 0.72),
    'instrument_table': (0.93, 0.65, 0.93),
    'secondary_table': (0.90, 0.30, 0.63),
    'instrument': (1.0, 0.811, 0.129),
    'object': (0.61, 0.48, 0.04),
    'Patient': (0, 1., 0),
    'human_0': (1., 0., 0),
    'human_1': (0.9, 0., 0),
    'human_2': (0.85, 0., 0),
    'human_3': (0.8, 0., 0),
    'human_4': (0.75, 0., 0),
    'human_5': (0.7, 0., 0),
    'human_6': (0.65, 0., 0),
    'human_7': (0.6, 0., 0)
    # 'drill': (0.90, 0.30, 0.30),
    # 'saw': (1.0, 0.811, 0.129),
    # '': (0.90, 0.30, 0.30),
    # 'c-arm': (1.0, 0.811, 0.129),
    # 'c-arm_base': (0.61, 0.48, 0.04),
    # 'unidentified_1': (0.34, 0.65, 0.36),
    # 'unidentified_2': (0.22, 0.3, 0.83)
}

OBJECT_LABEL_MAP = {
    'anesthesia_equipment': 0,
    'operating_table': 1,
    'instrument_table': 2,
    'secondary_table': 3,
    'instrument': 4,
    'object': 5,
    'Patient': 9,
    'human_0': 10,
    'human_1': 11,
    'human_2': 12,
    'human_3': 13,
    'human_4': 14,
    'human_5': 15,
    'human_6': 16,
    'human_7': 17
}

# OR_4D_DATA_ROOT_PATH = Path('../../4D-OR')
OR_4D_DATA_ROOT_PATH = Path('/home/data/4D-OR')

EXPORT_HOLISTICS_PATHS = list(OR_4D_DATA_ROOT_PATH.glob('export_holistic_take*'))

MMOR_TAKE_FOLDERS = ['001_PKA', '002_PKA', '003_TKA', '004_PKA', '005_TKA', '006_PKA', '007_TKA', '008_PKA', '009_TKA', '010_PKA', '011_TKA', '012_1_PKA', '013_PKA', '014_PKA', '015-018_PKA',
                     '019-022_PKA', '023-032_PKA', '033_PKA', '035_PKA', '036_PKA', '037_TKA', '038_TKA']
# thesee are the takes that are actually used
MMOR_TAKE_NAMES = ['001_PKA', '002_PKA', '003_TKA', '004_PKA', '005_TKA', '006_PKA', '007_TKA', '008_PKA', '009_TKA', '010_PKA', '011_TKA', '012_1_PKA', '012_2_PKA', '013_PKA', '014_PKA',
                   '015_PKA', '016_PKA', '017_PKA', '018_1_PKA', '018_2_PKA', '019_PKA', '020_PKA', '021_PKA', '022_PKA', '023_PKA', '024_PKA', '025_PKA', '026_PKA', '027_PKA', '028_PKA',
                   '029_PKA', '030_PKA', '031_PKA', '032_PKA', '033_PKA', '035_PKA', '036_PKA', '037_TKA', '038_TKA']
# TODO Take 028_PKA contains many examples of running against the robot or Operating Table, seperate them into multiple examples. Same for 029_PKA and 030_PKA
# seperate some takes even further. Rely on the take_jsons. Make sure combined takes are getting loaded correctly
MMOR_TAKE_NAME_TO_FOLDER = {'012_1_PKA': '012_PKA', '012_2_PKA': '012_PKA', '015_PKA': '015-018_PKA', '016_PKA': '015-018_PKA', '017_PKA': '015-018_PKA', '018_1_PKA': '015-018_PKA',
                            '018_2_PKA': '015-018_PKA',
                            '019_PKA': '019-022_PKA', '020_PKA': '019-022_PKA', '021_PKA': '019-022_PKA', '022_PKA': '019-022_PKA', '023_PKA': '023-032_PKA', '024_PKA': '023-032_PKA',
                            '025_PKA': '023-032_PKA', '026_PKA': '023-032_PKA',
                            '027_PKA': '023-032_PKA', '028_PKA': '023-032_PKA', '029_PKA': '023-032_PKA', '030_PKA': '023-032_PKA', '031_PKA': '023-032_PKA', '032_PKA': '023-032_PKA'}

MMOR_SPLIT_TO_TAKES = {
    "train": ['001_PKA', '003_TKA', '005_TKA', '006_PKA', '008_PKA', '010_PKA', '012_1_PKA', '035_PKA', '037_TKA'],  # deleted , '012_PKA'
    "small_train": ['001_PKA', '003_TKA', '035_PKA', '037_TKA', '005_TKA'],
    "over_train": ['001_PKA'],
    "over_train2": ['001_PKA'],
    "mini_train": ['013_PKA'],  # just for debugging
    "val": ['002_PKA', '007_TKA', '009_TKA'],
    "test": ['004_PKA', '011_TKA', '036_PKA', '038_TKA'], # 004_PKA is flickering, if that is a big issue maybe change it to val. Maybe a validation take should be here also?
    'short_clips': ['013_PKA', '014_PKA', '015_PKA', '016_PKA', '017_PKA', '018_1_PKA', '018_2_PKA', '019_PKA', '020_PKA', '021_PKA', '022_PKA', '023_PKA', '024_PKA', '025_PKA', '026_PKA', '027_PKA',
                    '028_PKA', '029_PKA', '030_PKA', '031_PKA', '032_PKA', '033_PKA']  # these are not considered main takes, but can be used for some other things including anomaly detection
}
# MMOR_DATA_ROOT_PATH = Path('../../MM-OR_processed')
MMOR_DATA_ROOT_PATH = Path('/home/polyaxon-data/data1/MM-OR_processed')

TRACKER_OBJECT_MAP = {
    '8000050': 'base_array',
    '8000056': 'calibration_array',
    '8000057': 'upper_tracker',
    '8000058': 'lower_tracker',
    '8000054': 'green_tip',
    '8000053': 'blue_tip',
    '8000999': 'calibration_array'
}

DEPTH_SCALING = 2000

LIMBS = [
    [5, 4],  # (righthip-lefthip)
    [9, 7],  # (rightwrist - rightelbow)
    [7, 3],  # (rightelbow - rightshoulder)
    [2, 6],  # (leftshoulder - leftelbow)
    [6, 8],  # (leftelbow - leftwrist)
    [5, 3],  # (righthip - rightshoulder)
    [4, 2],  # (lefthip - leftshoulder)
    [3, 1],  # (rightshoulder - neck)
    [2, 1],  # (leftshoulder - neck)
    [1, 0],  # (neck - head)
    [10, 4],  # (leftknee,lefthip),
    [11, 5],  # (rightknee,righthip),
    [12, 10],  # (leftfoot,leftknee),
    [13, 11]  # (rightfoot,rightknee),

]

HUMAN_POSE_COLOR_MAP = {
    0: (255, 0, 0),
    1: (200, 0, 0),
    2: (68, 240, 65),
    3: (50, 166, 48),
    4: (65, 201, 224),
    5: (42, 130, 145),
    6: (66, 179, 245),
    7: (44, 119, 163),
    8: (245, 173, 66),
    9: (186, 131, 50)
}

IDX_TO_BODY_PART = ['head', 'neck', 'leftshoulder', 'rightshoulder', 'lefthip', 'righthip', 'leftelbow', 'rightelbow', 'leftwrist', 'rightwrist', 'leftknee',
                    'rightknee', 'leftfoot', 'rightfoot']

STATIONARY_OBJECTS = ['instrument_table',
                      'secondary_table']  # We might have to seperate these into different takes, if an object is only stationary in one take etc.

TRACK_TO_METAINFO = {
    'instrument_table': {'color': (255, 51, 153), 'label': 1},
    'ae': {'color': (0, 0, 255), 'label': 2},
    'ot': {'color': (255, 255, 0), 'label': 3},
    'mps_station': {'color': (133, 0, 133), 'label': 4},
    'patient': {'color': (255, 0, 0), 'label': 5},
    'drape': {'color': (183, 91, 255), 'label': 6},
    'anest': {'color': (177, 255, 110), 'label': 7},
    'circulator': {'color': (255, 128, 0), 'label': 8},
    'assistant_surgeon': {'color': (116, 166, 116), 'label': 9},
    'head_surgeon': {'color': (76, 161, 245), 'label': 10},
    'mps': {'color': (125, 100, 25), 'label': 11},
    'nurse': {'color': (128, 255, 0), 'label': 12},
    'drill': {'color': (0, 255, 128), 'label': 13},  # Changed
    'hammer': {'color': (204, 0, 0), 'label': 15},
    'saw': {'color': (0, 255, 234), 'label': 16},
    'tracker': {'color': (255, 128, 128), 'label': 17},  # Changed
    'mako_robot': {'color': (60, 75, 255), 'label': 18},  # Changed
    'monitor': {'color': (255, 255, 128), 'label': 24},  # Changed
    'c_arm': {'color': (0, 204, 128), 'label': 25},  # Changed
    'unrelated_person': {'color': (255, 255, 255), 'label': 26},
    'student': {'color': (162, 232, 108), 'label': 27},
    'secondary_table': {'color': (153, 0, 153), 'label': 28},
    'cementer': {'color': (153, 76, 0), 'label': 29},
    '__background__': {'color': (0, 0, 0), 'label': 0}
}

LABEL_PROJECTION_MAP = {
    # DEBUG FOR OVERTRAINING
    # 10: {'color': (76, 161, 245), 'label': 0},
    # 13: {'color': (0, 255, 128), 'label': 1},
    # 15: {'color': (204, 0, 0), 'label': 2},

    # TRUE PROJECTION
    # 1: {'color': (255, 51, 153), 'label': 0},
    # 2: {'color': (0, 0, 255), 'label': 1},
    # 3: {'color': (255, 255, 0), 'label': 2},
    # 4: {'color': (133, 0, 133), 'label': 3},
    # 5: {'color': (255, 0, 0), 'label': 4},
    # 6: {'color': (183, 91, 255), 'label': 5},
    # 7: {'color': (177, 255, 110), 'label': 6},
    # 8: {'color': (255, 128, 0), 'label': 7},
    # 9: {'color': (116, 166, 116), 'label': 8},
    # 10: {'color': (76, 161, 245), 'label': 9},
    # 11: {'color': (125, 100, 25), 'label': 10},
    # 12: {'color': (128, 255, 0), 'label': 11},
    # 13: {'color': (0, 255, 128), 'label': 12},
    # 15: {'color': (204, 0, 0), 'label': 13},
    # 16: {'color': (0, 255, 234), 'label': 14},
    # 17: {'color': (255, 128, 128), 'label': 15},
    # 18: {'color': (60, 75, 255), 'label': 16},
    # 24: {'color': (255, 255, 128), 'label': 17},
    # 25: {'color': (0, 204, 128), 'label': 18},
    # 26: {'color': (255, 255, 255), 'label': 19},
    # 27: {'color': (162, 232, 108), 'label': 20},
    # 28: {'color': (153, 0, 153), 'label': 21},
    # 29: {'color': (153, 76, 0), 'label': 22},

    # PROJECTION WITH HUMAN LABEL
    1: {'color': (255, 51, 153), 'label': 0},    # 'instrument_table'
    2: {'color': (0, 0, 255), 'label': 1},       # 'ae'
    3: {'color': (255, 255, 0), 'label': 2},     # 'ot'
    4: {'color': (133, 0, 133), 'label': 3},     # 'mps_station'
    5: {'color': (255, 0, 0), 'label': 4},       # 'patient'
    6: {'color': (183, 91, 255), 'label': 5},    # 'drape'
    7: {'color': (177, 255, 110), 'label': 6},   # 'anest'
    8: {'color': (255, 128, 0), 'label': 6},     # 'circulator'
    9: {'color': (116, 166, 116), 'label': 6},   # 'assistant_surgeon'
    10: {'color': (76, 161, 245), 'label': 6},   # 'head_surgeon'
    11: {'color': (125, 100, 25), 'label': 6},   # 'mps'
    12: {'color': (128, 255, 0), 'label': 6},    # 'nurse'
    13: {'color': (0, 255, 128), 'label': 7},    # 'drill'
    15: {'color': (204, 0, 0), 'label': 8},      # 'hammer'
    16: {'color': (0, 255, 234), 'label': 9},    # 'saw'
    17: {'color': (255, 128, 128), 'label': 10}, # 'tracker'
    18: {'color': (60, 75, 255), 'label': 11},   # 'mako_robot'
    24: {'color': (255, 255, 128), 'label': 12}, # 'monitor'
    25: {'color': (0, 204, 128), 'label': 13},   # 'c_arm'
    26: {'color': (255, 255, 255), 'label': 6},  # 'unrelated_person'
    27: {'color': (162, 232, 108), 'label': 6},  # 'student'
    28: {'color': (153, 0, 153), 'label': 14},   # 'secondary_table'
    29: {'color': (153, 76, 0), 'label': 15},    # 'cementer'
}