# carfollow/config.py
# -------------------------------------------------
PATH_ZARR                = r"C:/Users/frank/Downloads/CarFollowing/CarFollowing"

MAX_JERK                 = 10     # m/s³
DESIRED_JERK             = 2.4     # m/s³
MAX_ACCELERATION         = 3.07   # m/s²
MAX_DECELERATION         = 5      # m/s²
DESIRED_DECELERATION     = 2
DESIRED_ACCELERATION     = 2
#Conservative
CONSERVATIVE_PARAMS      = 1.6, 1.0, 2.6, 2.6 #jerk, accel, decel, sft
#Balanced
BALANCED_PARAMS          = 2.0, 1.4, 2.6, 2.2 #jerk, accel, decel, sft
#Aggresive
AGGRESIVE_PARAMS         = 2.4, 1.8, 3.0, 1.8 #jerk, accel, decel, sft

#Human Driver
HUMAN_MEANS              = 2.0, 1.4, 2.2, 2 #jerk, accel, decel, sft
HUMAN_DEVIATIONS         = 0.25, 0.3, 0, 0.11 #jerk, accel, decel, sft

MAX_DESIRED_VELOCITY     = 30.0   # m/s

REACTION_TIME            = 0.1    # s
SAFE_FOLLOW_TIME         = 1.8    # s
START_DISTANCE           = 15     # m

TTC_THRESHOLD            = 2    # s
TTC_CLIP                 = 5      # s

TET_THRESHOLD            = 2      # timesteps
DRAC_THRESHOLD           = 4      # m/s^2
DRAC_CLIP                = 8      # m/s^2
DX_THRESHOLD             = 0.01


MIN_DISTANCE             = 2.0    # m
TEST_DISTANCE            = 1      # m

MAX_CASES                = 100000 # maximum cases

PENETRATION_RATE         = 0.6    #~6/10 cars are human
TOTAL_CARS               = 7     # Total number of cars in the simulation

