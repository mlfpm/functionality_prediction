# @author semese

import numpy as np

# demographic columns
demogr_cols = [
    'Lives_Father_-1.0', 'Lives_Father_0.0', 'Lives_Father_1.0',
    'Lives_Mother_-1.0', 'Lives_Mother_0.0', 'Lives_Mother_1.0',
    'Lives_Child_-1.0', 'Lives_Child_0.0', 'Lives_Child_1.0',
    'Lives_Siblings_-1.0', 'Lives_Siblings_0.0', 'Lives_Siblings_1.0',
    'Lives_Family_-1.0', 'Lives_Family_0.0', 'Lives_Family_1.0',
    'Lives_Couple_-1.0', 'Lives_Couple_0.0', 'Lives_Couple_1.0',
    'Lives_Friends_-1.0', 'Lives_Friends_0.0', 'Lives_Friends_1.0',
    'Lives_Alone_-1.0', 'Lives_Alone_0.0', 'Lives_Alone_1.0',
    'Lives_Sharing_-1.0', 'Lives_Sharing_0.0', 'Lives_Sharing_1.0',
    'Lives_Homeless_-1.0', 'Lives_Homeless_0.0', 'Lives_Homeless_1.0',
    'SexUser_1.0', 'SexUser_2.0', 'SexUser_4.0', 'SexUser_5.0',
    'FamStatus_-1.0', 'FamStatus_1.0', 'FamStatus_4.0', 'FamStatus_5.0', 'FamStatus_6.0',
    'CurrentActivity_-1.0', 'CurrentActivity_1.0', 'CurrentActivity_2.0',
    'CurrentActivity_5.0', 'CurrentActivity_6.0', 'CurrentActivity_7.0',
    'CurrentActivity_8.0', 'age_by_decade_NA', 'age_by_decade_20-',
    'age_by_decade_20s', 'age_by_decade_30s', 'age_by_decade_40s',
    'age_by_decade_50s', 'age_by_decade_60s', 'age_by_decade_70+'
]


# whodas columns
mobility_questions = ['WHODAS2007', 'WHODAS2008',
                      'WHODAS2009', 'WHODAS2010', 'WHODAS2011']
transformed_scored = ['WHODAS_mobility', 'WHODAS_mobility_cat',
                      'WHODAS_mobility_3cls', 'WHODAS_mobility_bin']

# 48-slot columns
slots_cols_dict = {
    col: [col + str(s) for s in range(48)] for col in ['steps_', 'dist_', 'at_home_', 'activity_']
}
slot_cols = np.asarray(list(slots_cols_dict.values())).ravel()

# daily summary columns
daily_sum_cols = ['location_distance', 'location_clusters_count', 'location_time_home', 'location_time_work',
                  'steps_steps_total', 'sleep_duration', 'app_usage_total',
                  'emotions_emotions_count', 'exercise_time']
