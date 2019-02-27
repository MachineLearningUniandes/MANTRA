BASE = ['Class']
LOW = BASE + [
    'amplitude', 'beyond1st',
    'flux_percentile_ratio_mid20', 'flux_percentile_ratio_mid35',
    'flux_percentile_ratio_mid50', 'flux_percentile_ratio_mid65',
    'flux_percentile_ratio_mid80', 'kurtosis', 'max_slope',
    'median_absolute_deviation', 'median_buffer_range_percentage',
    'pair_slope_trend', 'pair_slope_trend_last_30', 'percent_amplitude',
    'percent_difference_flux_percentile', 'skew', 'small_kurtosis', 'std',
    'stetson_j', 'stetson_k'
]

MID = LOW + ['poly1_t1', 'poly2_t2', 'poly2_t1',
             'poly3_t3', 'poly3_t2', 'poly3_t1']

HIGH = MID + ['poly4_t4', 'poly4_t3', 'poly4_t2', 'poly4_t1']

ALL_NO_CLASS = HIGH[1:]
