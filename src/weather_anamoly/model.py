
# TODO: confirm exact order from encoder.classes_ in the notebook
LABEL_DECODER = {
    0: "genuine",
    1: "spike",
    2: "frozen",
    3: "fault",
}

# TODO: ENCODING THE LABELS 
LABEL_ENCODER = {
    "genuine": 0,
    "spike": 1,
    "frozen": 2,
    "fault": 3,
}

# TODO: must match X_train.columns order EXACTLY
FEATURE_COLUMNS = [
    'temperature', 'humidity', 'pressure', 'cos_hour','sin_hour', 
    'cos_month', 'sin_month', 'temp_gradient', 'humid_gradient',
    'press_gradient', '6hr_gradient_temp', '6hr_gradient_press',
    '6hr_gradient_humid', 'elevation', 'latitude', 'longitude'
]