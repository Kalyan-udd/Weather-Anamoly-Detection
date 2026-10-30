from weather_anamoly.lstm import lstm_multivariate_feature_generation
from weather_anamoly.utils import ImportCoordinates, ImportData
from weather_anamoly.lstm import transform_3D_test
from tensorflow import keras
from keras_tuner import Hyperband
from tensorflow.keras.callbacks import EarlyStopping

start = "2010-01-01"
end = "2026-01-01"

coor = ImportCoordinates()
lat, long = coor.Fetch_coordinates("machilipatnam")
data = ImportData()

df = data.extract(start_date=start, end_date=end, latitude=lat, longitude=long)
df_ = lstm_multivariate_feature_generation(df=df)
features = ["z_temperature", "z_humidity", "z_pressure"]
df_train = df_
df_train.reset_index(drop=True, inplace=True)
x_train, y_train = transform_3D_test(df_train)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(24,3)))
    hp.units = hp.Choice('units', values = [16, 32, 64, 128, 256])
    hp.layer = hp.Choice('layers', values = [1, 2, 3, 4, 5])
    for i in range(hp.layer):
        model.add(keras.layers.LSTM(units=hp.units, return_sequences=True, activation='tanh'))
    model.add(keras.layers.LSTM(units=hp.units, activation='tanh'))
    model.add(keras.layers.Dense(units=3))
    hp.learning_rate = hp.Choice('learning rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.learning_rate), loss='mae', metrics=['mse'])
    return model


tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=20,
    directory = "LSTM_hyperparameter_tuning",
    project_name = "Weather Forecast"
)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5
)

tuner.search(
    x_train, y_train,
    batch_size = 64,
    epochs = 30,
    validation_split=0.2,
    callbacks=[early_stop],
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.get_best_models(num_models=1)[0]

model = tuner.hypermodel.build(hp=best_hps)

best_model.save(filepath="/artifacts/lstm_forecast_model_server_trained.keras")