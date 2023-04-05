from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import pickle
model = load_model('model/model0.h5')
pickle.dump(model, open('model/model0.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

