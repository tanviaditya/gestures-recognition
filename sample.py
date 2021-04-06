# # import sqlite3

# # # Create a SQL connection to our SQLite database
# # con = sqlite3.connect("gesture_db.db")

# # cur = con.cursor()

# # # The result of a "cursor.execute" can be iterated over by row
# # for row in cur.execute('SELECT * FROM gesture;'):
# #     print(row)

# # # Be sure to close the connection
# # con.close()

from keras.models import load_model
import numpy as np
import pickle
from keras.utils import np_utils

with open("val_images", "rb") as f:
    val_images = np.array(pickle.load(f))
with open("val_labels", "rb") as f:
    val_labels = np.array(pickle.load(f), dtype=np.int32)

val_images = np.reshape(val_images, (val_images.shape[0], 50, 50, 1))
val_labels = np_utils.to_categorical(val_labels)

model = load_model('cnn_model_keras2_new3.h5')
scores = model.evaluate(val_images, val_labels, verbose=0)
print(model.summary())
# from twilio.rest import Client

# import smtplib 
# import geocoder
# import reverse_geocoder as rg
# import pprint
# g = geocoder.ip('me')
# print(g.latlng)
# coords= str(g.latlng[0])+", "+str(g.latlng[1])
# print(coords)

# # g = geocoder.google([g.latlng[0], g.latlng[1]], method='reverse')
# # # from geopy.geocoders import Nominatim
# # # geolocator = Nominatim(user_agent="sign_language_alert")
# # # location = geolocator.reverse('52.509669, 13.376294')
# # # print(location.address)
# # print(g)
# coordinates =(28.613939, 77.209023)
# result = rg.search(coordinates)
    
# # result is a list containing ordered dictionary.
# pprint.pprint(result) 