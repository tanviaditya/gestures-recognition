Step 1)
Creating the image histogram:
	Run set_hand_histogram.py.
	A window will open displaying the webcam.
	Position your hand so that it appears over the green squares and press 'C'.
	Another window will appear displaying the threshhold.
	Make sure that the lighting around your enviroment is sufficient enough for your hand to sharply appear on the histogram window.
	When you are satisfied with the quality of the histogram, press 'S'.


Step 2)
Creating the dataset:
	Run create_gestures.py.
	Enter the gesture id(always start from 0 if you are creating a new dataset from scratch).
	Enter the gesture name.
	A window will open displaying the webcam. It will contain a green square on it.
	Also displayed will be the threshhold for the captured screen.
	Position your gesture so that it appears inside the green box.
	When you are satisfied with the quality of the gesture outline in the threshhold, press 'C'.
	The images will start capturing one by one. The count is displayed on the screen. Once all the images have been captured, the window will close automatically.
	Repeat the above process for another gesture.

Step 3)
Augmenting the dataset.
	Run Rotate_images.py.
	The dataset size will be doubled by taking the mirror images of each existing image.

Step 4)
Creating the testing, training and validation sets.
	Run load_images.py.
	Several binary files for the testing, training and validation sets will be created.

Step 5)
Creating and storing the model.
	Run cnn_model_train.py.
	The CNN will be created and trained using the dataset created above.
	The trained CNN will be stored on disk.

Step 6)
Running the application.
	Run final.py.
	A window will appear opening the webcamera. Place your gesture inside the green box in the window.
	The gesture will be detected and its interpretation will appear at the side.



