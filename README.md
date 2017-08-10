# Research Experience for Undergraduates Summer 2017
*Resources and code from my Summer 2017 REU with [Dr. Anthony Maida](http://www.cacs.louisiana.edu/~maida/)*

My reseach was focused on creating a Deep Learning neural network that can predict object movement from video input.
The model created uses a LSTM model that utilizes convoloution to process images, called a convLSTM.
The purpose of this research was to recreate a model similar to that of [PredNet](https://coxlab.github.io/prednet/).
It uses an unsupervised convLSTM model that trains off a variety of video data, but is much to complicated for this summer.
I helped create a simpler convLSTM model that can detect 2D movement of a black sphere on a white background.
This model is basic, but can be expanded by the addition of more featuremaps and slight code reshaping.

## My Time Spent
During this summer REU, I worked alongside Dr. Maida and his lab group. We had weekly lab presentations where we would showcase our progress. I started off with reading the papers linked below and Google's Deep Learning course on [Udacity](https://classroom.udacity.com/courses/ud730). The convLSTM model created was primarily dictated by Dr. Maida, but went through individual updates as the research group progressed.

## My Contribution
- Created a [Blender](https://www.blender.org) python script to automate video creation.
- Created a script to load the created animations into ndarrays of size [frame_count, 256, 256].
- Created a method to load the each animation ndarray into a .pickle serialized file for use in the convLSTM.
- Optimized the learning rate by adding an exponentially decaying learning as opposed to a static rate.
- Created a new optimizer block to improve loss from ~800 to ~130 utilizing tf.clip_by_global_norm() to handle exploding or vanishing gradients.
- Created a new model that uses the convLSTM model as a class, able to feed multiple animations and return the new trained weights (loss increased, but predictions became more accurate).
- Various bug fixes during development.

## Using Blender Resources
- Download the contained .blend file
- Make sure Blender is installed (from [here](https://www.blender.org))
- Double click the .blend file, or open it in Blender with File -> Open..
- Edit the animation output location (line 49) (*Default: /tmp/anims*)
- Navigate to the bottom of the python code window
- Find the button labeled **Run Script** above the console.
- Click the **Run Script** button
- Wait some time, this process may take a while. Blender will freeze but will continue working.

## Using load.py
This python script will take the animations created, push them into 3-D arrays of size (frame_count,image_length,image_width), and create a pickle file for each animation.
- Place this file in the folder containing the animation folders (*Default: /tmp/anims/load.py*)

## Resources (for citation purposes)
- [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104)
- BLOG: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)
- BOOK: [Deep Learning Book](http://www.deeplearningbook.org/)
