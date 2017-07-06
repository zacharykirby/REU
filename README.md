# REU
*Resources and code from my Summer 2017 REU*

## Using Blender Resources
- Download the contained .blend file
- Make sure Blender is installed (from [here](https://www.blender.org))
- Double click the .blend file, or open it in Blender with File -> Open..
- Edit the animation output location (line 49) (*Default: C:/anims/*)
- Navigate to the bottom of the python code window
- Find the button labeled **Run Script** above the console.
- Click the **Run Script** button
- Wait some time, this process may take a while. Blender will freeze but will continue working.

## Using load.py
This python script will take the animations created, push them into 3-D arrays of size (frame_count,image_length,image_width), and create a pickle file for each animation.
- Place this file in the folder containing the animation folders (*Default: C:/anims/load.py*)
