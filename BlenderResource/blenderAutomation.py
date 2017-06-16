# Zachary Kirby < zkirby@live.com >

import bpy
import random

# Min X=-4.69
# Max X=4.69
# Min Y=-4.69
# Max =4.69
   
#default frame total
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 30

for x in range(5000):
    
    #clean all keyframes, prevent overlap
    bpy.ops.anim.keyframe_clear_v3d()
    
    #need starting location as floats between 0.0 - 4.0
    startX = random.uniform(-4.69,4.69)
    startY = random.uniform(-4.69,4.69)
    
    #move the circle to the starting location
    bpy.context.scene.frame_set(0)
    bpy.context.object.location[0] = startX
    bpy.context.object.location[1] = startY
    
    #create keyframe at the starting location
    bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')
    
    #determine possible end X location (plus .1, just to ensure some kind of movement)
    #Y location will NOT change
    endX = random.uniform(-4.69,4.69)
    while(endX < (startX + .1)):
        #retry random number if location if left of start
        #!!potentially slow, revise!!
        endX = random.uniform(-4.69,4.69)
    
    #end location determined, compute number of frames @ 10FPS
    #Frames = 3.75 * (endX - startX) + 1
    frames = (3.75 * (endX - startX)) + 1
    bpy.context.scene.frame_end = frames
    
    #goto last frame, move object, set keyframe
    bpy.context.scene.frame_set(frames)
    bpy.context.object.location[0] = endX
    bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')
    
    #prepare animation output
    #change location to your desired output folder
    #add an extra backslash to path
    location = "C:\\Users\\zkirb\\Documents\\Blender\\REU\\anims\\"
    fileName = str('{0:.5f}'.format(startX)) + '_' + str('{0:.5f}'.format(endX)) + '.avi'
    bpy.context.scene.render.filepath = location + fileName
    
    #render animation
    bpy.ops.render.render(animation=True)