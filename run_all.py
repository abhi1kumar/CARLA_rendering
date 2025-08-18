import sys
import os

# NOTE BELOW COMMAND IS FOR NVIDIA NGC

CMD = "ngc batch run --name \"ml-model.carla-generate-YAW{2}-PITCH{3}-HEIGHT{4}\" --preempt RUNONCE --min-timeslice 86400s --total-runtime 86400s --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \"bash /tzofi/repos/scrape-bev-carla/run_new.sh {0} {1} {2} {3} {4}\" --result /results --image \"nvidian/scrape-bev-carla:latest\" --org nvidian --team swaiinf --workspace K4NXh5DnTseAzWe0m1BYQQ:/tzofi:RW"

args = [[0, "pitch", 0, 0, 0],
        ["-4_6", "pitch_height", 0, -4, 0.0762*2],
        ["-8_12", "pitch_height", 0, -8, 0.0762*4],
        ["-12_18", "pitch_height", 0, -12, 0.0762*6],
        ["-16_24", "pitch_height", 0, -16, 0.0762*8],
        ["-20_30", "pitch_height", 0, -20, 0.0762*10],
        [0, "", 0, -4, 0.0762*2],
        [-4, "pitch", 0, -4, 0],
        [-8, "pitch", 0, -8, 0],
        [-12, "pitch", 0, -12, 0],
        [-16, "pitch", 0, -16, 0],
        [-20, "pitch", 0, -20, 0],
        [4, "pitch", 0, 4, 0],
        [8, "pitch", 0, 8, 0],
        [12, "pitch", 0, 12, 0],
        [16, "pitch", 0, 16, 0],
        [20, "pitch", 0, 20, 0],
        [-4, "yaw", -4, 0, 0],
        [-8, "yaw", -8, 0, 0],
        [-12, "yaw", -12, 0, 0],
        [-16, "yaw", -16, 0, 0],
        [-20, "yaw", -20, 0, 0],
        [4, "yaw", 4, 0, 0],
        [8, "yaw", 8, 0, 0],
        [12, "yaw", 12, 0, 0],
        [16, "yaw", 16, 0, 0],
        [20, "yaw", 20, 0, 0],
        [3, "height", 0, 0, 0.0762],
        [6, "height", 0, 0, 2*0.0762],
        [9, "height", 0, 0, 3*0.0762],
        [12, "height", 0, 0, 4*0.0762],
        [15, "height", 0, 0, 5*0.0762],
        [18, "height", 0, 0, 6*0.0762],
        [21, "height", 0, 0, 7*0.0762],
        [24, "height", 0, 0, 8*0.0762],
        [27, "height", 0, 0, 9*0.0762],
        [30, "height", 0, 0, 10*0.0762]]

for arg in args:
    #print(CMD.format(arg[0], arg[1], arg[2], arg[3], arg[4]))
    #exit()
    os.system(CMD.format(arg[0], arg[1], arg[2], arg[3], arg[4]))
