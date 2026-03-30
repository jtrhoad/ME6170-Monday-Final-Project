from Speech_Lib import Speech
import os,sys,time,threading,random
sys.path.append('/home/pi/project_demo/lib')
#导入麦克纳姆小车驱动库 Import Mecanum Car Driver Library
from McLumk_Wheel_Sports import *

mySpeech = Speech()

Scommand = 999
close_light = 0 #关灯标志 close flag
g_brightness = 100


def rgb_remix(val):
    last_val = 0
    if abs(val - last_val) < val % 30:
        val = (val + last_val) % 255
    last_val = val % 255
    return last_val

def rgb_remix_u8(r, g, b):
    if r > 50 and g > 50 and b > 50:
        index = random.randint(0, 2)
        if index == 0:
            r = 0
        elif index == 1:
            g = 0
        elif index == 2:
            b = 0
    return r, g, b

#渐变灯 gradient_light
def gradient_light(speed):
        grad_color = 0
        grad_index = 0
        global close_light,g_brightness

        while close_light == 0:
            if grad_color % 2 == 0:
                gt_red = random.randint(0, g_brightness)
                gt_green = random.randint(0, g_brightness)
                gt_blue = random.randint(0, g_brightness)

                gt_green = rgb_remix(gt_green)
                gt_red, gt_green, gt_blue = rgb_remix_u8(gt_red, gt_green, gt_blue)
                grad_color += 1

            if grad_color == 1:
                if grad_index < 14:
                    number = (grad_index % 14) + 1  # Adjusting for 1-based indexing
                    bot.Ctrl_WQ2812_brightness_Alone(number, gt_red, gt_green, gt_blue)
                    grad_index += 1
                if grad_index >= 14:
                    grad_color = 2
                    grad_index = 0

            elif grad_color == 3:
                if grad_index < 14:
                    number = ((14 - grad_index) % 14)   # Reverse mapping, adjusted for 1-based indexing
                    bot.Ctrl_WQ2812_brightness_Alone(number, gt_red, gt_green, gt_blue)
                    grad_index += 1
                if grad_index >= 14:
                    grad_color = 0
                    grad_index = 0

            time.sleep(speed)

        bot.Ctrl_WQ2812_ALL(0,0)

#流水灯 run_river_light
def run_river_light(speed):

    colors = [0, 1, 2, 3, 4, 5, 6]

    color_index = 0

    num_lights = 14

    global close_light

    while close_light==0:

        for i in range(num_lights - 2):

            bot.Ctrl_WQ2812_Alone(i, 1, colors[color_index])

            bot.Ctrl_WQ2812_Alone(i+1, 1, colors[color_index])

            bot.Ctrl_WQ2812_Alone(i+2, 1, colors[color_index])

            time.sleep(speed)

            bot.Ctrl_WQ2812_ALL(0, 0)

            time.sleep(speed)

        color_index = (color_index + 1) % len(colors)

    bot.Ctrl_WQ2812_ALL(0,0)


#呼吸灯 breathing_light
def breathing_light(speed,current_color):
        breath_direction = 0
        breath_count = 0
        global close_light,g_brightness

        while close_light==0:

            if current_color == 0:  # Red
                r, g, b = breath_count, 0, 0
            elif current_color == 1:  # Green
                r, g, b = 0, breath_count, 0
            elif current_color == 2:  # Blue
                r, g, b = 0, 0, breath_count
            elif current_color == 3:  # Yellow
                r, g, b = breath_count, breath_count, 0
            elif current_color == 4:  # Purple
                r, g, b = breath_count, 0, breath_count
            elif current_color == 5:  # Cyan
                r, g, b = 0, breath_count, breath_count
            elif current_color == 6:  # White
                r, g, b = breath_count, breath_count, breath_count
            else:
                r, g, b = 0, 0, 0  # Default to black if invalid color code

            bot.Ctrl_WQ2812_brightness_ALL(r, g, b)
            time.sleep(speed)

            if breath_direction == 0:
                breath_count += 2
                if breath_count >= g_brightness:
                    breath_count=g_brightness
                    breath_direction = 1
            else:
                breath_count -= 2
                if breath_count < 0:
                    breath_direction = 0
                    breath_count = 0
                    
            bot.Ctrl_WQ2812_ALL(0,0)



def start_action():
    global Scommand,close_light 
        
    if Scommand == 11:
        bot.Ctrl_WQ2812_ALL(1,0) #红色 red
        
    elif Scommand == 12:
        bot.Ctrl_WQ2812_ALL(1,1)#绿色  green
        
    elif Scommand == 13:
        bot.Ctrl_WQ2812_ALL(1,2)#蓝色 blue
        
    elif Scommand == 14:
        bot.Ctrl_WQ2812_ALL(1,3)#黄色 yellow
        
    elif Scommand == 15:
        close_light = 1
        time.sleep(0.02)
        close_light = 0
        run_river_light(0.03)#流水 run_river_light
        
        
    elif Scommand == 16:
        close_light = 1
        time.sleep(0.02)
        close_light = 0
        gradient_light(0.08)
        
    elif Scommand == 17:
        close_light = 1
        time.sleep(0.02)
        close_light = 0
        breathing_light(0.03,4)#呼吸 breath

        
    Scommand = 999







try:
    while 1:
        time.sleep(0.2)
        num = mySpeech.speech_read()
        if num !=999 and num !=0:
            #print(num)
            mySpeech.void_write(num)
            Scommand = num
            
            if Scommand == 10:
                bot.Ctrl_WQ2812_ALL(0,0)
                close_light = 1
                
            else:
                close_light = 0
                action_thread = threading.Thread(target=start_action)
                action_thread.daemon = True  
                action_thread.start()
            
            num =999
except:
    mySpeech.__del__()
    print("serial close")
