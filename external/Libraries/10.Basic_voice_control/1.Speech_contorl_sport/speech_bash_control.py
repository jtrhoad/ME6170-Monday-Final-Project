from Speech_Lib import Speech
import os,sys,time,threading
sys.path.append('/home/pi/project_demo/lib')
#导入麦克纳姆小车驱动库 Import Mecanum Car Driver Library
from McLumk_Wheel_Sports import *

speed = 10
stop_mytime = 5 
mySpeech = Speech()


Scommand = 999


def start_action():
    global Scommand
    if Scommand == 4:
        move_forward(speed)
        time.sleep(stop_mytime)
        stop_robot()
        
    elif Scommand == 5:
        move_backward(speed)
        time.sleep(stop_mytime)
        stop_robot()
        
    elif Scommand == 6:
        move_left(speed)
        time.sleep(stop_mytime)
        stop_robot()
        
    elif Scommand == 7:
        move_right(speed)
        time.sleep(stop_mytime)
        stop_robot()
        
    elif Scommand == 8:
        rotate_left(speed)  
        time.sleep(stop_mytime)
        stop_robot()
                        
    elif Scommand == 9:
        rotate_right(speed)
        time.sleep(stop_mytime)
        stop_robot()
        
    Scommand = 999








try:
    while 1:
        time.sleep(0.2)
        num = mySpeech.speech_read()
        if num !=999 and num !=0:
            #print(num)
            mySpeech.void_write(num)
            Scommand = num
            
            if num <= 3 and num > 0:#停车
                stop_robot()
                #print(num)
                
            else:
                action_thread = threading.Thread(target=start_action)
                action_thread.daemon = True  
                action_thread.start()
            
            num =999
except:
    mySpeech.__del__()
    print("serial close")