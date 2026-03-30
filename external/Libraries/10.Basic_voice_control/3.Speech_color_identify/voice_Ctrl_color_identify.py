import cv2 as cv
from Speech_Lib import Speech
import time,threading

H_min=0 
H_max=0
S_min=0
H_center = (H_min+H_max)//2


def start_action():
    global H_min,H_max,S_min,H_center
    myspe = Speech()
    while True:
        time.sleep(0.02)
        command_result = myspe.speech_read()
        if command_result !=999:
            print(command_result)   
        if command_result == 60:
            H_center = (H_min+H_max)//2
            print(H_min,H_max,H_center,S_min)
            
            if H_min <= 10 or H_max >= 170:  # 红色检测 red
                if S_min > 50: 
                    print("red")
                    myspe.void_write(61)
            elif 20 <= H_center <= 75:       # 黄色检测 yellow
                if S_min > 50: 
                    print("yellow")
                    myspe.void_write(64)
            elif 75 <= H_center <= 85:       # 绿色检测 green
                if S_min > 50: 
                    # 额外区分高饱和度绿色
                    if S_min < 200:
                        print("green")
                        myspe.void_write(63)
            elif 90 <= H_center <= 130:     # 蓝色检测 blue
                if S_min > 110:  
                    print("blue")
                    myspe.void_write(62)


class Color_identify():
    def __init__(self):
        self.img = None
        self.hsv_range = ()
        self.Roi_init = ()
        self.cols, self.rows = 0, 0
        self.Mouse_XY = (0, 0)
        self.select_flags = False
        self.windows_name = "frame"
    def onMouse(self, event, x, y, flags, param):
        if event == 1:
            self.select_flags = True
            self.Mouse_XY = (x,y)
        if event == 4:
            self.select_flags = False
        if self.select_flags == True:
            self.cols = min(self.Mouse_XY[0], x), min(self.Mouse_XY[1], y)
            self.rows = max(self.Mouse_XY[0], x), max(self.Mouse_XY[1], y)
            self.Roi_init = (self.cols[0], self.cols[1], self.rows[0], self.rows[1])
            print(self.Roi_init)
    def process(self,rgb_img):
        global H_min,H_max,S_min
        H = [];
        S = [];
        V = [];
        cv.setMouseCallback(self.windows_name, self.onMouse, 0)
        if self.select_flags == True:
            cv.line(rgb_img, self.cols, self.rows, (255, 0, 0), 2)
            cv.rectangle(rgb_img, self.cols, self.rows, (0, 255, 0), 2)
            if self.Roi_init[0]!=self.Roi_init[2] and self.Roi_init[1]!=self.Roi_init[3]:
                HSV = cv.cvtColor(rgb_img,cv.COLOR_BGR2HSV)
                for i in range(self.Roi_init[0], self.Roi_init[2]):
                    for j in range(self.Roi_init[1], self.Roi_init[3]):
                        H.append(HSV[j, i][0])
                        S.append(HSV[j, i][1])
                        V.append(HSV[j, i][2])
                H_min = min(H); H_max = max(H)
                S_min = min(S); S_max = 253
                V_min = min(V); V_max = 255
                # print("H_max: ",H_max)
                # print("H_min: ",H_min)        
                lowerb = 'lowerb : (' + str(H_min) + ' ,' + str(S_min) + ' ,' + str(V_min) + ')'
                upperb = 'upperb : (' + str(H_max) + ' ,' + str(S_max) + ' ,' + str(V_max) + ')'
                cv.putText(rgb_img, lowerb, (150, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv.putText(rgb_img, upperb, (150, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
        #command_result = self.spe.speech_read()
        #self.spe.void_write(command_result)
        

        return rgb_img
                        
                    

if __name__ == '__main__':
    try:
        color_identify = Color_identify()
        capture = cv.VideoCapture(0)
        cv_edition = cv.__version__
        if cv_edition[0]=='3': capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'XVID'))
        else: capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cv.namedWindow("frame",cv.WINDOW_AUTOSIZE)
        imshow_num = 0
        
        #启动语音识别线程 start voice thread
        action_thread = threading.Thread(target=start_action)
        action_thread.daemon = True  
        action_thread.start()
        
        while capture.isOpened():
            start = time.time()
            ret, frame = capture.read()
            #cv.imshow("frame", frame)
            action = cv.waitKey(10) & 0xFF
            rgb_img = color_identify.process(frame)
            imshow_num +=1
    
            if imshow_num%2==0:
                cv.imshow("frame", rgb_img)
                imshow_num = 0
    
            if action == ord('q') or action == 113: break
    except:
        capture.release()
        cv.destroyAllWindows()












