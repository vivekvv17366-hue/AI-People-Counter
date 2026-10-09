import cv2
import csv
from ultralytics import YOLO
from datetime import datetime

# we use yolo11n.pt  here 
model=YOLO('yolo11n.pt')


video=cv2.VideoCapture(0)

entered=0
exited=0

last_side={}


csv_file=open('people_count.csv','a',newline='')

csv_writer=csv.writer(csv_file)

if csv_file.tell()==0:
    csv_writer.writerow(['Time','Entered','Exited','Current People'])

last_save_time=datetime.now()

while True:
    suc,img=video.read()

    if not suc:
        print('Could not access webcam')
        break

    height,width=img.shape[:2]

    line_y= height // 2

    results=model.track(img,persist=True)

    result=results[0]

    display_img=img.copy()

    cv2.rectangle(display_img,(0,0),(300,130),(40,40,40),-1)

    cv2.line(display_img,(0,line_y),(width,line_y),(255,0,0),2)

    for bound in result.boxes:

        class_id=int(bound.cls[0])

        if class_id == 0:

            x1, y1, x2, y2 = map(int,bound.xyxy[0])

            confidence=float(bound.conf[0])

            if bound.id is not None:
                track_id=int(bound.id[0])
            else:
                track_id= -1

            center_x=(x1 + x2) // 2
            center_y=(y1 + y2) // 2


            if center_y < line_y:
                current_side = 'above'
            else:
                current_side= "below"

            if track_id in last_side:
                previous_side=last_side[track_id]

                if previous_side == 'above' and current_side == 'below':
                    entered +=1

                elif previous_side == 'below' and current_side == 'above':
                    exited+=1

            last_side[track_id]=current_side



            cv2.rectangle(display_img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.circle(display_img,(center_x,center_y),5,(0,0,255),-1)

            label= f"person ID:{track_id} | {confidence:.2f}"

            cv2.putText(display_img,label,(x1,y1 -10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            cv2.putText(display_img,f'Entered:{entered}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.putText(display_img,f'EXITED:{exited}',(20,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            
            current_people= entered - exited

            current_time=datetime.now()

            if (current_time-last_save_time).total_seconds() >=5:
                csv_writer.writerow([current_time.strftime("%Y-%m-%d %H:%M:%S"),entered,exited,current_people])
            csv_file.flush()
            last_save_time=current_time

            cv2.putText(display_img,'AI PEOPLE COUNTER',(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            cv2.putText(display_img,f"ENTERED {entered}",(15,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            cv2.putText(display_img,f"Exited {exited}",(15,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            cv2.putText(display_img,f"Current {current_people}",(15,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(display_img,f'CURRENT PEOPLE:{current_people}',(20,110),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('AI People Detector',display_img)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
csv_file.close()