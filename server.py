
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import RPi.GPIO as GPIO
import time
#
# # import the motor library
from RpiMotorLib import RpiMotorLib

# video capture likely to be 0 or 1
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/public", StaticFiles(directory="public"), name="public")

selection = "None"



def gen2_frames():
    # Stepper Motor Setup
    GpioPins = [18, 23, 24, 25]

    # Declare a named instance of class pass a name and motor type
    mymotortest = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")
    # min time between motor steps (ie max speed)
    step_time = .002

    # PID Gain Values (these are just starter values)
    Kp = 0.003
    Kd = 0.0001
    Ki = 0.0001

    # error values
    d_error = 0
    last_error = 0
    sum_error = 0

    while (True):
        _, frame = cap.read()

        output = frame
        if selection == "Red Object":
            output, display = detect_red(frame)
            print("Doing Red")
        elif selection == "Light-Blue Object":
            output, display = detect_blue(frame)
        elif selection == "Circular Object":
            output, x, y = detect_circles(frame)
            display = output
        elif selection == "Facial Recognition":
            output = detect_face(frame)
            display = output
        else:

            ret, buffer = cv2.imencode('.jpg', output)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue


        # run an opening to get rid of any noise
        mask = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(output, cv2.MORPH_OPEN, mask)

        ret, buffer = cv2.imencode('.jpg', display)
        frame_out = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_out + b'\r\n')

        # run connected components algo to return all objects it sees.
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 4, cv2.CV_32S)
        b = np.matrix(labels)
        if num_labels > 1:
            start = time.time()
            # extracts the label of the largest none background component
            # and displays distance from center and image.
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)],
                                      key=lambda x: x[1])
            Obj = b == max_label
            Obj = np.uint8(Obj)
            Obj[Obj > 0] = 255

            # calculate error from center column of masked image
            error = -1 * (320 - centroids[max_label][0])
            speed = Kp * error + Ki * sum_error + Kd * d_error

            # if negative speed change direction
            if speed < 0:
                direction = True 
            else:
                direction = False 

            # inverse speed set for multiplying step time
            # (lower step time = faster speed)
            speed_inv = abs(1 / (speed))
            # get delta time between loops
            delta_t = time.time() - start
            # calculate derivative error
            d_error = (error - last_error) / delta_t
            # integrated error
            sum_error += (error * delta_t)
            last_error = error

            # buffer of 20 only runs within 20
            if abs(error) > 20:
                mymotortest.motor_run(GpioPins, speed_inv * step_time, 1, direction, False, "full", .05)
            else:
                # run 0 steps if within an error of 20
                mymotortest.motor_run(GpioPins, step_time, 0, direction, False, "full", .05)

        else:
            print('no object in view')


    GPIO.cleanup()



def gen_frames():
    # Stepper Motor Setup
    GpioPins = [18, 23, 24, 25]

    # Declare a named instance of class pass a name and motor type
    mymotortest = RpiMotorLib.BYJMotor("MyMotorOne", "28BYJ")
    # min time between motor steps (ie max speed)
    step_time = .002

    # PID Gain Values (these are just starter values)
    Kp = 0.003
    Kd = 0.0001
    Ki = 0.0001

    # error values
    d_error = 0
    last_error = 0
    sum_error = 0

    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)

            output = frame
            if selection == "Red Object":
                output = detect_red(frame)
                print("Doing Red")
            elif selection == "Light-Blue Object":
                output = detect_blue(frame)
            elif selection == "Circular Object":
                output = detect_circles(frame)
            elif selection == "Facial Recognition":
                output = detect_face(frame)
            else:

                ret, buffer = cv2.imencode('.jpg', output)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue


            ret, buffer = cv2.imencode('.jpg', output)
            frame = buffer.tobytes()
            mask = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(output, cv2.MORPH_OPEN, mask)
            try:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 4, cv2.CV_32S)
                b = np.matrix(labels)
            except cv2.error as e:
                print(e)
                continue

            if num_labels > 1:
                start = time.time()
                # extracts the label of the largest none background component
                # and displays distance from center and image.
                max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)],
                                          key=lambda x: x[1])
                Obj = b == max_label
                Obj = np.uint8(Obj)
                Obj[Obj > 0] = 255

                # calculate error from center column of masked image
                error = -1 * (320 - centroids[max_label][0])
                speed = Kp * error + Ki * sum_error + Kd * d_error

                # if negative speed change direction
                if speed < 0:
                    direction = False
                else:
                    direction = True

                # inverse speed set for multiplying step time
                # (lower step time = faster speed)
                speed_inv = abs(1 / (speed))
                # get delta time between loops
                delta_t = time.time() - start
                # calculate derivative error
                d_error = (error - last_error) / delta_t
                # integrated error
                sum_error += (error * delta_t)
                last_error = error

                # buffer of 20 only runs within 20
                if abs(error) > 20:
                    mymotortest.motor_run(GpioPins, speed_inv * step_time, 1, direction, False, "full", .05)
                else:
                    # run 0 steps if within an error of 20
                    mymotortest.motor_run(GpioPins, step_time, 0, direction, False, "full", .05)

            else:
                print('no object in view')

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Define blue color range
        # red is on the upper and lower end of the HSV scale, requiring 2 ranges
    lower1 = np.array([0, 150, 20])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])

        # masks input image with upper and lower red ranges
    red_only1 = cv2.inRange(hsv, lower1, upper1)
    red_only2 = cv2.inRange(hsv, lower2, upper2)
    red_only = red_only1 + red_only2

    img_blue_objects = cv2.bitwise_and(frame, frame, mask=red_only)
    # Create a mask of blue pixels in the image
    return red_only, img_blue_objects 


def detect_circles(frame):
    output = frame.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
    else:

    return output, x, y


def detect_blue(img):
    # Using masks to find objects of a certain color:

    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define blue color range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask of blue pixels in the image
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the original image to extract only blue objects
    img_blue_objects = cv2.bitwise_and(img, img, mask=mask)

    return mask, img_blue_objects 

def detect_face(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face detection using Haar cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    img_faces = img.copy()

    # Draw the faces on the original imIage
    for (x, y, w, h) in faces:
        cv2.rectangle(img_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return img_faces


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen2_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.put('/selection')
async def change_selection(request: Request):
    data = await request.json()

    global selection
    selection = data["selection"]
    print(selection)
    return

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
