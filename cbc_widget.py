import cv2
import sys
import time
import os
import random
import threading

import numpy as np
import pandas as pd
import pyzed.sl as sl
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ultralytics import YOLO
from coreAIThread import CoreAIThread
from queue import Queue
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime, timedelta

# Parameters for station 4
MIN_MOTION_PIXELS = 500
MAX_CONSECUTIVE_STOP_FRAMES = 60


class Station6Det(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QtGui.QImage)
    datas = QtCore.pyqtSignal(list)

    # curr_ids = QtCore.pyqtSignal(int)

    def __init__(self, rtsp_url, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.status = True
        # self.cap = True

        self.FPS = 1 / 50
        self.FPS_MS = int(self.FPS * 1000)
        self.rtsp_url = rtsp_url
        self.model = YOLO("weights/best_yolov8.pt")
        self.classes = self.model.names
        self.line_cascade = cv2.CascadeClassifier('cascade.xml')
        self.device = 'cuda'

        try:
            self.cap = cv2.VideoCapture('http://192.168.1.100:81/stream')
        except:
            print("=======ERROR WITH THE CAP=======")
            self.cap = cv2.VideoCapture(self.rtsp_url)

        # self.cap = cv2.VideoCapture("rtsp://192.168.1.200:8554/test")
        self.stop_requested = False
        self.detection_thread = None

        self.name = ""

        self.resolution = (640, 480)
        self.id = 0
        self.defect_id = 0

        self.defect_bbox = []

        self.prev_id = 0
        self.pack_id = 0
        self.pack_bbox = []

        self.cas_x = 0
        self.cas_y = 0
        self.flag_cass = None

        self.date = 0

        self.coreai_queue = Queue()

    def send_data_to_coreai(self, data):
        self.coreai_queue.put(data)

    def cascade(self, frame):
        line_cas = self.line_cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        line_rect = line_cas.detectMultiScale(blur, scaleFactor=3, minNeighbors=12, minSize=(12, 12))

        if line_rect is ():
            flag = False
        else:
            flag = True

        for (x, y, w, h) in line_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            x_center = x + w / 2
            y_center = y + (h / 2)
            self.cas_x = x_center
            self.cas_y = y_center
            self.defect_bbox = [x, y, w, h]

        return frame, flag, self.defect_bbox

    def z_detection(self, bbox, flag):
        cas_x, cas_y = self.cas_x, self.cas_y

        if cas_x > 0 and cas_y > 0 and flag:
            if (bbox[0][0] < cas_x < bbox[0][2]) and (bbox[0][1] < cas_y < bbox[0][3]):
                if self.defect_id != self.id:  # Check if self.defect_id is not equal to self.id

                    self.defect_id = self.id
                    self.curr_id(self.defect_id, True)
                    self.data()

        else:
            self.curr_id(self.id, False)

    def crop_z(self, frame, matching_id):
        path = r'D:\crop_st6\\'
        os.chdir(path)

        if matching_id is not None and matching_id != self.name:
            filename = f"{self.name}.png"
            output_path = os.path.join(path, filename)
            cv2.imwrite(output_path, frame)

    def data(self):
        crop_id = self.defect_id
        date = self.date
        margin = round(0.7 * random.uniform(0, 2), 2)
        self.datas.emit([str(date), str(crop_id), str(margin)])

    def curr_id(self, ids, defect):
        if ids != self.prev_id:  # Check if the current id is different from the previous id
            self.date = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            self.name = f'{str(self.date)}-ID-{ids}'

            # self.curr_ids.emit(ids)
            self.pack_id = ids

            if defect:
                self.send_data_to_coreai(["Station_6", str(self.name), self.defect_bbox, self.pack_id])
            else:
                self.name = f'{str(self.date)}-ID-{ids}'
                self.send_data_to_coreai(["Station_6", str(self.name), [], self.pack_id])

            self.prev_id = ids  # Update the previous id with the current id

    def detection_func(self):

        x_center = 0
        frame_count = 0
        # self.cap = cv2.VideoCapture()

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 12)

        while not self.stop_requested:
            ret, frame_orig = self.cap.read()

            if not ret:
                # self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
                continue

            start_time = time.perf_counter()
            time.sleep(self.FPS)
            frame_res = cv2.resize(frame_orig, self.resolution)
            frame_cp = frame_orig.copy()

            results = self.model.track(frame_res, imgsz=640, persist=True, tracker="bytetrack.yaml",
                                       conf=0.8, iou=0.6, verbose=False)

            if results[0].boxes is None or results[0].boxes.id is None:
                continue

            # Get the boxes and track IDs
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy()
            self.id = int(track_ids[0])

            # # Visualize the results on the frame
            # annotated_frame = results[0].plot()

            frame1, flag_cas, bbox_cas = self.cascade(frame_res)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                x_center = x1 + (x2 - x1) // 2

                cv2.rectangle(frame_res, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Put the track_id as text next to the rectangle
                text = f"ID: {int(track_id)}"
                cv2.putText(frame_res, text, (x1 + 30, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # cv2.line(frame_res, (330, 0), (330, frame_res.shape[0]), (0, 255, 0), 2)  # Green line for x=400
                # cv2.line(frame_res, (400, 0), (400, frame_res.shape[0]), (0, 255, 0), 2)  # Green line for x=500

            self.flag_cass = flag_cas

            if flag_cas and 330 < self.cas_x < 400:
                self.z_detection(boxes, True)
                self.crop_z(frame_cp, self.id)
            elif 440 < x_center < 455:
                self.z_detection(boxes, False)

            frame_count += 1

            end_time = time.perf_counter()
            frame_time = end_time - start_time
            fps = 3.0 / frame_time
            fps_display = f"FPS: {fps:.1f}"
            cv2.putText(frame_res, str(fps_display), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # cv2.waitKey(self.FPS_MS)

            rgb_img = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytesPerLine = ch * w
            convert2qt = QtGui.QImage(rgb_img.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            scale = convert2qt.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
            self.changePixmap.emit(scale)

    def run(self):

        try:
            coreai_thread = CoreAIThread(self.coreai_queue)
            coreai_thread.start()

            self.detection_thread = threading.Thread(target=self.detection_func)
            self.detection_thread.start()
            self.detection_thread.join()
        except Exception as e:
            print("Exception in thread:", e)
        finally:
            # Clean up resources when the thread exits, regardless of exceptions
            self.cap.release()  # Release video capture object


class Station4Det(QtCore.QThread):
    changePixmap2 = QtCore.pyqtSignal(QtGui.QImage)
    station_4_data = QtCore.pyqtSignal(list)

    def __init__(self, video_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        # Initialize ZED camera
        # self.zed = sl.Camera()
        # init_params = sl.InitParameters()
        # init_params.camera_resolution = sl.RESOLUTION.HD720
        # # init_params.grab_compute_capping_fps = 60
        # init_params.camera_fps = 120
        # err = self.zed.open(init_params)
        # if err != sl.ERROR_CODE.SUCCESS:
        #     print("Error initializing ZED camera!")
        #     exit(1)

        self.image = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

        self.stop_requested = False
        self.timer = 0
        self.state = None
        self.frame_shot = None
        self.date = None
        self.detection_thread = None

        self.FPS = 1 / 1000
        self.FPS_MS = int(self.FPS * 1000)

        self.coreai_queue = Queue()

    def send_data_to_coreai(self, data):
        self.coreai_queue.put(data)

    def states_emit(self):

        timer = self.timer
        self.send_data_to_coreai(["Station_4", timer, self.date])
        self.station_4_data.emit([str(self.date), str(timer)])

    def station_stopped_frame(self, frame):
        path = r'D:\crop_st4\\'
        os.chdir(path)
        self.date = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        filename = f"{self.date}.png"
        output_path = os.path.join(path, filename)
        cv2.imwrite(output_path, frame)

    def detect_func(self):

        # Initialize variables
        frame_index = 0
        motion_window = []
        motion_threshold_low = 10.5
        motion_threshold_high = 30
        motion_flag = False
        motion_start_time = None
        thresholded = None

        # ROI
        x = 250
        y = 180
        width = 90
        height = 100

        key = ''
        while not self.stop_requested:
            # if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            #     self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            #     frame = self.image.get_data()
            #
            # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            ret, frame_orig = self.cap.read()
            if not ret:
                # self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
                continue
            h = frame_orig.shape[0]
            w = frame_orig.shape[1]
            frame_cut = frame_orig[0:int(h), 0:int(w / 2)]
            frame_res = cv2.resize(frame_cut, (640, 480))

            frame_cropped = frame_res[y:y + height, x:x + width]

            # Calculate the change in pixels between consecutive frames
            if frame_index > 0:
                prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_frame = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_frame, curr_frame)
                blurred = cv2.GaussianBlur(diff, (5, 5), 0)
                _, thresholded = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

                # frame_diff = cv2.absdiff(prev_frame_blurred, curr_frame_blurred)
                cv2.rectangle(frame_res, (x, y), (x + width, y + height), (0, 255, 0), 2)

                motion_pixels = np.count_nonzero(thresholded)
                motion_percentage = (motion_pixels / thresholded.size) * 100
                motion_window.append(motion_percentage)

            # Update previous frame
            prev_frame = frame_cropped.copy()
            frame_index += 1

            # Calculate average motion for every 10 frames
            if frame_index % 5 == 0:
                avg_motion_percentage = np.mean(motion_window)

                if not motion_flag and avg_motion_percentage < motion_threshold_low:
                    motion_flag = True
                    motion_start_time = time.time()
                    self.state = motion_flag
                    self.station_stopped_frame(frame_res)
                elif motion_flag and avg_motion_percentage > motion_threshold_high:
                    motion_flag = False
                    elapsed_time = time.time() - motion_start_time
                    self.state = motion_flag
                    self.timer = "{:.2f}".format(elapsed_time)
                    try:
                        self.states_emit()

                    except:
                        print("=======ERROR CONNECTION WITH COREAI=======")
                        continue
                    motion_start_time = None

                motion_window = []

            # cv2.waitKey(self.FPS_MS)

            # if thresholded is not None:
            display_img = frame_res  # Change this to see different stages of processing
            rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytesPerLine = ch * w
            convert2qt = QtGui.QImage(rgb_img.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            scale = convert2qt.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
            self.changePixmap2.emit(scale)

        # sys.exit(-1)

    def run(self):
        try:
            coreai_thread = CoreAIThread(self.coreai_queue)
            coreai_thread.start()

            self.detection_thread = threading.Thread(target=self.detect_func)
            self.detection_thread.start()
            self.detection_thread.join()
        except Exception as e:
            print("Exception in thread:", e)
        finally:
            self.cap.release()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.th = None
        self.th2 = None
        self.data_dict = {}
        self.setWindowTitle("Test App")
        self.resize(1100, 600)
        self.row_inx = 0
        self.row_inx2 = 0

        self.signal = None

        # Graph/Plot

        # St6
        self.defects_list = []
        self.margins_list = []
        self.total_fail_margins = 0
        self.total_entries = 0

        # St4
        self.start_time = None
        self.total_time_on = timedelta()
        self.total_time_off = timedelta()

        # CSV
        self.buffer_size_limit = 20  # Save every 20 records
        self.last_csv_creation_time = datetime.now()

        self.data_buffer_table1 = []
        self.data_buffer_table2 = []
        self.current_csv_path_table1 = self.generate_csv_filename("table1")
        self.current_csv_path_table2 = self.generate_csv_filename("table2")

        # QTimer to save data every 5 minutes
        self.save_timer = QtCore.QTimer(self)
        self.save_timer.timeout.connect(lambda: self.flush_buffer_to_csv("table1"))
        self.save_timer.timeout.connect(lambda: self.flush_buffer_to_csv("table2"))

        self.save_timer.start(5 * 60 * 1000)  # 5 minutes in milliseconds

        # Declarations
        self.table = QtWidgets.QTableWidget(self)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.table.verticalHeader().setVisible(False)

        self.table2 = QtWidgets.QTableWidget(self)
        self.table2.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table2.resizeColumnsToContents()
        self.table2.verticalHeader().setVisible(False)

        self.vid_label = QtWidgets.QLabel(self)
        self.vid_label2 = QtWidgets.QLabel(self)

        # CV Threads
        # Z
        # self.th = Station6Det("rtsp://192.168.1.200:8554/test")

        # self.th = Station6Det('videos/test28.avi')
        # self.th.finished.connect(self.close)
        # self.th.changePixmap.connect(self.setImage)
        # self.th.datas.connect(self.event_table)
        #
        #
        #
        # # Station 4 (work/halt)
        #
        # self.th2 = Station4Det("videos/test_main.mp4")
        # self.th2.finished.connect(self.close)
        # self.th2.changePixmap2.connect(self.setImage2)
        # self.th2.station_4_data.connect(self.event_table2)

        # Main Layout
        main_layout = QtWidgets.QGridLayout()
        lay1_v = QtWidgets.QVBoxLayout()

        # Misc
        self.button1 = QtWidgets.QPushButton("Start")
        self.button2 = QtWidgets.QPushButton("Stop")
        self.button3 = QtWidgets.QPushButton("Start")
        self.button4 = QtWidgets.QPushButton("Stop")

        # Create tabs
        # Tab1
        self.tab1 = QtWidgets.QWidget()
        self.tab1.layout = QtWidgets.QVBoxLayout()
        self.tab1.layout2 = QtWidgets.QHBoxLayout()

        self.tab1.layout.addWidget(self.table, 0, QtCore.Qt.AlignRight)
        self.tab1.layout.addWidget(self.button1, 0, QtCore.Qt.AlignBaseline)
        self.tab1.layout.addWidget(self.button2, 0, QtCore.Qt.AlignBaseline)
        self.tab1.layout2.addWidget(self.vid_label, 0, QtCore.Qt.AlignLeft)

        self.tab1.layout2.addLayout(self.tab1.layout)
        self.tab1.setLayout(self.tab1.layout2)

        # Tab2
        self.tab2 = QtWidgets.QWidget()
        self.tab2.layout = QtWidgets.QHBoxLayout()
        self.tab2.layout2 = QtWidgets.QVBoxLayout()

        self.tab2.layout2.addWidget(self.table2, 0, QtCore.Qt.AlignRight)
        self.tab2.layout2.addWidget(self.button3, 0, QtCore.Qt.AlignBaseline)
        self.tab2.layout2.addWidget(self.button4, 0, QtCore.Qt.AlignBaseline)
        self.tab2.layout.addWidget(self.vid_label2, 0, QtCore.Qt.AlignLeft)

        self.tab2.layout.addLayout(self.tab2.layout2)
        self.tab2.setLayout(self.tab2.layout)

        # Tab 3
        self.tab3 = QtWidgets.QWidget()
        self.tab3.layout = QtWidgets.QGridLayout()

        # Create the canvas for Station 6 and add it to the grid layout
        self.canvas_station6 = FigureCanvas(self.plot_data_station6())
        self.tab3.layout.addWidget(self.canvas_station6, 0, 0)  # Add the canvas for Station 6 to the left

        # Create the canvas for Station 4 and add it to the grid layout
        self.canvas_station4 = FigureCanvas(self.plot_data_station4())
        self.tab3.layout.addWidget(self.canvas_station4, 0, 1)  # Add the canvas for Station 4 to the right

        self.tab3.setLayout(self.tab3.layout)

        # Tab Parent
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab1, "Station 6")
        self.tabs.addTab(self.tab2, "Station 4")
        self.tabs.addTab(self.tab3, "Overall")

        # Set Main layout
        main_layout.addWidget(self.tabs, 0, 0)
        self.setLayout(main_layout)

        # Actions
        # Station 6

        # self.button1.pressed.connect(lambda: self.start_thread(self.th, self.button1, self.button2))
        # self.button3.pressed.connect(lambda: self.start_thread(self.th2, self.button3, self.button4))

        # self.button2.pressed.connect(lambda: self.kill_thread(self.th, self.button1, self.button2))
        # self.button4.pressed.connect(lambda: self.kill_thread(self.th2, self.button3, self.button4))
        #
        # self.start_thread(self.th, self.button1, self.button2)
        # self.start_thread(self.th2, self.button3, self.button4)

        self.button2.setEnabled(False)
        self.button4.setEnabled(False)

        self.init_threads()

    def init_threads(self):

        try:
            self.th = Station6Det('http://192.168.1.100:81/stream')
            # self.th = Station6Det("rtsp://192.168.1.200:8554/test")
            # self.th = Station6Det("videos/test25.avi")
            self.th.finished.connect(self.close)
            self.th.changePixmap.connect(self.setImage)
            self.th.datas.connect(self.event_table)

            # Station 4 (work/halt)

            self.th2 = Station4Det("videos/test_main.mp4")
            self.th2.finished.connect(self.close)
            self.th2.changePixmap2.connect(self.setImage2)
            self.th2.station_4_data.connect(self.event_table2)
        except:
            print("Error initializing threads. Retrying in 5 seconds...")
            QtCore.QTimer.singleShot(5000, self.initialize_threads)

    def showEvent(self, event):
        super(MainWindow, self).showEvent(event)

        # Start the first thread after 1 seconds
        QtCore.QTimer.singleShot(1000, lambda: self.start_thread(self.th, self.button1, self.button2))

        # Start the second thread after 8 seconds
        QtCore.QTimer.singleShot(7000, lambda: self.start_thread(self.th2, self.button3, self.button4))

    def kill_thread(self, thread, start_button, stop_button):

        print("Finishing...")
        stop_button.setEnabled(False)
        start_button.setEnabled(True)
        thread.stop_requested = True
        thread.wait()
        thread.cap.release()
        cv2.destroyAllWindows()

    def start_thread(self, thread, start_button, stop_button):
        print("Starting...")
        stop_button.setEnabled(True)
        start_button.setEnabled(False)

        if thread == self.th2:
            self.start_time = datetime.now()

        thread.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.vid_label.setPixmap(QtGui.QPixmap.fromImage(image))

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage2(self, image):
        self.vid_label2.setPixmap(QtGui.QPixmap.fromImage(image))

    def generate_csv_filename(self, table_name):
        now = datetime.now()
        filename = "{}_{}.csv".format(table_name, now.strftime("%Y_%m_%d_%H_%M_%S"))
        return r'C:\Users\User\Desktop\CBC_Project\tables\{}'.format(filename)

    def add_to_buffer(self, data, table_name):
        if table_name == "table1":
            self.data_buffer_table1.append(data)
            if len(self.data_buffer_table1) >= self.buffer_size_limit:
                self.flush_buffer_to_csv("table1")
        elif table_name == "table2":
            self.data_buffer_table2.append(data)
            if len(self.data_buffer_table2) >= self.buffer_size_limit:
                self.flush_buffer_to_csv("table2")

    def flush_buffer_to_csv(self, table_name):
        # Check if 10 hours have passed since the last CSV creation
        now = datetime.now()
        hours_elapsed = (now - self.last_csv_creation_time).total_seconds() / 3600

        if hours_elapsed >= 10:
            self.last_csv_creation_time = now
            self.flush_buffer_to_csv("table1")
            self.flush_buffer_to_csv("table2")
            self.current_csv_path_table1 = self.generate_csv_filename("table1")
            self.current_csv_path_table2 = self.generate_csv_filename("table2")
            self.row_inx = 0
            self.row_inx2 = 0

        if table_name == "table1":
            buffer = self.data_buffer_table1
            current_csv_path = self.current_csv_path_table1
            columns = ['Date', 'ID', 'Fail Marg.']  # Adjust columns as needed for table1
        elif table_name == "table2":
            buffer = self.data_buffer_table2
            current_csv_path = self.current_csv_path_table2
            columns = ['Date', 'Down Time']  # Adjust columns as needed for table2

        if buffer:
            df = pd.DataFrame(buffer, columns=columns)
            df.to_csv(current_csv_path, mode='a', header=False, index=False)  # Append mode
            buffer.clear()  # Clear the buffer

    def event_table(self, data_of_cropped):
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Date', 'ID', 'Fail Marg.'])

        # Insert data at the top row
        self.table.insertRow(0)
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(data_of_cropped[0]))
        self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(data_of_cropped[1]))
        self.table.setItem(0, 2, QtWidgets.QTableWidgetItem(data_of_cropped[2]))

        self.row_inx += 1

        # Remove the oldest row if row count exceeds 35
        if self.table.rowCount() > 35:
            self.table.removeRow(self.table.rowCount() - 1)
            self.row_inx -= 1

        self.defects_list.append(int(data_of_cropped[1]))
        self.margins_list.append(float(data_of_cropped[2]))

        self.add_to_buffer([data_of_cropped[0], data_of_cropped[1],
                            data_of_cropped[2]], "table1")

        self.canvas_station6.figure = self.plot_data_station6()
        self.canvas_station6.draw()

    def event_table2(self, state):
        self.table2.setColumnCount(2)
        self.table2.setHorizontalHeaderLabels(['Date', 'Down Time'])

        # Insert data at the top row
        self.table2.insertRow(0)
        self.table2.setItem(0, 0, QtWidgets.QTableWidgetItem(state[0]))
        self.table2.setItem(0, 1, QtWidgets.QTableWidgetItem(state[1]))

        self.row_inx2 += 1

        # Remove the oldest row if row count exceeds 35
        if self.table2.rowCount() > 35:
            self.table2.removeRow(self.table2.rowCount() - 1)
            self.row_inx2 -= 1

        self.add_to_buffer([state[0], state[1]], "table2")

        halt_time_seconds = float(state[1])

        self.total_time_on += timedelta(seconds=1)
        self.total_time_off += timedelta(seconds=halt_time_seconds)

        # Calculate the total_time_on dynamically
        if self.start_time:
            elapsed_time = datetime.now() - self.start_time
            self.total_time_on = elapsed_time - self.total_time_off

        # Refresh the Station 4 graph
        self.canvas_station4.figure = self.plot_data_station4()
        self.canvas_station4.draw()

    def plot_data_station6(self):
        # Extract data from the QTableWidget
        rows = self.table.rowCount()
        margins_list = [float(self.table.item(row, 2).text()) for row in range(rows)]

        # Use only the last 20 entries
        margins_list = margins_list[-35:]

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(margins_list, label='Fail Margin')

        # Check if margins_list is not empty
        if margins_list:
            # Calculate the mean of the entire margins_list and plot it
            mean_value = sum(margins_list) / len(margins_list)
            ax.axhline(mean_value, color='red', linestyle='-.', label='Mean Value')

            # Annotate the mean value on the plot
            ax.annotate(f'Mean: {mean_value:.2f}', xy=(len(margins_list) - 1, mean_value),
                        xytext=(len(margins_list) - 1, mean_value + 0.1),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        horizontalalignment='right')
        else:
            mean_value = 0

        ax.set_title('Station 6 Stat')
        ax.set_xlabel('Index')
        ax.set_ylabel('Fail Margin')
        ax.legend()

        return fig

    def plot_data_station4(self):
        fig2 = Figure(figsize=(5, 4), dpi=100)
        ax2 = fig2.add_subplot(111)

        if self.total_time_on and self.total_time_off:
            efficiency = self.total_time_on.total_seconds() / (
                    self.total_time_on.total_seconds() + self.total_time_off.total_seconds())
        else:
            efficiency = 0

        inefficiency = 1 - efficiency
        ax2.pie([efficiency, inefficiency], labels=['Time On', 'Time Off'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Efficiency')
        return fig2


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ex = MainWindow()

    File = open("Combinear.qss", 'r')

    with File:
        qss = File.read()
        app.setStyleSheet(qss)

    ex.show()
    sys.exit(app.exec_())
