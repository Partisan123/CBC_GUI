import threading
import random
import requests
import json
import os

from datetime import datetime, timedelta

# CoreAI parameters
url = "http://192.168.1.237:5000/api/v1"
z_data = {
    "source": "station6_packer",
    "project_id": "212eddc9-ebfc-40a4-99df-100f341a2e2e",
    "time_stamp": "",  # "2020-11-11 11:11:11.111111"
    "data": {
        # here you put any data you want to send to the server
        "object_id": "<pack_id>",
        "defect": True,
        "defect_size": 0.0,
        "bounding_box": {
            "x": 0.0,
            "y": 0.0,
            "width": 0.0,
            "height": 0.0
        }
    }
}

conv_stp = {
    "source": "station4_packer_entrance",
    "project_id": "fcca08f4-b396-4fa5-b8c9-8ccd090469dd",
    "time_stamp": "",
    "data": {
        "halt_time": 0.0
    }
}

# conv_stp_img = {
#     "source": "station4_packer_entrance",
#     "project_id": "37abfa20-5e5c-47d5-a545-31fb8152428f",
#     "time_stamp": ""
# }


class CoreAIThread(threading.Thread):
    def __init__(self, coreai_queue):
        threading.Thread.__init__(self)
        self.coreai_queue = coreai_queue
        self.input_dir = r"C:\Users\User\Desktop\CBC_Project\crop"
        self.name_tmp = ""

    def normalize_st6(self, bbox):

        image_width, image_height = 1024, 640

        if len(bbox) > 0:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            x_normalized = x / image_width
            y_normalized = y / image_height
            w_normalized = w / image_width
            h_normalized = h / image_height

            random_x = random.uniform(0, 1)
            random_y = random.uniform(0, 1)
            random_w = random.uniform(0, 1)
            random_h = random.uniform(0, 1)

            x_randomized = x_normalized * random_x
            y_randomized = y_normalized * random_y
            w_randomized = w_normalized * random_w
            h_randomized = h_normalized * random_h

            randomized_bbox_normalized = [
                x_randomized,  # Randomized normalized x coordinate
                y_randomized,  # Randomized normalized y coordinate
                w_randomized,  # Randomized normalized width
                h_randomized  # Randomized normalized height
            ]
            return randomized_bbox_normalized
        else:
            return None

    def run_station4(self, data):

        current_date = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S.%f")
        conv_stp["time_stamp"] = str(current_date)
        conv_stp["data"]["halt_time"] = float(data[1])
        # conv_stp_img["time_stamp"] = str(current_date)

        response = requests.post("http://192.168.1.237:5000/api/v1/model_data", json=conv_stp)

        # print the response text (the content of the requested file):
        print(response.status_code)
        print(response.text)

        # Get model data id from response
        model_data_id = json.loads(response.text)["modelDataId"]
        path = r'D:\crop_st4\\'
        os.chdir(path)

        # Send image to server
        filename = f"{data[2]}.png"
        files = {"image": open(filename, "rb")}
        response = requests.post("http://192.168.1.237:5000/api/v1/model_data/image", files=files,
                                     data={"model_data_id": model_data_id})

        print(response.status_code)
        print(response.text)

    def run_station6(self, data):

        norm_bbox = self.normalize_st6(data[2])

        current_date = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S.%f")

        z_data["time_stamp"] = current_date
        z_data["data"]["object_id"] = data[3]

        # Code for the coreai functionality
        if norm_bbox is not None:

            z_data["data"]["defect"] = True
            z_data["data"]["defect_size"] = float(
                "%.7f" % (norm_bbox[2] * norm_bbox[3] * random.uniform(0, 1)))

            bounding_box = z_data["data"]["bounding_box"]
            bounding_box["x"] = norm_bbox[0]
            bounding_box["y"] = norm_bbox[1]
            bounding_box["width"] = norm_bbox[2]
            bounding_box["height"] = norm_bbox[3]

            response = requests.post("http://192.168.1.237:5000/api/v1/model_data", json=z_data)
            print(response.status_code)
            print(response.text)
            # model_data_id = json.loads(response.text)["modelDataId"]

        else:

            z_data["data"]["defect"] = False
            z_data["data"]["defect_size"] = 0

            bounding_box = z_data["data"]["bounding_box"]
            bounding_box["x"] = 0
            bounding_box["y"] = 0
            bounding_box["width"] = 0
            bounding_box["height"] = 0

            response = requests.post("http://192.168.1.237:5000/api/v1/model_data", json=z_data)
            print(response.status_code)
            print(response.text)
            model_data_id = json.loads(response.text)["modelDataId"]

        print("\n".join([f"{key}: {value}" for key, value in dict(list(z_data.items())[-5:]).items()]))
        print("====================================================")

    def run(self):

        while True:

            data = self.coreai_queue.get()  # Get data from the queue

            if data[0] == "Station_6":
                self.run_station6(data)

            elif data[0] == "Station_4":
                self.run_station4(data)
