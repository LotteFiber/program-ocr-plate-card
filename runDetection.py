import paho.mqtt.client as mqtt
from PlateReadOCR import startProgramPlateDetection
from CardReadOCR import startProgramCardDetection
import json

host = "localhost"

def on_connect(self, client, userdata, rc):
    print("MQTT Connected")
    self.subscribe("start_Program_Plate_Detection_Image")
    self.subscribe("start_Program_Card_Detection_Image")
    self.subscribe("start_Program_Check_Card")
    # self.publish("program_status_update","Hello")


def on_message(client, userdata, msg):
    print("on_message.")
    data = json.loads(msg.payload.decode("utf-8", "strict"))
    print(data)
    if data['message'] == 'start program plate':
        uuid = data['uuid']
        startProgramPlateDetection(uuid)

    elif data['message'] == 'start program card':
        uuid = data['uuid']
        startProgramCardDetection(uuid)



client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(host)
client.loop_forever()
