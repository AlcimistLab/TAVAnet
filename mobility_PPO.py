import sys
import os
import math
import time
import subprocess
import csv
import socket
import json
import pytz
import datetime

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import traci

from threading import Thread
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.sumo.runner import sumo
from mn_wifi.link import wmediumd, mesh
from mn_wifi.wmediumdConnector import interference
from mn_wifi.node import Car
from scipy.special import gamma

########################################
# Basic Utility Functions (unchanged)  #
########################################

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))


def getdatetime():
    """Get current date/time in Asia/Jakarta timezone."""
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Jakarta"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME

def calculate_subnet_mask(number_of_nodes):
    """Calculate subnet mask based on the number of nodes."""
    number_of_ips_needed = number_of_nodes + 2
    bits_needed_for_hosts = math.ceil(math.log2(number_of_ips_needed))
    subnet_mask = 32 - bits_needed_for_hosts
    return subnet_mask

##############################################
# Example Car subclass for additional fields #
##############################################
class CustomCar(Car):
    def __init__(self, name, **params):
        super().__init__(name, **params)
        self.beacon_rate = 10
        self.power_transmission = 15
        self.vehicle_density = 1

########################################
# Data Logging and PPO Integration     #
########################################

def data_logging(
    net,
    duration=60,
    data_collection_interval=1,
    server_ip='127.0.0.1',
    server_port=9999
):
    """
    - Runs a simulation loop, logging car data at each step.
    - Sends the data to the PPO server (`PPO.py`) over a socket.
    - Receives updated beacon rate and power transmission from PPO.
    - Applies these new settings to each car in real time.
    """
    start_time = time.time()

    # Optionally dump captured data to CSV:
    csv_file = open(f"{os.getcwd()}/network_data_ppo.csv", mode='w', newline='')
    fieldnames = [
        'timestamp', 'car_id', 'position', 'speed', 'old_beacon_rate',
        'old_power_trans', 'new_beacon_rate', 'new_power_trans'
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Start the SUMO simulation loop
    while time.time() - start_time < duration:
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()

        for vehicle_id in vehicles:
            car_id_str = f'car{int(float(vehicle_id)) + 1}'
            try:
                car = net.getNodeByName(car_id_str)
            except KeyError:
                continue

            # Gather data from SUMO about position, speed
            x, y = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)

            # You might also track neighbors, channel busy ratio, etc.
            # Here we keep it simple:
            car.params['position'] = (x, y)
            car.params['speed'] = speed

            # Prepare data to send to PPO
            log_data = {
                "timestamp": getdatetime(),
                "car_id": car_id_str,
                "beacon_rate": car.beacon_rate,     # old beacon rate
                "power_transmission": car.power_transmission,  # old power
                "vehicle_density": 5.0  # or compute actual density if desired
            }

            # Send data to the PPO server and receive updated parameters
            new_beacon_rate, new_power_trans = send_and_receive_ppo(server_ip, server_port, log_data)

            # Apply new settings to the car (if valid)
            if new_beacon_rate is not None and new_power_trans is not None:
                # Update the car's internal variables
                old_beacon = car.beacon_rate
                old_power = car.power_transmission

                car.beacon_rate = new_beacon_rate
                car.power_transmission = new_power_trans

                # If you want to do something in Mininet-WiFi or SUMO:
                # For instance, if you had a param named "txpower", or if you want
                # to pass new beacon rates to TRACI, do so here:
                # e.g. traci.vehicle.setParameter(vehicle_id, "txpower", str(new_power_trans))

                # Log the changes
                csv_writer.writerow({
                    'timestamp': log_data['timestamp'],
                    'car_id': car_id_str,
                    'position': (x, y),
                    'speed': speed,
                    'old_beacon_rate': old_beacon,
                    'old_power_trans': old_power,
                    'new_beacon_rate': new_beacon_rate,
                    'new_power_trans': new_power_trans
                })

        time.sleep(data_collection_interval)

    csv_file.close()
    traci.close()
    net.stop()
    subprocess.run(["mn", "-c"])
    sys.exit(0)

def send_and_receive_ppo(server_ip, server_port, log_data):
    """
    Opens a socket connection to the PPO server,
    sends `log_data`, and retrieves the new beacon rate + power from the server.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        sock.sendall(json.dumps(log_data).encode('utf-8'))

        response = sock.recv(4096)
        if not response:
            return None, None

        decoded_resp = json.loads(response.decode('utf-8'))
        new_beacon_rate = decoded_resp.get("beacon_rate", None)
        new_power_trans = decoded_resp.get("power_transmission", None)

        return new_beacon_rate, new_power_trans

    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None
    finally:
        sock.close()

############################################
# Main Topology Setup (similar to mobility_Q)
############################################

def topology(
    num_cars=5,
    sumo_config_file="sumocfg/manhattangrid.sumocfg",
    duration=60,
    server_ip='127.0.0.1',
    server_port=9999
):
    """
    Creates a Mininet-WiFi network with `num_cars` vehicles,
    runs SUMO with `sumo_config_file`, and logs data for `duration` seconds.
    Sends data to the PPO server (PPO.py) at `server_ip:server_port`.
    """
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)

    # Create car nodes
    for i in range(num_cars):
        car = net.addCar(f'car{i+1}', cls=CustomCar, encrypt=['wpa2'])

    net.setPropagationModel(model="logDistance", exp=2.8)
    net.configureNodes()

    # Create mesh links (example: single interface per car)
#    for car in net.cars:
#       net.addLink(
#            car,
#            intf=car.wintfs[0].name,
#            mode='ac',
#            cls=mesh,
#            ssid='mesh-ssid',
#            channel=38,
#            freq=5180
#        )

    # Build network
    net.build()
    time.sleep(5)

    # Assign IP addresses
    subnet_mask = calculate_subnet_mask(num_cars)
    base_ip = 192 * 256**3 + 168 * 256**2
    for idx, car in enumerate(net.cars):
        ip = base_ip + idx + 1
        ip_str = (
            f'{(ip // (256**3)) % 256}.' +
            f'{(ip // (256**2)) % 256}.' +
            f'{(ip // 256) % 256}.' +
            f'{ip % 256}'
        )
        car.setIP(f'{ip_str}/{subnet_mask}', intf=f'{car.wintfs[0].name}')
        print(f"[INIT] {car.name} -> {ip_str}/{subnet_mask}")

    # Start SUMO
    sumoCmd = ["sumo-gui", "-c", sumo_config_file, "--start", "--delay", "200"]
    sumo_process = subprocess.Popen(sumoCmd)
    time.sleep(2)
    traci.start(["sumo", "-c", sumo_config_file], port=8813)
    traci.setOrder(1)

    # Start data logging loop
    data_logging(
        net,
        duration=duration,
        server_ip=server_ip,
        server_port=server_port
    )

    # Cleanup
    sumo_process.terminate()
    CLI(net)
    net.stop()

if __name__ == "__main__":
    # Example usage:
    topology(
        num_cars=5,
        sumo_config_file="sumocfg/manhattangrid.sumocfg",
        duration=60,
        server_ip='127.0.0.1',
        server_port=9999
    )
