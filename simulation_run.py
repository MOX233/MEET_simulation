
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import csv
from typing import DefaultDict

from utils.read_tripInfo import read_tripInfo
from utils.interface_for_FL import generate_FLtable_from_tripInfo
from utils.options import args_parser
from utils.FL_training import FL_training

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def generate_routefile(args):
    random.seed(args.seed)  # make tests reproducible
    
    num_steps = args.num_steps  # number of time steps
    Lambda = args.Lambda       # arrival rate of car flow
    accel = args.accel         # accelerate of car flow
    decel = args.decel         # decelerate of car flow
    sigma = args.sigma         # imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection
    carLength = args.carLength # length of cars
    minGap = args.minGap       # minimum interval between adjacent cars
    maxSpeed = args.maxSpeed   # maxSpeed for cars
    speedFactoer_mean = args.speedFactoer_mean 
    speedFactoer_dev = args.speedFactoer_dev
    speedFactoer_min = args.speedFactoer_min
    speedFactoer_max = args.speedFactoer_max
    
    speedFactoer = "normc({mean}, {dev}, {min}, {max})".format(**{
        "mean":speedFactoer_mean,
        "dev":speedFactoer_dev,
        "min":speedFactoer_min,
        "max":speedFactoer_max,
    }) # can be given as "norm(mean, dev)" or "normc(mean, dev, min, max)"


    with open("sumo_data/road.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typecar" accel="{accel}" decel="{decel}" sigma="{sigma}" length="{carLength}" minGap="{minGap}" maxSpeed="{maxSpeed}" speedFactoer="{speedFactoer}" guiShape="passenger"/>
        <route id="right" edges="right1 right12 right2" />
        <route id="left" edges="left2 left21 left1" />""".format(**{
            "accel":accel,
            "decel":decel,
            "sigma":sigma,
            "carLength":carLength,
            "minGap":minGap,
            "maxSpeed":maxSpeed,
            "speedFactoer":speedFactoer,
        }), file=routes)
        vehNr = 0
        for i in range(num_steps):
            # just right traffic
            if random.uniform(0, 1) < Lambda:
                print('    <vehicle id="car_%i" type="typecar" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)


def sumo_run(options):
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    if args.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile(args)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "sumo_data/road.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             '--step-length', str(args.step_length),])

    """execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()
    sys.stdout.flush()

def sumo_run_with_trajectoryInfo(options):
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    if args.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile(args)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "sumo_data/road.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             '--step-length', str(args.step_length),])

    """execute the TraCI control loop"""
    step = 0
    w = csv.writer(open(options.trajectoryInfo_path, 'w',newline=""))
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        for veh_i, veh in enumerate(traci.vehicle.getIDList()):
            (x, y),speed, = [f(veh) for f in [
                traci.vehicle.getPosition,
                traci.vehicle.getSpeed, #Returns the speed of the named vehicle within the last step [m/s]; error value: -1001

            ]]
            w.writerow([step,veh,veh_i,x,speed])
    traci.close()
    sys.stdout.flush()

# this is the main entry point of this script
if __name__ == "__main__":
    args = args_parser()
    #sumo_run(args)
    car_tripinfo = read_tripInfo(tripInfo_path='tripinfo.xml')
    FL_table = generate_FLtable_from_tripInfo(args)

    args.lidar_training_data='/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/lidar_input/lidar_train.npz'
    args.beam_training_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/beam_output/beams_output_train.npz'
    args.lidar_validation_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/lidar_input/lidar_validation.npz'
    args.beam_validation_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/beam_output/beams_output_validation.npz'
    args.model_path = './save/model.pt'

    args.local_iter = int(args.local_train_speed * args.local_train_time)
    FL_training(args,FL_table,car_tripinfo)