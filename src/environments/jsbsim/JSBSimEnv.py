import numpy as np
import jsbsim
import os
import time
import math
import random

class Env():

    def __init__(self, flightOrigin, flightDestination, n_acts, usePredefinedSeeds, dictObservation, dictRotation, speed, pause, qID, render, realTime, randomDesiredState, stateDepth, startingPitchRange, startingRollRange, desiredPitchRange, desiredRollRange):
        self.flightOrigin = flightOrigin
        self.flightDestination = flightDestination
        self.previousPosition = flightOrigin
        self.startingOrientation = []
        self.n_actions = n_acts
        self.dictObservation = dictObservation
        self.dictRotation = dictRotation
        self.startingVelocity = speed
        self.pauseDelay = pause
        self.stateList = []
        self.qID = qID
        self.fsToMs = 0.3048  # convertion from feet per sec to meter per sec
        self.msToFs = 3.28084  # convertion from meter per sec to feet per sec
        self.radToDeg = 57.2957795  # convertion from radiants to degree
        self.degToRad = 0.0174533  # convertion from deg to rad
        self.realTime = realTime
        self.id = "JSBSim"
        self.stateDepth = stateDepth
        self.randomDesiredState = randomDesiredState
        
        self.desiredPosition = {
            "lat_value": flightDestination[0],
            "lon_value": flightDestination[1],
            "alt_value": flightDestination[2],
        }
        if(usePredefinedSeeds):
            np.random.seed(42)
        self.startingPitchRange = startingPitchRange
        self.startingRollRange = startingRollRange
        self.desiredPitchRange = desiredPitchRange
        self.desiredRollRange = desiredRollRange
        os.environ["JSBSIM_DEBUG"] = str(0)  # set this before creating fdm to stop debug print outs
        self.fdm = jsbsim.FGFDMExec('./src/environments/jsbsim/jsbsim/', None)  # declaring the sim and setting the path
        self.physicsPerSec = int(1 / self.fdm.get_delta_t())  # default by jsb. Each physics step is a 120th of 1 sec
        self.realTimeDelay = self.fdm.get_delta_t()
        self.fdm.load_model('c172r')  # loading cassna 172
        if render:  # only when render is True
            # Open Flight gear and enter: --fdm=null --native-fdm=socket,in,60,localhost,5550,udp --aircraft=c172r --airport=RKJJ
            self.fdm.set_output_directive('./data_output/flightgear.xml')  # loads xml that initates udp transfer
        self.fdm.run_ic()  # init the sim
        self.fdm.print_simulation_configuration()

    def init_change_body(self, resetPosition):

        self.fdm["ic/lat-gc-deg"] = self.flightOrigin[self.dictObservation["lat"]]  # Latitude initial condition in degrees
        self.fdm["ic/long-gc-deg"] = self.flightOrigin[self.dictObservation["long"]]  # Longitude initial condition in degrees
        self.fdm["ic/h-sl-ft"] = self.flightOrigin[self.dictObservation["alt"]] * self.msToFs  # Height above sea level initial condition in feet

        self.fdm["ic/theta-deg"] = resetPosition[self.dictRotation["pitch"]]  # Pitch angle initial condition in degrees
        self.fdm["ic/phi-deg"] = resetPosition[self.dictRotation["roll"]]  # Roll angle initial condition in degrees
        self.fdm["ic/psi-true-deg"] = resetPosition[self.dictRotation["yaw"]]  # Heading angle initial condition in degrees

        self.fdm["ic/ve-fps"] = resetPosition[self.dictRotation["eastVelo"]] * self.msToFs  # Local frame y-axis (east) velocity initial condition in feet/second
        self.fdm["ic/vd-fps"] = -resetPosition[self.dictRotation["verticalVelo"]] * self.msToFs  # Local frame z-axis (down) velocity initial condition in feet/second
        self.fdm["ic/vn-fps"] = -resetPosition[self.dictRotation["northVelo"]] * self.msToFs  # Local frame x-axis (north) velocity initial condition in feet/second
        self.fdm["propulsion/refuel"] = True  # refules the plane?
        self.fdm["propulsion/active_engine"] = True  # starts the engine?
        self.fdm["propulsion/set-running"] = 0  # starts the engine?

        self.fdm["ic/q-rad_sec"] = 0  # Pitch rate initial condition in radians/second
        self.fdm["ic/p-rad_sec"] = 0  # Roll rate initial condition in radians/second
        self.fdm["ic/r-rad_sec"] = 0  # Yaw rate initial condition in radians/second
        self.fdm.run_ic()
        # client.sendDREF("sim/flightmodel/position/local_ax", 0)  # The acceleration in local OGL coordinates +ax=E -ax=W
        # client.sendDREF("sim/flightmodel/position/local_ay", 0)  # The acceleration in local OGL coordinates +=Vertical (up)
        # client.sendDREF("sim/flightmodel/position/local_az", 0)  # The acceleration in local OGL coordinates -az=S +az=N

    def send_body_Ctrl(self, ctrl):
        '''
        ctrl[0]: + Stick in (elevator pointing down) / - Stick back (elevator pointing up)
        ctrl[1]: + Stick right (right aileron up) / - Stick left (left aileron up)
        ctrl[2]: + Peddal (Rudder) left / - Peddal (Rudder) right
        '''
        self.fdm["fcs/elevator-cmd-norm"] = ctrl[0]  # Elevator control (stick in/out)?
        self.fdm["fcs/aileron-cmd-norm"] = -ctrl[1]  # Aileron control (stick left/right)? might need to switch
        self.fdm["fcs/rudder-cmd-norm"] = ctrl[2]  # Rudder control (peddals)
        self.fdm["fcs/throttle-cmd-norm"] = ctrl[3]  # throttle

    def get_body_Posi(self):
        lat = self.fdm["position/lat-gc-deg"]  # Latitude
        long = self.fdm["position/long-gc-deg"]  # Longitude
        alt = self.fdm["position/h-sl-ft"]  * self.fsToMs

        pitch = self.fdm["attitude/theta-deg"]  # pitch
        roll = self.fdm["attitude/phi-deg"]  # roll
        heading = self.fdm["attitude/psi-deg"]  # yaw

        return [lat, long, alt, pitch, roll, heading]
 
    def rewardFunction(self):
        '''
        flightOrigin, flightDestinaion: 0->lat, 1->lon, 2->alt 
        position: current plane [lat, long, alt, pitch, roll, heading]
        '''
        alpha = self.fdm["aero/alpha-deg"]
    
        # Normalization delta alttitude value
        
        delta_roll = self.delta_roll / 180
        delta_pitch = self.delta_pitch / 180
        delta_yaw = self.delta_yaw / 180
        
        # Calcuate reward function value

        reward = pow(float((3 - (delta_roll + delta_pitch + delta_yaw)) / 3), 2)

        if(self.delta_roll > 40 or self.delta_pitch > 40):
            reward = reward * 0.1
        elif(self.delta_roll > 20 or self.delta_pitch > 20):
            reward = reward * 0.25
        elif(self.delta_roll > 10 or self.delta_pitch > 10):
            reward = reward * 0.5
        elif(self.delta_roll > 5 or self.delta_pitch > 5):
            reward = reward * 0.75
        elif(self.delta_roll > 1 or self.delta_pitch > 1):
            reward = reward * 0.9

        # if(position[self.dictObservation["alt"]] <= 1000):
        #     reward = reward * 0.1

        done = False
        # if (alpha >= 16 or position[self.dictObservation["alt"]] <= 0):
        #     done = True
        #     reward = -1
            
        if(self.distance < 20):
            reward = reward * 1.5
            done = True
        
        return reward, done
    
    def get_model_input(self, position):
        observation = position.copy()
        ori_lat, ori_lon = math.radians(self.flightOrigin[0]), math.radians(self.flightOrigin[1]) 
        des_lat, des_lon = math.radians(self.flightDestination[0]), math.radians(self.flightDestination[1])
        cur_lat, cur_lon = math.radians(observation[self.dictObservation["lat"]]), math.radians(observation[self.dictObservation["long"]])
        
        # Position convert to meter X=2*sin(alpha/2)*R
        earth_radius = 6378737
        origin_x = 2 * math.sin(ori_lat / 2) * earth_radius
        origin_y = 2 * math.sin(ori_lon / 2) * earth_radius
        desired_x = 2 * math.sin(des_lat / 2) * earth_radius
        desired_y = 2 * math.sin(des_lon / 2) * earth_radius
        current_x = 2 * math.sin(cur_lat / 2) * earth_radius
        current_y = 2 * math.sin(cur_lon / 2) * earth_radius
        
        # Calculate Vertical distance and Horizontal distance
        
        self.d_vertical = self.flightDestination[2] - observation[self.dictObservation["alt"]]
        self.d_horizontal = abs((desired_y - origin_y) * current_x - (desired_x - origin_x) * current_y + desired_x * origin_y - desired_y * origin_x) / math.sqrt((desired_y - origin_y) ** 2 + (desired_x - origin_x) ** 2)
        self.distance = math.sqrt((desired_x - current_x) ** 2 + (desired_y - current_y) ** 2 + self.d_vertical  ** 2)
        
        # Calculate delta roll, pitch, yaw
        
        desired_roll = 0
        desired_pitch = math.atan(self.d_vertical / 10) / 3.141592654 * 180
        desired_yaw = math.atan(self.d_horizontal / 10) / 3.141592654 * 180
        
        self.delta_roll = desired_roll - observation[self.dictObservation["roll"]]
        self.delta_pitch = desired_pitch - observation[self.dictObservation["pitch"]]
        self.delta_yaw = desired_yaw - observation[self.dictObservation["yaw"]]
        # input : lat, long, alt, pitch, roll, heading
        # output : lat, lon, alt, pitch, roll, yaw, P, Q, R, Vx, Vy, Vz
        observation[self.dictObservation["lat"]] = observation[self.dictObservation["lat"]] - self.desiredPosition["lat_value"]
        observation[self.dictObservation["long"]] = observation[self.dictObservation["long"]] - self.desiredPosition["lon_value"]
        observation[self.dictObservation["alt"]] = observation[self.dictObservation["alt"]] - self.desiredPosition["alt_value"]
        observation[self.dictObservation["pitch"]] = self.delta_pitch
        observation[self.dictObservation["roll"]] = self.delta_roll
        observation[self.dictObservation["yaw"]] = self.delta_yaw
        P = self.fdm["velocities/p-rad_sec"] * self.radToDeg  # The roll rotation rates
        Q = self.fdm["velocities/q-rad_sec"] * self.radToDeg  # The pitch rotation rates
        R = self.fdm["velocities/r-rad_sec"] * self.radToDeg  # The yaw rotation rates
        Vx = self.fdm["velocities/u-aero-fps"]  # Velocity in the x direction
        Vy = self.fdm["velocities/v-aero-fps"]  # Velocity in the y direction
        Vz = self.fdm["velocities/w-aero-fps"]    
        velocities = [P, Q, R, Vx, Vy, Vz]
        state = tuple(observation) + tuple(velocities)
        self.stateList.append(state)
        if(len(self.stateList) > self.stateDepth):
            self.stateList.pop(0)
        state = self.stateList

        return state
    
    def step(self, action):
        '''
        pitch: + plane nose up/ - plane nose down 
        roll: + right wing down/ - left wing down
        yaw: + right clock/ - left clock
        '''
        position = self.get_body_Posi()
        ctrl = [0, 0, 0, 0.5]  # throttle set to .5 by default
        # translate the action value to the observation space value
        if action < 3:
            actionDimension = action + 3
            angle = position[actionDimension]
            if action == 2:
                angle = self.delta_yaw
            if action == 0:
                angle = self.delta_pitch
            sign = (angle > 0) - (angle < 0)

            if angle < -180 or angle > 180:
                ctrl[action] = 1 * sign
            elif -180 <= angle < -50 or 50 <= angle < 180:
                ctrl[action] = 0.75 * sign
            elif -50 <= angle < -25 or 25 <= angle < 50:
                ctrl[action] = 0.66 * sign
            elif -25 <= angle < -15 or 15 <= angle < 25:
                ctrl[action] = 0.5 * sign
            elif -15 <= angle < -10 or 10 <= angle < 15:
                ctrl[action] = 0.33 * sign
            elif -10 <= angle < -5 or 5 <= angle < 10:
                ctrl[action] = 0.25 * sign
            elif -5 <= angle < -2 or 2 <= angle < 5:
                ctrl[action] = 0.1 * sign
            elif -2 <= angle < -1 or 1 <= angle < 2:
                ctrl[action] = 0.05 * sign
            elif -1 <= angle < 0 or 0 <= angle < 1:
                ctrl[action] = 0.025 * sign
            else:
                print("DEBUG - should not get here")
        elif action == 3:
            ctrl = [0, 0, 0, 0.8]

        actions_binary = np.zeros(self.n_actions, dtype=int)
        actions_binary[action] = 1

        self.send_body_Ctrl(ctrl)
        for i in range(int(self.pauseDelay * self.physicsPerSec)):  # will mean that the input will be applied for pauseDelay seconds
            # If realTime is True, then the sim will slow down to real time, should only be used for viewing/debugging, not for training
            if(self.realTime):
                self.send_body_Ctrl(ctrl)
                self.fdm.run()
                time.sleep(self.realTimeDelay)
            # Non realTime code: this is default
            else:
                self.send_body_Ctrl(ctrl)
                self.fdm.run()

        position = self.get_body_Posi()
        state = self.get_model_input(position)

        done = False
        reward = 0
        reward, done = self.rewardFunction()

        info = [position, actions_binary, ctrl]

        return state, reward, done, info

    def reset(self):
        startingPitch = int(random.randrange(-self.startingPitchRange, self.startingPitchRange))
        startingRoll = int(random.randrange(-self.startingRollRange, self.startingRollRange))
        startingYaw = int(random.randrange(0, 360))

        angleRadPitch = math.radians(startingPitch)
        verticalVelocity = self.startingVelocity * math.sin(angleRadPitch)
        forwardVelocity = self.startingVelocity * math.cos(angleRadPitch)

        angleRadYaw = math.radians(startingYaw)
        eastVelocity = forwardVelocity * math.sin(angleRadYaw)
        northVelocity = - forwardVelocity * math.cos(angleRadYaw)

        self.startingOrientation = [startingRoll, startingPitch, startingYaw, northVelocity, eastVelocity, verticalVelocity]
        self.init_change_body(self.startingOrientation)
        
        self.stateList = []
        self.send_body_Ctrl([0, 0, 0, 0, 0, 0, 1])  # this means it will not control the stick during the reset
        new_posi = self.get_body_Posi()
        
        state = self.get_model_input(new_posi)

        return state
