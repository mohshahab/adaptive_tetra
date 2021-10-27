#!/usr/bin/env python

# ROS python API
import rospy
# Joy message structure
from sensor_msgs.msg import *
# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

import time

import numpy

#from std_msgs.msg import Float32MultiArray

# Flight modes class
# Flight modes are activated using ROS services
class fcuModes:
    def __init__(self):
        pass


    def setArm1(self):
        rospy.wait_for_service('tetra1/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra1/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            arm1 = armService(True)
	    print "Service arming call: %s"% arm1.success
        except rospy.ServiceException, e:
            print "Service arming call failed: %s"%e

    def setDisarm1(self):
        rospy.wait_for_service('tetra1/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra1/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s"%e

    def setOffboardMode1(self):
        rospy.wait_for_service('tetra1/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra1/mavros/set_mode', mavros_msgs.srv.SetMode)
#            flightModeService(custom_mode='OFFBOARD')
            success = flightModeService(custom_mode='OFFBOARD')
            print success
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set."%e

    def setAutoLandMode1(self):
        rospy.wait_for_service('tetra1/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra1/mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Autoland Mode could not be set."%e




    def setArm2(self):
        rospy.wait_for_service('tetra2/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra2/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            arm1 = armService(True)
	    print "Service arming call: %s"% arm1.success
        except rospy.ServiceException, e:
            print "Service arming call failed: %s"%e

    def setDisarm2(self):
        rospy.wait_for_service('tetra2/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra2/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s"%e

    def setOffboardMode2(self):
        rospy.wait_for_service('tetra2/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra2/mavros/set_mode', mavros_msgs.srv.SetMode)
#            flightModeService(custom_mode='OFFBOARD')
            success = flightModeService(custom_mode='OFFBOARD')
            print success
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set."%e

    def setAutoLandMode2(self):
        rospy.wait_for_service('tetra2/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra2/mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Autoland Mode could not be set."%e



    def setArm3(self):
        rospy.wait_for_service('tetra3/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra3/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            arm1 = armService(True)
	    print "Service arming call: %s"% arm1.success
        except rospy.ServiceException, e:
            print "Service arming call failed: %s"%e

    def setDisarm3(self):
        rospy.wait_for_service('tetra3/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra3/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s"%e

    def setOffboardMode3(self):
        rospy.wait_for_service('tetra3/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra3/mavros/set_mode', mavros_msgs.srv.SetMode)
#            flightModeService(custom_mode='OFFBOARD')
            success = flightModeService(custom_mode='OFFBOARD')
            print success
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set."%e

    def setAutoLandMode3(self):
        rospy.wait_for_service('tetra3/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra3/mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Autoland Mode could not be set."%e


    def setArm4(self):
        rospy.wait_for_service('tetra4/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra4/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            arm1 = armService(True)
	    print "Service arming call: %s"% arm1.success
        except rospy.ServiceException, e:
            print "Service arming call failed: %s"%e

    def setDisarm4(self):
        rospy.wait_for_service('tetra4/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra4/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s"%e

    def setOffboardMode4(self):
        rospy.wait_for_service('tetra4/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra4/mavros/set_mode', mavros_msgs.srv.SetMode)
#            flightModeService(custom_mode='OFFBOARD')
            success = flightModeService(custom_mode='OFFBOARD')
            print success
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set."%e

    def setAutoLandMode4(self):
        rospy.wait_for_service('tetra4/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra4/mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Autoland Mode could not be set."%e

# Main class: Converts joystick commands to position setpoints
class Controller:
    # initialization method
    def __init__(self):
        # Drone state
#        self.state = State()
        # Instantiate a setpoints message
#        self.sp         = PositionTarget()
	self.sp         = PoseStamped()
	self.mc = PoseStamped()
        # set the flag to use position setpoints and yaw angle
#        self.sp.type_mask    = int('010111111000', 2)
        # LOCAL_NED
#        self.sp.coordinate_frame= 1

        # We will fly at a fixed altitude for now
        # Altitude setpoint, [meters]
        self.ALT_SP        = 1.0
        # update the setpoint message with the required altitude
#        self.sp.position.z    = self.ALT_SP
	self.sp.pose.position.z    = self.ALT_SP
	#self.sp.pose.orientation.z    = 0
	#self.sp.pose.orientation.w    = 1

        # Instantiate a joystick message
        # self.joy_msg        = Joy()
        # initialize
        # self.joy_msg.axes=[0.0, 0.0, 0.0]

        # Step size for position update
        #self.STEP_SIZE = 2.0

        # Fence. We will assume a square fence for now
        #self.FENCE_LIMIT = 50.0

        # A Message for the current local position of the drone
        self.local_pos = Point(0.0, 0.0, 0.0)
        self.mocap_pos = PoseStamped()
	self.local_yaw = 0

	#self.modes = fcuModes()

	#self.kk = 0
	#self.kk = 10.0

	self.hover = 0
	self.hover2 = 0
	self.hover3 = 0
	self.hovering_time = 0
	self.land = 0
	self.initial = 1

    # Callbacks ##

    ## local position callback
    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z
	self.local_yaw = msg.pose.orientation.w

    def mocapCb(self, msg):
        self.mocap_pos.pose.position.x = msg.pose.position.x
        self.mocap_pos.pose.position.y = msg.pose.position.y
        self.mocap_pos.pose.position.z = msg.pose.position.z

        self.mocap_pos.pose.orientation.x = msg.pose.orientation.x
        self.mocap_pos.pose.orientation.y = msg.pose.orientation.y
        self.mocap_pos.pose.orientation.z = msg.pose.orientation.z
        self.mocap_pos.pose.orientation.w = msg.pose.orientation.w

        self.mocap_pos.header.seq = msg.header.seq
        self.mocap_pos.header.stamp = msg.header.stamp
        self.mocap_pos.header.frame_id = msg.header.frame_id







#        print(self.local_pos.x)

    ## joystick callback
#    def joyCb(self, msg):
#        self.joy_msg = msg

    ## Drone State callback
#    def stateCb(self, msg):
#        self.state = msg

    ###################
    ## End Callbacks ##
    ###################


    ## Update setpoint message
#    def updateSp(self):
        #x = 1.0*self.joy_msg.axes[1]
        #y = 1.0*self.joy_msg.axes[0]

	#self.sp.position.x = self.local_pos.x + self.STEP_SIZE*x
	#self.sp.position.y = self.local_pos.y + self.STEP_SIZE*y
    

#        self.sp.position.x = 0
#        self.sp.position.y = 0
#	self.sp.header.frame_id = "8"
#	self.sp.header.seq = 1

#	if self.xflag == 0:
#		if self.local_pos.z > 4.5:
#			self.xflag = 1
#			self.hovering_time = time.time()







#		self.hovering_time = time.time()


		

# Main function
def main():


    # initiate node
    rospy.init_node('setpoint_multi_node', anonymous=True)

    # flight mode object
    modes = fcuModes()
    # controller object
    cnt = Controller()

    # ROS loop rate, [Hz]
    rate = rospy.Rate(100.0)

    # Subscribe to drone state
#    rospy.Subscriber('tetra4/mavros/state', State, cnt.stateCb)


    ## TO DO - create a subscriber to drone's local position which calls cnt.posCb callback function
#    rospy.Subscriber('tetra1/mavros/local_position/pose', PoseStamped, cnt.posCb1, queue_size=1)
#    rospy.Subscriber('tetra2/mavros/local_position/pose', PoseStamped, cnt.posCb2, queue_size=1)
#    rospy.Subscriber('tetra3/mavros/local_position/pose', PoseStamped, cnt.posCb3, queue_size=1)
    rospy.Subscriber('tetra2/mavros/local_position/pose', PoseStamped, cnt.posCb, queue_size=1)

    rospy.Subscriber('/vrpn_client_node/tetra1g/pose', PoseStamped, cnt.mocapCb, queue_size=1)

    ## TO DO - create a subscriber to joystick topic which call cnt.joyCb callback function
    

    ## TO DO - create a publisher with name "sp_pub" which will publish the position setpoints
    
#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_raw/local', PoseStamped, queue_size=10)

#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_position/global', PoseStamped, queue_size=10)

    mocap_pub1 = rospy.Publisher('tetra5/mavros/vision_pose/pose', PoseStamped, queue_size=1)
    mocap_pub2 = rospy.Publisher('tetra5/mavros/vision_pose/pose', PoseStamped, queue_size=1)
    mocap_pub3 = rospy.Publisher('tetra5/mavros/vision_pose/pose', PoseStamped, queue_size=1)
    mocap_pub4 = rospy.Publisher('tetra5/mavros/vision_pose/pose', PoseStamped, queue_size=1)


    sp_pub1 = rospy.Publisher('tetra1/mavros/setpoint_position/local', PoseStamped, queue_size=1)
    sp_pub2 = rospy.Publisher('tetra2/mavros/setpoint_position/local', PoseStamped, queue_size=1)
    sp_pub3 = rospy.Publisher('tetra3/mavros/setpoint_position/local', PoseStamped, queue_size=1)
    sp_pub4 = rospy.Publisher('tetra4/mavros/setpoint_position/local', PoseStamped, queue_size=1)

#    sp_pub2 = rospy.Publisher('tetra2/mavros/setpoint_raw/local', PositionTarget, queue_size=1)



#    print(cnt.local_pos.x)



# Make sure the drone is armed
    #m=cnt.mm
#    if not cnt.state.armed:
#	for i in range(100):
##		cnt.sp.header.stamp = rospy.Time.now()
#       		sp_pub1.publish(cnt.sp)
#       		sp_pub2.publish(cnt.sp)
#       		sp_pub3.publish(cnt.sp)
#       		sp_pub4.publish(cnt.sp)
#       		rate.sleep()
#       		#m = m+1
#	#else:
#    modes.setArm2()
#       	rate.sleep()


#    kk=0
#    while kk<100:
#     	kk=kk+1
    









   
    mm=0
    while mm<50:
#	cnt.sp.header.stamp = rospy.Time.now()
	cnt.mocap_pos1=cnt.mocap_pos
	cnt.mocap_pos2=cnt.mocap_pos
	cnt.mocap_pos3=cnt.mocap_pos
	cnt.mocap_pos4=cnt.mocap_pos
	mocap_pub1.publish(cnt.mocap_pos1)
	mocap_pub2.publish(cnt.mocap_pos2)
	mocap_pub3.publish(cnt.mocap_pos3)
	mocap_pub4.publish(cnt.mocap_pos4)
        sp_pub1.publish(cnt.sp)
        sp_pub2.publish(cnt.sp)
        sp_pub3.publish(cnt.sp)
        sp_pub4.publish(cnt.sp)

        rate.sleep()
        mm = mm+1




    if cnt.initial:
    	cnt.sp.pose.position.y = cnt.local_pos.y
#    	cnt.sp.pose.position.x = 1.0
	cnt.x_pos = cnt.local_pos.x
	cnt.sp.pose.position.x = cnt.x_pos
  	cnt.sp.pose.orientation.w    = cnt.local_yaw
	cnt.initial = 0

 # activate OFFBOARD mode
#    modes.setOffboardMode1()
#    modes.setOffboardMode2()
#    modes.setOffboardMode3()
#    modes.setOffboardMode4()
    #cnt.set_mode("OFFBOARD", 5)
#    print "offboard enabled"
#    cnt.hovering_time = time.time()
#    cnt.xflag = 0



#    print(cnt.local_pos.x)



    # ROS main loop
    while not rospy.is_shutdown():
    # We need to send few setpoint messages, then activate OFFBOARD mode, to take effect
#	cnt.sp.header.stamp = rospy.Time.now()
#    	k=0
#	while k<50:
##		cnt.sp.header.stamp = rospy.Time.now()
#        	sp_pub.publish(cnt.sp)
#        	rate.sleep()
#        	k = k+1

#   	cnt.sp.header.stamp = rospy.Time.now()
#        print "here1"

#	print(cnt.mocap_pos)


        cnt.mocap_pos1=cnt.mocap_pos
        cnt.mocap_pos2=cnt.mocap_pos
        cnt.mocap_pos3=cnt.mocap_pos
        cnt.mocap_pos4=cnt.mocap_pos










#        cnt.updateSp()
	mocap_pub1.publish(cnt.mocap_pos1)
	mocap_pub2.publish(cnt.mocap_pos2)
	mocap_pub3.publish(cnt.mocap_pos3)
	mocap_pub4.publish(cnt.mocap_pos4)
        sp_pub1.publish(cnt.sp)
        sp_pub2.publish(cnt.sp)
        sp_pub3.publish(cnt.sp)
        sp_pub4.publish(cnt.sp)

        rate.sleep()
	if not cnt.hover:
	    	if cnt.local_pos.z >= cnt.ALT_SP - 0.1:
			print "here we go 1"
			cnt.hover = 1
			cnt.hovering_time = time.time()


#	d_pos = 1.0

#	if cnt.hover:
#		if time.time() - cnt.hovering_time > 5:
##			cnt.land =1
#			cnt.sp.pose.position.x = cnt.x_pos + d_pos

#	if not cnt.hover2:
#	    	if abs(cnt.local_pos.x - cnt.x_pos - d_pos) <= 0.1 :
#			print "here we go 2"
#			cnt.hover2 = 1
#			cnt.hovering_time = time.time()	
#	if not cnt.hover3:
#		if cnt.hover2:
#			if time.time() - cnt.hovering_time > 5:
##			cnt.land =1
#				cnt.sp.pose.position.y = cnt.y_pos


##		if cnt.hover2:
#		    	if abs(cnt.local_pos.y - cnt.y_pos) <= 0.1 :
#				print "here we go 3"
#				cnt.hover3 = 1
#				cnt.hovering_time = time.time()			

	if cnt.hover:
		if time.time() - cnt.hovering_time > 5:
			cnt.land =1




	if cnt.land:
		print "LANDING !"
		modes.setAutoLandMode1()
		modes.setAutoLandMode2()
		modes.setAutoLandMode3()
		modes.setAutoLandMode4()

	mocap_pub1.publish(cnt.mocap_pos1)
	mocap_pub2.publish(cnt.mocap_pos2)
	mocap_pub3.publish(cnt.mocap_pos3)
	mocap_pub4.publish(cnt.mocap_pos4)
        sp_pub1.publish(cnt.sp)
        sp_pub2.publish(cnt.sp)
        sp_pub3.publish(cnt.sp)
        sp_pub4.publish(cnt.sp)

        rate.sleep()




if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
