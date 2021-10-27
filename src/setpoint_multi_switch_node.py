#!/usr/bin/env python

# ROS python API
import rospy
# Joy message structure
from sensor_msgs.msg import *
# 3D point & Stamped Pose msgs
from geometry_msgs.msg import *
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import *

import time

import math

from itertools import combinations
from itertools import permutations

import numpy as np

from tf.transformations import euler_from_quaternion, quaternion_from_euler

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

   
        self.ang1 = Point(0.0, 0.0, 0.0)
        self.om1 = Point(0.0, 0.0, 0.0)
        self.ve1 = Point(0.0, 0.0, 0.0)
        self.acc1 = Point(0.0, 0.0, 0.0)


        self.ang2 = Point(0.0, 0.0, 0.0)
        self.om2 = Point(0.0, 0.0, 0.0)
        self.ve2 = Point(0.0, 0.0, 0.0)
        self.acc2 = Point(0.0, 0.0, 0.0)


        self.ang3 = Point(0.0, 0.0, 0.0)
        self.om3 = Point(0.0, 0.0, 0.0)
        self.ve3 = Point(0.0, 0.0, 0.0)
        self.acc3 = Point(0.0, 0.0, 0.0)


        self.ang4 = Point(0.0, 0.0, 0.0)
        self.om4 = Point(0.0, 0.0, 0.0)
        self.ve4 = Point(0.0, 0.0, 0.0)
        self.acc4 = Point(0.0, 0.0, 0.0)


	#self.modes = fcuModes()

	#self.kk = 0
	#self.kk = 10.0

	self.hover = 0
	self.hover2 = 0
	self.hover3 = 0
	self.hovering_time = 0
	self.land = 0
	self.initial = 1



    def RotMat(self, X):
	ax=X.x
	ay=X.y
	az=X.z

	R = [[math.cos(az)*math.cos(ay)-math.sin(az)*math.sin(ay)*math.sin(ax), -math.sin(az)*math.cos(ax), math.cos(az)*math.sin(ay)+math.cos(ay)*math.sin(ay)*math.sin(az)], [math.sin(az)*math.cos(ay)+math.cos(az)*math.sin(ay)*math.sin(ax), math.cos(az)*math.cos(ax), math.sin(az)*math.sin(ay)-math.cos(az)*math.sin(ax)*math.cos(ay)], [-math.cos(ax)*math.sin(ay), math.sin(ay), math.cos(ay)*math.cos(ax)]]

	return R

    def h_f(self, p, v, th, w, pbj):
	Ro=self.RotMat(th)

	bM = np.dot(Ro,pbj)

	bMx=np.array(bM)
	wx=np.array(w)	
	px=np.array(p)	

	xm=px + bM 
	vbx=np.array(np.cross(wx.T, bMx.T))
	vm=np.array(v) + vbx.T

	return xm, vm

    def hinv_f(self, pj, vj, thj, wj, pbs):
	Ro=self.RotMat(thj)

	bM = np.dot(Ro,pbs)

	bMx=np.array(bM)
	wx=np.array(wj)	
	px=np.array(pj)	

	xjm=px - bM 
	vbx=np.array(np.cross(wx.T, bMx.T))
	vjm=np.array(vj) - vbx.T

	return xjm, vjm




    def switching(self, J, II, It, swi):
	Ithat=[]
	for i in II:
		if J[i-1]< J[swi-1]:
			Ithat_temp = np.append(Ithat,i)
			Ithat = Ithat_temp

#	It = It.intersection(Ithat)
	It = list(map(int, set(It) & set(Ithat)))
#	print("here:", It)
	if not It:
		It=II

#	JJJ=np.append([1, 1],J[23-1])
#	print(JJJ[2])
	
	JJ=[]
	for i in It:
		JJ_temp = np.append(JJ,J[i-1])
		JJ = JJ_temp

#	print(JJ)

	ij = np.argmin(JJ)

	swi=It[ij]

	return swi, It
	
    def SWprint(self, sw):
	swP=[0, 0, 0, 0]
	for i in list(range(1,5)):
		if sw[i-1]==1:
			swP[i-1]="Top"
		if sw[i-1]==2:
			swP[i-1]="Front"
		if sw[i-1]==3:
			swP[i-1]="Right"
		if sw[i-1]==4:
			swP[i-1]="Left"
	
	return swP

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


    ##---------------sensor callbacks----------------------------------




    def imuCb1(self, msg):

        #global roll, pitch, yaw
        self.orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion (self.orientation_list)

	self.ang1.x = roll
	self.ang1.y = pitch
	self.ang1.z = yaw

        self.om1.x = msg.angular_velocity.x
        self.om1.y = msg.angular_velocity.y
        self.om1.z = msg.angular_velocity.z

        self.acc1.x = msg.linear_acceleration.x
        self.acc1.y = msg.linear_acceleration.y
        self.acc1.z = msg.linear_acceleration.z

    def imuCb2(self, msg):


        #global roll, pitch, yaw
        orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

	self.ang2.x = roll
	self.ang2.y = pitch
	self.ang2.z = yaw

        self.om2.x = msg.angular_velocity.x
        self.om2.y = msg.angular_velocity.y
        self.om2.z = msg.angular_velocity.z

        self.acc2.x = msg.linear_acceleration.x
        self.acc2.y = msg.linear_acceleration.y
        self.acc2.z = msg.linear_acceleration.z



    def imuCb3(self, msg):


        #global roll, pitch, yaw
        orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

	self.ang3.x = roll
	self.ang3.y = pitch
	self.ang3.z = yaw

        self.om3.x = msg.angular_velocity.x
        self.om3.y = msg.angular_velocity.y
        self.om3.z = msg.angular_velocity.z

        self.acc3.x = msg.linear_acceleration.x
        self.acc3.y = msg.linear_acceleration.y
        self.acc3.z = msg.linear_acceleration.z



    def imuCb4(self, msg):


        #global roll, pitch, yaw
        orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

	self.ang4.x = roll
	self.ang4.y = pitch
	self.ang4.z = yaw

        self.om4.x = msg.angular_velocity.x
        self.om4.y = msg.angular_velocity.y
        self.om4.z = msg.angular_velocity.z

        self.acc4.x = msg.linear_acceleration.x
        self.acc4.y = msg.linear_acceleration.y
        self.acc4.z = msg.linear_acceleration.z




    def vecCb1(self, msg):

#        self.ve1.x = msg.twist.linear.x
#        self.ve1.y = msg.twist.linear.y
#        self.ve1.z = msg.twist.linear.z

        self.ve1.x = msg.twist.twist.linear.x
        self.ve1.y = msg.twist.twist.linear.y
        self.ve1.z = msg.twist.twist.linear.z

    def vecCb2(self, msg):

#        self.ve2.x = msg.twist.linear.x
#        self.ve2.y = msg.twist.linear.y
#        self.ve2.z = msg.twist.linear.z

        self.ve2.x = msg.twist.twist.linear.x
        self.ve2.y = msg.twist.twist.linear.y
        self.ve2.z = msg.twist.twist.linear.z

    def vecCb3(self, msg):

#        self.ve3.x = msg.twist.linear.x
#        self.ve3.y = msg.twist.linear.y
#        self.ve3.z = msg.twist.linear.z

        self.ve3.x = msg.twist.twist.linear.x
        self.ve3.y = msg.twist.twist.linear.y
        self.ve3.z = msg.twist.twist.linear.z

    def vecCb4(self, msg):

#        self.ve4.x = msg.twist.linear.x
#        self.ve4.y = msg.twist.linear.y
#        self.ve4.z = msg.twist.linear.z

        self.ve4.x = msg.twist.twist.linear.x
        self.ve4.y = msg.twist.twist.linear.y
        self.ve4.z = msg.twist.twist.linear.z




    ###################
    ## End Callbacks ##
    ###################





		

# Main function
def main():


    # initiate node
    rospy.init_node('setpoint_multi_switch_node', anonymous=True)

    # flight mode object
    modes = fcuModes()
    # controller object
    cnt = Controller()

    # ROS loop rate, [Hz]
    dtHz= 100.0	
    dt=1/dtHz
    rate = rospy.Rate(dtHz)


    # interval tau
    tauc = 50

    # Subscribe to drone state
#    rospy.Subscriber('tetra4/mavros/state', State, cnt.stateCb)


##---------------MoCap subscribe--------------------------

    rospy.Subscriber('/vrpn_client_node/tetra1g/pose', PoseStamped, cnt.mocapCb, queue_size=1)


##-------------local position subscribe-------------------


#    rospy.Subscriber('tetra1/mavros/local_position/pose', PoseStamped, cnt.posCb1, queue_size=1)
#    rospy.Subscriber('tetra2/mavros/local_position/pose', PoseStamped, cnt.posCb2, queue_size=1)
#    rospy.Subscriber('tetra3/mavros/local_position/pose', PoseStamped, cnt.posCb3, queue_size=1)
    rospy.Subscriber('tetra4/mavros/local_position/pose', PoseStamped, cnt.posCb, queue_size=1)


##------------- sensors subscribe-------------------------


    rospy.Subscriber('tetra1/mavros/local_position/odom', Odometry, cnt.vecCb1)
    rospy.Subscriber('tetra1/mavros/imu/data', Imu, cnt.imuCb1, queue_size=1)

    rospy.Subscriber('tetra2/mavros/local_position/odom', Odometry, cnt.vecCb2)
    rospy.Subscriber('tetra2/mavros/imu/data', Imu, cnt.imuCb2, queue_size=1)

    rospy.Subscriber('tetra3/mavros/local_position/odom', Odometry, cnt.vecCb3)
    rospy.Subscriber('tetra3/mavros/imu/data', Imu, cnt.imuCb3, queue_size=1)

    rospy.Subscriber('tetra4/mavros/local_position/odom', Odometry, cnt.vecCb4)
    rospy.Subscriber('tetra4/mavros/imu/data', Imu, cnt.imuCb4, queue_size=1)




##-----------position (modified) publish--------------
    


    mocap_pub1 = rospy.Publisher('tetra1/mavros/vision_pose/pose', PoseStamped, queue_size=1)
    mocap_pub2 = rospy.Publisher('tetra2/mavros/vision_pose/pose', PoseStamped, queue_size=1)
    mocap_pub3 = rospy.Publisher('tetra3/mavros/vision_pose/pose', PoseStamped, queue_size=1)
    mocap_pub4 = rospy.Publisher('tetra4/mavros/vision_pose/pose', PoseStamped, queue_size=1)



##-------------setpoint publish------------------------


#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_raw/local', PoseStamped, queue_size=10)

#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_position/global', PoseStamped, queue_size=10)


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


    	##------ positions table

    L=0.25
    Lz=0.25
    pbTop=([[0.0],[0.0],[Lz]])
    pbFront=([[L],[0.0],[0.0]])
    pbRight=([[-0.5*L],[-0.866025*L],[0.0]])
    pbLeft=([[-0.5*L],[0.866025*L],[0.0]])

	
    pb=np.array([pbTop,pbFront,pbRight,pbLeft])



        #--------- initializations


    N=4
    jj=list(range(1,N+1))
    ni=math.factorial(N)

    Jt= [0]*ni
    Jt_now = [0]*ni
    dJ=[0]*ni

    sw_perm = list(permutations(jj))


    II=list(range(1,ni+1))
    I0=II

    It=I0

    time_sw = 0
    time_prv = time_sw
    time_real = time.time()

    swi=11
    sw0=sw_perm[swi-1]

    pd=[cnt.sp.pose.position.x,cnt.sp.pose.position.y,cnt.sp.pose.position.z]
    vd=[0.0,0.0,0.0]
    xd=[[pd],[vd]]

    lm = -20

    v1e=[[0.0],[0.0],[0.0]]
    v2e=v1e
    v3e=v1e
    v4e=v1e

    p1e=v1e
    p2e=v1e
    p3e=v1e
    p4e=v1e



    # ROS main loop
    while not rospy.is_shutdown():









##--------------- variables setup------------------


	#---- var defs


	ax1=cnt.ang1.x
	ay1=cnt.ang1.y
	az1=cnt.ang1.z

	vx1=cnt.ve1.x
	vy1=cnt.ve1.y
	vz1=cnt.ve1.z

	wx1=cnt.om1.x
	wy1=cnt.om1.y
	wz1=cnt.om1.z

	acx1=cnt.acc1.x
	acy1=cnt.acc1.y
	acz1=cnt.acc1.z

	px1=cnt.local_pos.x
	py1=cnt.local_pos.y
	pz1=cnt.local_pos.z	

	pv1=[[px1],[py1],[pz1]]
	vv1=[[vx1],[vy1],[vz1]]
	wv1=[[wx1],[wy1],[wz1]]
	x1_acc=[[acx1],[acy1],[acz1]]
	th1=Point(ax1,ay1,az1)

	ax2=cnt.ang2.x
	ay2=cnt.ang2.y
	az2=cnt.ang2.z

	vx2=cnt.ve2.x
	vy2=cnt.ve2.y
	vz2=cnt.ve2.z

	wx2=cnt.om2.x
	wy2=cnt.om2.y
	wz2=cnt.om2.z

	acx2=cnt.acc2.x
	acy2=cnt.acc2.y
	acz2=cnt.acc2.z

	px2=cnt.local_pos.x
	py2=cnt.local_pos.y
	pz2=cnt.local_pos.z	

	pv2=[[px2],[py2],[pz2]]
	vv2=[[vx2],[vy2],[vz2]]
	wv2=[[wx2],[wy2],[wz2]]
	x2_acc=[[acx2],[acy2],[acz2]]
	th2=Point(ax2,ay2,az2)

	ax3=cnt.ang3.x
	ay3=cnt.ang3.y
	az3=cnt.ang3.z

	vx3=cnt.ve3.x
	vy3=cnt.ve3.y
	vz3=cnt.ve3.z

	acx3=cnt.acc3.x
	acy3=cnt.acc3.y
	acz3=cnt.acc3.z


	wx3=cnt.om3.x
	wy3=cnt.om3.y
	wz3=cnt.om3.z

	px3=cnt.local_pos.x
	py3=cnt.local_pos.y
	pz3=cnt.local_pos.z	

	pv3=[[px3],[py3],[pz3]]
	vv3=[[vx3],[vy3],[vz3]]
	wv3=[[wx3],[wy3],[wz3]]
	x3_acc=[[acx3],[acy3],[acz3]]
	th3=Point(ax1,ay3,az3)

	ax4=cnt.ang4.x
	ay4=cnt.ang4.y
	az4=cnt.ang4.z

	vx4=cnt.ve4.x
	vy4=cnt.ve4.y
	vz4=cnt.ve4.z

	wx4=cnt.om4.x
	wy4=cnt.om4.y
	wz4=cnt.om4.z

	acx4=cnt.acc4.x
	acy4=cnt.acc4.y
	acz4=cnt.acc4.z

	px4=cnt.local_pos.x
	py4=cnt.local_pos.y
	pz4=cnt.local_pos.z	

	pv4=[[px4],[py4],[pz4]]
	vv4=[[vx4],[vy4],[vz4]]
	wv4=[[wx4],[wy4],[wz4]]
	x4_acc=[[acx4],[acy4],[acz4]]
	th4=Point(ax4,ay4,az4)


	#---- ref vehicle

	

	[pB, vB] = cnt.hinv_f(pv4, vv4, th4, wv4, pbTop)
        thB=th4
        wB=wv4






	#----- simulation outputs

	
	[pv1, vv1s]= cnt.h_f(pB, vB, thB, wB, pbFront)
	[pv2, vv2s]= cnt.h_f(pB, vB, thB, wB, pbLeft)
	[pv3, vv3s]= cnt.h_f(pB, vB, thB, wB, pbRight)
	[pv4, vv4s]= cnt.h_f(pB, vB, thB, wB, pbTop)
	
 
#        print("------pos------")
#	print(pv2[0])
#	print(pv2[1])
#	print(pv2[2])
#        print("------vel------")
#	print(vv2s[0])
#	print(vv2s[1])
#	print(vv2s[2])





	#-------------------------------
	#-------------------------------

#	print(pb[1,:])

	for i in II:
#		print(i)
		[pj1, vj1]=cnt.hinv_f(pv1, vv1, th1, wv1, pb[sw_perm[i-1][0]-1, :])
		[pj2, vj2]=cnt.hinv_f(pv2, vv2, th2, wv2, pb[sw_perm[i-1][1]-1, :])
		[pj3, vj3]=cnt.hinv_f(pv3, vv3, th3, wv3, pb[sw_perm[i-1][2]-1, :])
		[pj4, vj4]=cnt.hinv_f(pv4, vv4, th4, wv4, pb[sw_perm[i-1][3]-1, :])


		v1e = v1e + dt * (lm * v1e - lm * vj1 + x1_acc)
		v2e = v2e + dt * (lm * v2e - lm * vj2 + x2_acc)
		v3e = v3e + dt * (lm * v3e - lm * vj3 + x3_acc)
		v4e = v4e + dt * (lm * v4e - lm * vj4 + x4_acc)



		p1e = p1e + dt * (lm * p1e - lm * pj1 + v1e)
		p2e = p2e + dt * (lm * p2e - lm * pj2 + v2e)
		p3e = p3e + dt * (lm * p3e - lm * pj3 + v3e)
		p4e = p4e + dt * (lm * p4e - lm * pj4 + v4e)




		ep1= p1e - pj1
		ep2= p2e - pj2
		ep3= p3e - pj3
		ep4= p4e - pj4

		ev1= v1e - vj1
		ev2= v2e - vj2
		ev3= v3e - vj3
		ev4= v4e - vj4

		

#		ep1= pd - pj1
#		ep2= pd - pj2
#		ep3= pd - pj3
#		ep4= pd - pj4

#		ev1= vd - vj1
#		ev2= vd - vj2
#		ev3= vd - vj3
#		ev4= vd - vj4
	
#		print(vj1,vj2,vj3,vj4)

#		Jtsq=np.linalg.norm([[ep1],[ep2],[ep3],[ep4],[ev1],[ev2],[ev3],[ev4]])
#		Jtsq=np.linalg.norm([[ep1],[ep2],[ep3],[ep4]])
		Jtsq=np.linalg.norm([[ev1],[ev2],[ev3],[ev4]])


		Jti_temp = Jt[i-1] + Jtsq * Jtsq
		Jt[i-1] = Jti_temp
#		Jt[i-1] = Jtsq
#		print(Jtsq)
#		print("------")




#	print(time_sw*dt)
##	print(time_prv)	
##	print(dt)
#	print("-----------")
#	print(time.time()-time_real)
		

	print(cnt.SWprint(sw_perm[swi-1][:]))

	if time_sw - time_prv > tauc:
		time_prv = time_sw
#		print(Jt)
#		print("------")
#		print(Jt_now)
# 		dJ = np.subtract(Jt, Jt_now)
#		dJ=[0]*24
#		for i in II:
#			dJ[i-1]=Jt[i-1] - Jt_now[i-1]
			

#		print(dJ)


#		[swi, It] = cnt.switching(dJ, II, It, swi)
		[swi, It] = cnt.switching(Jt, II, It, swi)

		Jt_temp = Jt
	
		Jt_now = Jt_temp
#		Jt= [0]*ni

		
		
#	print(dJ)
		

#		print(sw_perm[swi-1][:])

#		print(cnt.SWprint(sw_perm[swi-1][:]))

	
	

	time_sw = time_sw + 1



##----------- modified feedback------------------

	[psw1, vsw1]=cnt.hinv_f(pv1, vv1, th1, wv1, pb[sw_perm[swi-1][0]-1, :])
	[psw2, vsw2]=cnt.hinv_f(pv2, vv2, th2, wv2, pb[sw_perm[swi-1][1]-1, :])
	[psw3, vsw3]=cnt.hinv_f(pv3, vv3, th3, wv3, pb[sw_perm[swi-1][2]-1, :])
	[psw4, vsw4]=cnt.hinv_f(pv4, vv4, th4, wv4, pb[sw_perm[swi-1][3]-1, :])


        cnt.mocap_pos1=cnt.mocap_pos
        cnt.mocap_pos2=cnt.mocap_pos
        cnt.mocap_pos3=cnt.mocap_pos
        cnt.mocap_pos4=cnt.mocap_pos

#	cnt.mocap_pos1.pose.position.x=float(psw1[0])
#	cnt.mocap_pos1.pose.position.y=float(psw1[1])
#	cnt.mocap_pos1.pose.position.z=float(psw1[2])

#	cnt.mocap_pos2.pose.position.x=float(psw2[0])
#	cnt.mocap_pos2.pose.position.y=float(psw2[1])
#	cnt.mocap_pos2.pose.position.z=float(psw2[2])

#	cnt.mocap_pos3.pose.position.x=float(psw3[0])
#	cnt.mocap_pos3.pose.position.y=float(psw3[1])
#	cnt.mocap_pos3.pose.position.z=float(psw3[2])

#	cnt.mocap_pos4.pose.position.x=float(psw4[0])
#	cnt.mocap_pos4.pose.position.y=float(psw4[1])
#	cnt.mocap_pos4.pose.position.z=float(psw4[2])





##----------------setpoint control-------------------




#        cnt.updateSp()
	mocap_pub1.publish(cnt.mocap_pos1)
	mocap_pub2.publish(cnt.mocap_pos2)
	mocap_pub3.publish(cnt.mocap_pos3)
	mocap_pub4.publish(cnt.mocap_pos4)
        sp_pub1.publish(cnt.sp)
        sp_pub2.publish(cnt.sp)
        sp_pub3.publish(cnt.sp)
        sp_pub4.publish(cnt.sp)


#	print(cnt.local_pos.z)
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
