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

    def setArm(self):
        rospy.wait_for_service('tetra4/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra4/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            arm1 = armService(True)
	    print "Service arming call: %s"% arm1.success
        except rospy.ServiceException, e:
            print "Service arming call failed: %s"%e

    def setDisarm(self):
        rospy.wait_for_service('tetra4/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('tetra4/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException, e:
            print "Service disarming call failed: %s"%e

    def setStabilizedMode(self):
        rospy.wait_for_service('tetra4/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra4/mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='STABILIZED')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Stabilized Mode could not be set."%e

    def setOffboardMode(self):
        rospy.wait_for_service('tetra4/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra4/mavros/set_mode', mavros_msgs.srv.SetMode)
#            flightModeService(custom_mode='OFFBOARD')
            success = flightModeService(custom_mode='OFFBOARD')
            print success
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Offboard Mode could not be set."%e

    def setAltitudeMode(self):
        rospy.wait_for_service('tetra4/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra4/mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='ALTCTL')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Altitude Mode could not be set."%e

    def setPositionMode(self):
        rospy.wait_for_service('tetra4/mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('tetra4/mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='POSCTL')
        except rospy.ServiceException, e:
            print "service set_mode call failed: %s. Position Mode could not be set."%e

    def setAutoLandMode(self):
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
        self.state = State()
        # Instantiate a setpoints message
#        self.sp         = PositionTarget()
	self.sp         = PoseStamped()
        # set the flag to use position setpoints and yaw angle
#        self.sp.type_mask    = int('010111111000', 2)
        # LOCAL_NED
#        self.sp.coordinate_frame= 1

        # We will fly at a fixed altitude for now
        # Altitude setpoint, [meters]
        #self.ALT_SP        = 1.0
        # update the setpoint message with the required altitude
#        self.sp.position.z    = self.ALT_SP
	self.sp.pose.position.z    = 1
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

	#self.modes = fcuModes()

	#self.kk = 0
	#self.kk = 10.0

    # Callbacks ##

    ## local position callback
    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z

#        print(self.local_pos.x)

    ## joystick callback
#    def joyCb(self, msg):
#        self.joy_msg = msg

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

    ###################
    ## End Callbacks ##
    ###################


    ## Update setpoint message
#    def updateSp(self):
        #x = 1.0*self.joy_msg.axes[1]
        #y = 1.0*self.joy_msg.axes[0]

	#self.sp.position.x = self.local_pos.x + self.STEP_SIZE*x
	#self.sp.position.y = self.local_pos.y + self.STEP_SIZE*y
    
#        self.sp.pose.position.x = 0
#        self.sp.pose.position.y = 0
#        self.sp.position.x = 0
#        self.sp.position.y = 0
#	self.sp.header.frame_id = "8"
#	self.sp.header.seq = 1

#	if self.xflag == 0:
#		if self.local_pos.z > 4.5:
#			self.xflag = 1
#			self.hovering_time = time.time()

	xb_t=0
	yb_t=0
	zb_t=1

	xb_f=1
	yb_f=0
	zb_f=0

	xb_r=-0.5
	yb_r=0.5
	zb_r=0

	xb_l=-0.5
	yb_l=-0.5
	zb_l=0


	xb1=0
	yb1=0
	zb1=1


#	BfT=numpy.array([[1,yb_f,-xb_f,0],[0,1,0,1],[0,0,0,1],[0,0,0,1]])
#	Bf=BfT.T

#	BrT=numpy.array([[1,yb_r,-xb_r,0],[0,1,0,1],[0,0,0,1],[0,0,0,1]])
#	Br=BrT.T

#	BlT=numpy.array([[1,yb_l,-xb_l,0],[0,1,0,1],[0,0,0,1],[0,0,0,1]])
#	Bl=BlT.T

#	BtT=numpy.array([[1,yb_t,-xb_t,0],[0,1,0,1],[0,0,0,1],[0,0,0,1]])
#	Bt=BtT.T

#	Btilde=numpy.block([[Bt,Bf,Br,Bl]])

#	BBT=numpy.dot(Btilde,Btilde.T)

	

	A=yb_t+yb_l+yb_r+yb_f
	B=-xb_t-xb_l-xb_r-xb_f
	C=yb_t*yb_t+yb_l*yb_l+yb_r*yb_r+yb_f*yb_f
	D=xb_t*xb_t+xb_l*xb_l+xb_r*xb_r+xb_f*xb_f
	E=-yb_t*xb_t-yb_l*xb_l-yb_r*xb_r-yb_f*xb_f

	M=4*(4*(4+C)*(4+D)-4*E*E)+A*(4*B*E-4*A*(4+D))+B*(4*E*A-4*B*(4+C))

	F11=4*(4+C)*(4+D)-4*E*E
	F12=4*B*E-4*A*(4+D)
	F13=4*A*E-4*B*(4+C)

	F21=4*B*E-4*A*(4+D)
	F22=16*(4+D)-4*B*B
	F23=4*A*B-16*E

	F31=4*A*E-4*B*(4+C)
	F32=4*A*B-16*E
	F33=16*(4+C)-4*A*A

	F44=4*((4+C)*(4+D)-E*E)-A*(A*(4+D)-B*E)+B*(A*E-B*(4+C))



	K_mx_th=F21/M
	K_mx_mx=F22/M
	K_mx_my=F23/M

	K_my_th=F31/M
	K_my_mx=F32/M
	K_my_my=F33/M

	K_mz_mz=F44/M

	Kt_th_th=(F11+yb_t*F21-xb_t*F31)/M
	Kt_th_mx=(F12+yb_t*F22-xb_t*F32)/M
	Kt_th_my=(F13+yb_t*F23-xb_t*F33)/M

	Kr_th_th=(F11+yb_r*F21-xb_r*F31)/M
	Kr_th_mx=(F12+yb_r*F22-xb_r*F32)/M
	Kr_th_my=(F13+yb_r*F23-xb_r*F33)/M

	Kl_th_th=(F11+yb_l*F21-xb_l*F31)/M
	Kl_th_mx=(F12+yb_l*F22-xb_l*F32)/M
	Kl_th_my=(F13+yb_l*F23-xb_l*F33)/M

	Kf_th_th=(F11+yb_f*F21-xb_f*F31)/M
	Kf_th_mx=(F12+yb_f*F22-xb_f*F32)/M
	Kf_th_my=(F13+yb_f*F23-xb_f*F33)/M





	
	#--------


	kk_th_th = 1.0 * 1.0
	kk_th_mx = 0.0 * 1.0
	kk_th_my = 0.0 * 1.0

	kk1_th_th = kk_th_th
	kk1_th_mx = kk_th_mx
	kk1_th_my = kk_th_my

	kk2_th_th = kk_th_th
	kk2_th_mx = kk_th_mx
	kk2_th_my = kk_th_my

	kk3_th_th = kk_th_th
	kk3_th_mx = kk_th_mx
	kk3_th_my = kk_th_my

	kk4_th_th = kk_th_th
	kk4_th_mx = kk_th_mx
	kk4_th_my = kk_th_my

	#-------------

	## 1g   2b    3r    4w

	kk1_th_th = Kt_th_th
	kk1_th_mx = Kt_th_mx
	kk1_th_my = Kt_th_my

	kk2_th_th = Kl_th_th
	kk2_th_mx = Kl_th_mx
	kk2_th_my = Kl_th_my

	kk3_th_th = Kr_th_th
	kk3_th_mx = Kr_th_mx
	kk3_th_my = Kr_th_my

	kk4_th_th = Kf_th_th
	kk4_th_mx = Kf_th_mx
	kk4_th_my = Kf_th_my

	



	#-------------

#	kk_mx_th = K_mx_th
#	kk_mx_mx = K_mx_mx
#	kk_mx_my = K_mx_my

#	kk_my_th = K_my_th
#	kk_my_mx = K_my_mx
#	kk_my_my = K_my_my

#	kk_mz_mz = K_mz_mz

	kk_mx_th = 0.0 * 1.0
	kk_mx_mx = 1.0 * 1.0
	kk_mx_my = 0.0 * 1.0

	kk_my_th = 0.0 * 1.0
	kk_my_mx = 0.0 * 1.0
	kk_my_my = 1.0 * 1.0

	kk_mz_mz = 1.0 * 1.0

	#-----------------


#		#B matrix = [THM_TH_C,THM_MX_C,THM_MY_C,THM_MZ_C,
#		#		MXM_TH_C,MXM_MX_C,MXM_MY_C,MXM_MZ_C,
#		#		MYM_TH_C,MYM_MX_C,MYM_MY_C,MYM_MZ_C,
#		#		MZM_TH_C,MZM_MX_C,MZM_MY_C,MZM_MZ_C]


#		rospy.loginfo("here we go")


#=============================================

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_TH_MC', value=ParamValue(integer=0, real=kk1_th_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MX_MC', value=ParamValue(integer=0, real=kk1_th_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MY_MC', value=ParamValue(integer=0, real=kk1_th_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e



# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_TH_MC', value=ParamValue(integer=0, real=kk_mx_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MX_MC', value=ParamValue(integer=0, real=kk_mx_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MY_MC', value=ParamValue(integer=0, real=kk_mx_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e






# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_TH_MC', value=ParamValue(integer=0, real=kk_my_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_MX_MC', value=ParamValue(integer=0, real=kk_my_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MYM_MY_MC', value=ParamValue(integer=0, real=kk_my_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e


# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MZM_MZ_MC', value=ParamValue(integer=0, real=kk_mz_mz))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e


##================================

# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_TH_MC', value=ParamValue(integer=0, real=kk2_th_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MX_MC', value=ParamValue(integer=0, real=kk2_th_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MY_MC', value=ParamValue(integer=0, real=kk2_th_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e



# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_TH_MC', value=ParamValue(integer=0, real=kk_mx_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MX_MC', value=ParamValue(integer=0, real=kk_mx_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MY_MC', value=ParamValue(integer=0, real=kk_mx_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e






# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_TH_MC', value=ParamValue(integer=0, real=kk_my_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_MX_MC', value=ParamValue(integer=0, real=kk_my_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MYM_MY_MC', value=ParamValue(integer=0, real=kk_my_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e


# 		rospy.wait_for_service('/tetra2/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra2/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MZM_MZ_MC', value=ParamValue(integer=0, real=kk_mz_mz))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e





##=============================


# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_TH_MC', value=ParamValue(integer=0, real=kk3_th_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MX_MC', value=ParamValue(integer=0, real=kk3_th_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MY_MC', value=ParamValue(integer=0, real=kk3_th_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e



# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_TH_MC', value=ParamValue(integer=0, real=kk_mx_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MX_MC', value=ParamValue(integer=0, real=kk_mx_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MY_MC', value=ParamValue(integer=0, real=kk_mx_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e






# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_TH_MC', value=ParamValue(integer=0, real=kk_my_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_MX_MC', value=ParamValue(integer=0, real=kk_my_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MYM_MY_MC', value=ParamValue(integer=0, real=kk_my_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e


# 		rospy.wait_for_service('/tetra3/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra3/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MZM_MZ_MC', value=ParamValue(integer=0, real=kk_mz_mz))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e



##=====================================


# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_TH_MC', value=ParamValue(integer=0, real=kk4_th_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MX_MC', value=ParamValue(integer=0, real=kk4_th_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='THM_MY_MC', value=ParamValue(integer=0, real=kk4_th_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e



# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_TH_MC', value=ParamValue(integer=0, real=kk_mx_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MX_MC', value=ParamValue(integer=0, real=kk_mx_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MXM_MY_MC', value=ParamValue(integer=0, real=kk_mx_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e






# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_TH_MC', value=ParamValue(integer=0, real=kk_my_th))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MyM_MX_MC', value=ParamValue(integer=0, real=kk_my_mx))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e

# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MYM_MY_MC', value=ParamValue(integer=0, real=kk_my_my))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e


# 		rospy.wait_for_service('/tetra4/mavros/param/set')
#		try:
#			change_mode = rospy.ServiceProxy('/tetra4/mavros/param/set', ParamSet)
#			resp = change_mode(param_id='MZM_MZ_MC', value=ParamValue(integer=0, real=kk_mz_mz))
#			print "setmode send ok %s" % resp.success
#		except rospy.ServiceException, e:
#			print "Failed SetMode: %s" % e


##==========================================





#		self.hovering_time = time.time()


		

# Main function
def main():


    # initiate node
    rospy.init_node('setpoint_v2_node', anonymous=True)

    # flight mode object
    modes = fcuModes()
    # controller object
    cnt = Controller()

    # ROS loop rate, [Hz]
    rate = rospy.Rate(20.0)

    # Subscribe to drone state
    rospy.Subscriber('tetra4/mavros/state', State, cnt.stateCb)


    ## TO DO - create a subscriber to drone's local position which calls cnt.posCb callback function
    rospy.Subscriber('tetra4/mavros/local_position/pose', PoseStamped, cnt.posCb, queue_size=1)

    ## TO DO - create a subscriber to joystick topic which call cnt.joyCb callback function
    

    ## TO DO - create a publisher with name "sp_pub" which will publish the position setpoints
    
#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_raw/local', PoseStamped, queue_size=10)
    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_position/local', PoseStamped, queue_size=1)
#    sp_pub = rospy.Publisher('tetra4/mavros/setpoint_position/global', PoseStamped, queue_size=10)






# Make sure the drone is armed
    #m=cnt.mm
    if not cnt.state.armed:
	for i in range(100):
#		cnt.sp.header.stamp = rospy.Time.now()
       		sp_pub.publish(cnt.sp)
       		rate.sleep()
       		#m = m+1
	#else:
       	modes.setArm()
       	rate.sleep()






   
    mm=0
    while mm<100:
#	cnt.sp.header.stamp = rospy.Time.now()
        sp_pub.publish(cnt.sp)
        rate.sleep()
        mm = mm+1

 # activate OFFBOARD mode
    modes.setOffboardMode()
    #cnt.set_mode("OFFBOARD", 5)
#    print "offboard enabled"
#    cnt.hovering_time = time.time()
#    cnt.xflag = 0







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

#        cnt.updateSp()
        sp_pub.publish(cnt.sp)
        rate.sleep()

    	if cnt.local_pos.z >= 0.8:
		print "here we go"
#		cnt.wflag=0
#	       	cnt.sp.pose.position.z = 0.0
		modes.setAutoLandMode()
    		kk=0
		while kk<20:
#			cnt.sp.header.stamp = rospy.Time.now()
        		sp_pub.publish(cnt.sp)
	        	rate.sleep()
	        	kk = kk+1
		modes.setDisarm()
	        rate.sleep()

#	sp_pub.publish(cnt.sp)
#        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
