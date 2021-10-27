#!/usr/bin/env python3

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





class Controller:
    # initialization method
    def __init__(self):
        # Drone state
        self.state1 = State()
        self.state2 = State()
        self.state3 = State()
        self.state4 = State()
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
        self.local_posB = Point(0.0, 0.0, 0.0)
        self.local_vecB = Point(0.0, 0.0, 0.0)

        self.local_pos1 = Point(0.0, 0.0, 0.0)
        self.ang1 = Point(0.0, 0.0, 0.0)
        self.om1 = Point(0.0, 0.0, 0.0)
        self.ve1 = Point(0.0, 0.0, 0.0)
        self.acc1 = Point(0.0, 0.0, 0.0)

        self.local_pos2 = Point(0.0, 0.0, 0.0)
        self.ang2 = Point(0.0, 0.0, 0.0)
        self.om2 = Point(0.0, 0.0, 0.0)
        self.ve2 = Point(0.0, 0.0, 0.0)
        self.acc2 = Point(0.0, 0.0, 0.0)

        self.local_pos3 = Point(0.0, 0.0, 0.0)
        self.ang3 = Point(0.0, 0.0, 0.0)
        self.om3 = Point(0.0, 0.0, 0.0)
        self.ve3 = Point(0.0, 0.0, 0.0)
        self.acc3 = Point(0.0, 0.0, 0.0)

        self.local_pos4 = Point(0.0, 0.0, 0.0)
        self.ang4 = Point(0.0, 0.0, 0.0)
        self.om4 = Point(0.0, 0.0, 0.0)
        self.ve4 = Point(0.0, 0.0, 0.0)
        self.acc4 = Point(0.0, 0.0, 0.0)

	#self.modes = fcuModes()

	#self.kk = 0
	#self.kk = 10.0

    # Callbacks ##

    ## local position callback
    def posRb(self, msg):
        self.local_posB.x = msg.pose.position.x
        self.local_posB.y = msg.pose.position.y
        self.local_posB.z = msg.pose.position.z


    def posCb1(self, msg):
        self.local_pos1.x = msg.pose.position.x
        self.local_pos1.y = msg.pose.position.y
        self.local_pos1.z = msg.pose.position.z

    def posCb2(self, msg):
        self.local_pos2.x = msg.pose.position.x
        self.local_pos2.y = msg.pose.position.y
        self.local_pos2.z = msg.pose.position.z

    def posCb3(self, msg):
        self.local_pos3.x = msg.pose.position.x
        self.local_pos3.y = msg.pose.position.y
        self.local_pos3.z = msg.pose.position.z

    def posCb4(self, msg):
        self.local_pos4.x = msg.pose.position.x
        self.local_pos4.y = msg.pose.position.y
        self.local_pos4.z = msg.pose.position.z


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

        self.ve1.x = msg.twist.linear.x
        self.ve1.y = msg.twist.linear.y
        self.ve1.z = msg.twist.linear.z

#        self.ve1.x = msg.twist.twist.linear.x
#        self.ve1.y = msg.twist.twist.linear.y
#        self.ve1.z = msg.twist.twist.linear.z

    def vecCb2(self, msg):

        self.ve2.x = msg.twist.linear.x
        self.ve2.y = msg.twist.linear.y
        self.ve2.z = msg.twist.linear.z

#        self.ve2.x = msg.twist.twist.linear.x
#        self.ve2.y = msg.twist.twist.linear.y
#        self.ve2.z = msg.twist.twist.linear.z

    def vecCb3(self, msg):

        self.ve3.x = msg.twist.linear.x
        self.ve3.y = msg.twist.linear.y
        self.ve3.z = msg.twist.linear.z

#        self.ve3.x = msg.twist.twist.linear.x
#        self.ve3.y = msg.twist.twist.linear.y
#        self.ve3.z = msg.twist.twist.linear.z

    def vecCb4(self, msg):

        self.ve4.x = msg.twist.linear.x
        self.ve4.y = msg.twist.linear.y
        self.ve4.z = msg.twist.linear.z

#        self.ve4.x = msg.twist.twist.linear.x
#        self.ve4.y = msg.twist.twist.linear.y
#        self.ve4.z = msg.twist.twist.linear.z






    ## Drone State callback
    def stateCb1(self, msg):
        self.state1 = msg

    def stateCb2(self, msg):
        self.state2 = msg

    def stateCb3(self, msg):
        self.state3 = msg

    def stateCb4(self, msg):
        self.state4 = msg

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

	

    ###################
    ## End Callbacks ##
    ###################


	

# Main function
def main():


    # initiate node
    rospy.init_node('switch_v1_node', anonymous=True)

    
    # controller object
    cnt = Controller()

    # ROS loop rate, [Hz]
    dtHz=10.0
    rate = rospy.Rate(dtHz)
    dt=1/dtHz

    # interval tau
    tauc = 50
    

    

    # Subscribe to drone state
#    rospy.Subscriber('tetra1/mavros/state', State, cnt.stateCb1)
#    rospy.Subscriber('tetra1/mavros/state', State, cnt.stateCb2)
#    rospy.Subscriber('tetra1/mavros/state', State, cnt.stateCb3)
#    rospy.Subscriber('tetra1/mavros/state', State, cnt.stateCb4)


    ## TO DO - create a subscriber to drone's local position which calls cnt.posCb callback function
    rospy.Subscriber('tetra1/mavros/local_position/pose', PoseStamped, cnt.posCb1)
    rospy.Subscriber('tetra2/mavros/local_position/pose', PoseStamped, cnt.posCb2, queue_size=1)
    rospy.Subscriber('tetra3/mavros/local_position/pose', PoseStamped, cnt.posCb3, queue_size=1)
    rospy.Subscriber('tetra4/mavros/local_position/pose', PoseStamped, cnt.posCb4, queue_size=1)

    rospy.Subscriber('tetra1/mavros/local_position/velocity', TwistStamped, cnt.vecCb1)
    rospy.Subscriber('tetra2/mavros/local_position/velocity', TwistStamped, cnt.vecCb2)
    rospy.Subscriber('tetra3/mavros/local_position/velocity', TwistStamped, cnt.vecCb3)
    rospy.Subscriber('tetra4/mavros/local_position/velocity', TwistStamped, cnt.vecCb4)

#    rospy.Subscriber('tetra1/mavros/local_position/odom', Odometry, cnt.vecCb1)
#    rospy.Subscriber('tetra2/mavros/local_position/odom', Odometry, cnt.vecCb2)
#    rospy.Subscriber('tetra3/mavros/local_position/odom', Odometry, cnt.vecCb3)
#    rospy.Subscriber('tetra4/mavros/local_position/odom', Odometry, cnt.vecCb4)


    #imu data
    rospy.Subscriber('tetra1/mavros/imu/data', Imu, cnt.imuCb1)
#    data1	
    rospy.Subscriber('tetra2/mavros/imu/data', Imu, cnt.imuCb2, queue_size=1)
    rospy.Subscriber('tetra3/mavros/imu/data', Imu, cnt.imuCb3, queue_size=1)
    rospy.Subscriber('tetra4/mavros/imu/data', Imu, cnt.imuCb4, queue_size=1)
    


#    dataB = rospy.Subscriber('/vrpn_client_node/drone/pose', PoseStamped, cnt.posRb, queue_size=1)


    	#------ positions table

    L=1
    Lz=1
    pbTop=([[0],[0],[Lz]])
    pbFront=([[0],[L],[0]])
    pbRight=([[-0.5*L],[-0.866025*L],[0]])
    pbLeft=([[-0.5*L],[0.866025*L],[0]])

	
    pb=np.array([pbTop,pbFront,pbRight,pbLeft])

	#[2,4,3,1]



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

    time = 0
    time_prv = time

    swi=18
    sw0=sw_perm[swi-1]



    pd=[0,0,0]
    vd=[0,0,0]
    xd=[[pd],[vd]]


    vx2s=0.0

    # ROS main loop
    while not rospy.is_shutdown():
        rate.sleep()


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

        px1=cnt.local_pos1.x
        py1=cnt.local_pos1.y
        pz1=cnt.local_pos1.z	

        pv1=[[px1],[py1],[pz1]]
        vv1=[[vx1],[vy1],[vz1]]
        wv1=[[wx1],[wy1],[wz1]]
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

        px2=cnt.local_pos2.x
        py2=cnt.local_pos2.y
        pz2=cnt.local_pos2.z	

        pv2=[[px2],[py2],[pz2]]
        vv2=[[vx2],[vy2],[vz2]]
        wv2=[[wx2],[wy2],[wz2]]
        th2=Point(ax2,ay2,az2)

        ax3=cnt.ang3.x
        ay3=cnt.ang3.y
        az3=cnt.ang3.z

        vx3=cnt.ve3.x
        vy3=cnt.ve3.y
        vz3=cnt.ve3.z


        wx3=cnt.om3.x
        wy3=cnt.om3.y
        wz3=cnt.om3.z

        px3=cnt.local_pos3.x
        py3=cnt.local_pos3.y
        pz3=cnt.local_pos3.z	

        pv3=[[px3],[py3],[pz3]]
        vv3=[[vx3],[vy3],[vz3]]
        wv3=[[wx3],[wy3],[wz3]]
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

        px4=cnt.local_pos4.x
        py4=cnt.local_pos4.y
        pz4=cnt.local_pos4.z	

        pv4=[[px4],[py4],[pz4]]
        vv4=[[vx4],[vy4],[vz4]]
        wv4=[[wx4],[wy4],[wz4]]
        th4=Point(ax4,ay4,az4)
        
        
        acc1s=cnt.acc1
        acc2s=cnt.acc2
        acc3s=cnt.acc3
        acc4s=cnt.acc4

#        print([vx1,vx2,vx3,vx4])

#        vx2s=vx2s+dt*acc2s.x


#        print(vx2s)


	#---- ref vehicle
	# make sure before testing
	

        [pB, vB] = cnt.hinv_f(pv4, vv4, th4, wv4, pbTop)
        wB=wv4
        thB=th4
#	print(pv4)
#	print("------------------")

	#[-0.314328134059906], [0.3057825863361359], [0.4692144393920898]



	#----- simulation outputs
	# make sure before testing

	
        [pv1, vv1s]= cnt.h_f(pB, vB, thB, wB, pbFront)
        vv1=np.array(vv1)
#	wv1=wB
#	th1=thB





        [pv2, vv2s]= cnt.h_f(pB, vB, thB, wB, pbLeft)
        vv2=np.array(vv2)
#	wv2=wB
#	th2=thB

        [pv3, vv3s]= cnt.h_f(pB, vB, thB, wB, pbRight)
        vv3=np.array(vv3)
#	wv3=wB
#	th3=thB

        [pv4, vv4s]= cnt.h_f(pB, vB, thB, wB, pbTop)
        vv4=np.array(vv4)
#	wv4=wB
#	th4=thB

#	print(pv4)

#	print(pv2)
#	print("-------")
#	print(vv2)

	#-------------------------------
	#-------------------------------

#	print(pb[1,:])

        for i in II:
#		print(i)
        	[pj1, vj1]=cnt.hinv_f(pv1, vv1, th1, wv1, pb[sw_perm[i-1][0]-1, :])
        	[pj2, vj2]=cnt.hinv_f(pv2, vv2, th2, wv2, pb[sw_perm[i-1][1]-1, :])
        	[pj3, vj3]=cnt.hinv_f(pv3, vv3, th3, wv3, pb[sw_perm[i-1][2]-1, :])
        	[pj4, vj4]=cnt.hinv_f(pv4, vv4, th4, wv4, pb[sw_perm[i-1][3]-1, :])

        	ep1= pd - pj1
        	ep2= pd - pj2
        	ep3= pd - pj3
        	ep4= pd - pj4

        	ev1= vd - vj1
        	ev2= vd - vj2
        	ev3= vd - vj3
        	ev4= vd - vj4
	
#		print(vj1,vj2,vj3,vj4)

#		Jtsq=np.linalg.norm([[ep1],[ep2],[ep3],[ep4],[ev1],[ev2],[ev3],[ev4]])
#		Jtsq=np.linalg.norm([[ep1],[ep2],[ep3],[ep4]])
	        Jtsq=np.linalg.norm([[ev1],[ev2],[ev3],[ev4]])


	        Jti_temp = Jt[i-1] + Jtsq * Jtsq
	        Jt[i-1] = Jti_temp
#		Jt[i-1] = Jtsq
#		print(Jtsq)




			
		



        if time - time_prv > tauc:
        	time_prv = time
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
#        	Jt=0.0

#		Jt_temp = Jt
#	
#		Jt_now = Jt_temp
#		Jt= [0]*ni

		
		
#	print(dJ)
		

#		print(sw_perm[swi-1][:])

	        print(cnt.SWprint(sw_perm[swi-1][:]))

	
	

	




	



	
        time = time + 1
	

	
	
	

	
  

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
