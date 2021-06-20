#!/usr/bin/env python

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
from sympy import pi 
from sympy.matrices import Matrix 



def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        ### Your FK code here
        # Create symbols
        q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')


	# Create Modified DH parameters
        DH_table = {alpha0:      0, a0:      0, d1:  0.75, q1:        q1,
              alpha1: -pi/2 , a1:   0.35, d2:     0, q2: -pi/2 + q2,
              alpha2:      0, a2:   1.25, d3:     0, q3:        q3,
              alpha3: -pi/2 , a3: -0.054, d4:   1.5, q4:        q4,
              alpha4:  pi/2 , a4:      0, d5:     0, q5:        q5,
              alpha5: -pi/2 , a5:      0, d6:     0, q6:        q6,
              alpha6:      0, a6:      0, d7: 0.303, q7:         0}
	
	# Define Modified DH Transformation matrix
        def Trans(q, d, a, alpha):
            t = Matrix([
                    [             cos(q),            -sin(q),            0,              a],
                    [ sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                    [ sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                    [                   0,                   0,            0,               1]])
            return t

	
	# Create individual transformation matrices
	#
        T0_1=Trans(q1 , d1 , a0 , alpha0).subs(DH_table)
        T1_2=Trans(q2 , d2 , a1 , alpha1).subs(DH_table)
        T2_3=Trans(q3 , d3 , a2 , alpha2).subs(DH_table)
        T3_4=Trans(q4 , d4 , a3 , alpha3).subs(DH_table)
        T4_5=Trans(q5 , d5 , a4 , alpha4).subs(DH_table)
        T5_6=Trans(q6 , d6 , a5 , alpha5).subs(DH_table)
        T6_7=Trans(q7 , d7 , a6 , alpha6).subs(DH_table)

        T0_3=T0_1*T1_2*T2_3
        T0_7=T0_3*T3_4*T4_5*T5_6*T6_7

	#
        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

	    # Extract end-effector position and orientation from request
	    # px,py,pz = end-effector position
	    # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here
	    # Compensate for rotation discrepancy between DH parameters and Gazebo
	    #
            r,p,y = symbols('r p y')

                # Roll
            ROT_x = Matrix([[       1,       0,       0],
                            [       0,  cos(r), -sin(r)],
                            [       0,  sin(r),  cos(r)]])
            # Pitch
            ROT_y = Matrix([[  cos(p),       0,  sin(p)],
                            [       0,       1,       0],
                            [ -sin(p),       0,  cos(p)]])
            # Yaw
            ROT_z = Matrix([[  cos(y), -sin(y),       0],
                            [  sin(y),  cos(y),       0],
                            [       0,       0,       1]])    
            
            
            ROT_Cor = ROT_z * ROT_y * ROT_x
            ROT_corr = ROT_z.subs(y, radians(180)) * ROT_y.subs(p, radians(-90))
            ROT_EE = ROT_Cor * ROT_corr
            ROT_EE = ROT_EE.subs({'r': roll, 'p': pitch, 'y': yaw})
	    # Calculate joint angles using Geometric IK method

            P=Matrix([[px],[py],[pz]]) 
            wc=P-0.303*ROT_EE[:,2]

        #Calculating the parameters required to calculate the q2 and q3

            a=1.501
            c=1.25
            b=sqrt((sqrt(wc[0]**2+wc[1]**2)-0.35)**2+(wc[2]-0.75)**2)
            angle_x=atan2(wc[2]-0.75,sqrt(wc[0]**2+wc[1]**2)-0.35)
            angle_a=acos((b**2+c**2-a**2)/(2*b*c))
            angle_b=acos((a**2+c**2-b**2)/(2*a*c))


        #Calculating the value of Q1, q2 and q3    

            theta1 = atan2(wc[1],wc[0])
            theta2 = pi/2-angle_x-angle_a 
            theta3 = pi/2-angle_b-0.036
            
        #subsituting the value of q1, q2 and q3 in the roation matrix, so that we can calculate the
        # value of q4, q5 and q5

            R0_3=T0_3[0:3,0:3]  
            R0_3=R0_3.subs({q1: theta1, q2: theta2, q3: theta3})
            R3_6 = R0_3.inv(method="LU")*ROT_EE

            #Finding the value of  q4, q5 and q6 from the roation matrix
            theta4 = atan2(R3_6[2,2], -R3_6[0,2])
            theta5 = atan2(sqrt(R3_6[0,2]*R3_6[0,2] + R3_6[2,2]*R3_6[2,2]), R3_6[1,2])
            theta6 = atan2(-R3_6[1,1], R3_6[1,0])

            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
