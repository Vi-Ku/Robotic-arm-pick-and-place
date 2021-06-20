# This file is created to test the final transformation matrix        
# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *

# Create symbols
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')    
#   
# Create Modified DH parameters
s = {alpha0: 0, 	    a0: 0, 		d1: 0.75, 	q1: q1,
		    alpha1: -pi/2., a1: 0.35,	d2: 0, 		q2: -pi/2. + q2,
		    alpha2: 0, 	    a2: 1.25, 	d3: 0, 		q3: q3,
		    alpha3: -pi/2,  a3: -0.054, d4: 1.5, 	q4: q4,
		    alpha4: pi/2, 	a4: 0, 		d5: 0, 		q5: q5,
		    alpha5: -pi/2., a5: 0, 		d6: 0, 		q6: q6,
		    alpha6: 0, 	    a6: 0, 		d7: 0.303, 	q7: 0}
#
# Define Modified DH Transformation matrix
def Trans(q, d, a, alpha):
        t = Matrix([[             cos(q),            -sin(q),            0,              a],
                [ sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                [ sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                [                   0,                   0,            0,               1]])
        return t

#
# Create individual transformation matrices
T0_1=Trans(q1 , d1 , a0 , alpha0).subs(s)
T1_2=Trans(q2 , d2 , a1 , alpha1).subs(s)
T2_3=Trans(q3 , d3 , a2 , alpha2).subs(s)
T3_4=Trans(q4 , d4 , a3 , alpha3).subs(s)
T4_5=Trans(q5 , d5 , a4 , alpha4).subs(s)
T5_6=Trans(q6 , d6 , a5 , alpha5).subs(s)
T6_7=Trans(q7 , d7 , a6 , alpha6).subs(s)
T0_EE = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_EE
#
# Extract rotation matrices from the transformation matrices
r, p , y = symbols('r p y')

ROT_x = Matrix([[1, 0 ,      0],
		        [0, cos(r), -sin(r)],
   		        [0, sin(r), cos(r)]]) 

ROT_y = Matrix([[cos(p), 	0 , 	sin(p)],
		        [0, 		1, 	    0],
   		        [-sin(p), 	0, 	    cos(p)]]) 

ROT_z = Matrix([[cos(y), -sin(y), 0],
		        [sin(y), cos(y), 0],
		        [   0  ,    0  ,1 ]]) 
ROT_xyz = ROT_z * ROT_y * ROT_x
ROT_corr = ROT_z.subs(y, radians(180)) * ROT_y.subs(p, radians(-90))
ROT_final = ROT_xyz * ROT_corr

print("T0_EE = ")
pprint(T0_EE.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))