from mpmath import *
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables for DH parameters
alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha1:7')
a1, a2, a3, a4, a5, a6 = symbols('a1:7')
d1, d2, d3, d4, d5, d6 = symbols('d1:7')
theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1:7')

# Define DH parameters for each joint in a dictionary
DH_Table = {
    alpha1: pi/2,  a1: 0,       d1: 183.3/1000, theta1: pi/2 + theta1,
    alpha2: 0,     a2: 737.31/1000,  d2: 0,     theta2: pi/2 + theta2,
    alpha3: 0,     a3: 387.8/1000,   d3: 0,     theta3: theta3,
    alpha4: pi/2,  a4: 0,       d4: -95.5/1000, theta4: -pi/2 + theta4,
    alpha5: pi/2,  a5: 0,       d5: -115.5/1000,theta5: pi + theta5,
    alpha6: 0,     a6: 0,       d6: -76.8/1000, theta6: -pi + theta6
}

# Define the DH transformation matrix
def DH_tfmat(alpha, a, d, theta):
    return Matrix([
        [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Create transformation matrices for each joint using the DH parameters
T0_1 = DH_tfmat(alpha1, a1, d1, theta1).subs(DH_Table)
T1_2 = DH_tfmat(alpha2, a2, d2, theta2).subs(DH_Table)
T2_3 = DH_tfmat(alpha3, a3, d3, theta3).subs(DH_Table)
T3_4 = DH_tfmat(alpha4, a4, d4, theta4).subs(DH_Table)
T4_5 = DH_tfmat(alpha5, a5, d5, theta5).subs(DH_Table)
T5_EE = DH_tfmat(alpha6, a6, d6, theta6).subs(DH_Table)

# Calculate the end-effector transformation matrix
T0_EE = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_EE
T0_EE = simplify(T0_EE)

# Define the tool offset in the Z-axis direction
tool_length = 4.5  # Tool length in cm
T_EE_tool = Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, tool_length],
    [0, 0, 0, 1]
])

# Update the end-effector transformation with the tool offset
T0_EE_with_tool = T0_EE * T_EE_tool

# Compute cumulative transformation matrices for Jacobian calculation
T0_2 = T0_1 * T1_2
T0_3 = T0_2 * T2_3
T0_4 = T0_3 * T3_4
T0_5 = T0_4 * T4_5

# Extract position vectors (origins) and z-axis vectors for Jacobian
O0 = Matrix([0, 0, 0])
O1 = T0_1[:3, 3]
O2 = T0_2[:3, 3]
O3 = T0_3[:3, 3]
O4 = T0_4[:3, 3]
O5 = T0_5[:3, 3]
O6 = T0_EE_with_tool[:3, 3]

Z0 = Matrix([0, 0, 1])
Z1 = T0_1[:3, 2]
Z2 = T0_2[:3, 2]
Z3 = T0_3[:3, 2]
Z4 = T0_4[:3, 2]
Z5 = T0_5[:3, 2]

# Compute linear and angular velocity parts of the Jacobian matrix
Jv1 = Z0.cross(O6 - O0)
Jv2 = Z1.cross(O6 - O1)
Jv3 = Z2.cross(O6 - O2)
Jv4 = Z3.cross(O6 - O3)
Jv5 = Z4.cross(O6 - O4)
Jv6 = Z5.cross(O6 - O5)

Jw1 = Z0
Jw2 = Z1
Jw3 = Z2
Jw4 = Z3
Jw5 = Z4
Jw6 = Z5

# Combine linear and angular components into the Jacobian matrix
Jacobian_matrix = Matrix.vstack(
    Matrix.hstack(Jv1, Jv2, Jv3, Jv4, Jv5, Jv6),
    Matrix.hstack(Jw1, Jw2, Jw3, Jw4, Jw5, Jw6)
)

for i in range(Jacobian_matrix.shape[0]):  # Iterate over the rows
    print(f"\033[1mRow {i+1}:\033[0m")
    pprint(Jacobian_matrix.row(i))

# Function to compute the full Jacobian given current joint angles
def jacobian(q):
    J_full = Jacobian_matrix.evalf(subs={
        theta1: q[0],
        theta2: q[1],
        theta3: q[2],
        theta4: q[3],
        theta5: q[4],
        theta6: q[5]
    })
    return np.array(J_full).astype(np.float64)

# Function to compute joint velocities given end-effector velocity
def compute_joint_velocities(q, v_ee):
    J = jacobian(q)  # Use full 6x6 Jacobian for 6D motion
    
    J_inv = np.linalg.pinv(J + np.eye(J.shape[0]) * 1e-6)
    return J_inv.dot(v_ee)

def compute_joint_accelerations(q, q_dot, a_ee, dt):
    J = jacobian(q)  # Compute the current Jacobian
    J_inv = np.linalg.pinv(J + np.eye(J.shape[0]) * 1e-6)
    
    # Compute Jacobian derivative (J_dot) using finite differences
    J_next = jacobian(q + q_dot * dt)
    J_dot = (J_next - J) / dt
    
    # Compute joint accelerations
    q_ddot = J_inv.dot(a_ee - J_dot.dot(q_dot))
    
    return q_ddot

# Generate a semi-circular trajectory in the counterclockwise direction with center (5, 5)
def generate_semi_circle_trajectory(radius, steps, time, center=(5, 5)):
    angles = np.linspace(0, np.pi, steps)
    x_vals = center[0] + radius * np.cos(angles)
    y_vals = center[1] + radius * np.sin(angles)
    v_vals = np.vstack([np.gradient(x_vals, time/steps), np.gradient(y_vals, time/steps), np.zeros(steps)])
    return np.array([x_vals, y_vals]).T, v_vals.T

# Generate a straight-line trajectory in the negative Y-direction
def generate_straight_line_y(start_point, length, steps, time):
    y_vals = np.linspace(start_point[1], start_point[1] - length, steps)  # Move in negative y-direction
    x_vals = np.full(steps, start_point[0])  # Keep x constant
    v_vals = np.vstack([np.zeros(steps), np.gradient(y_vals, time/steps), np.zeros(steps)])  # No change in x or z
    return np.array([x_vals, y_vals]).T, v_vals.T

# Generate a straight-line trajectory in the positive x-direction
def generate_straight_line_x(start_point, length, steps, time):
    x_vals = np.linspace(start_point[0], start_point[0] + length, steps)  # Move in positive x-direction
    y_vals = np.full(steps, start_point[1])  # Keep y constant
    v_vals = np.vstack([np.gradient(x_vals, time/steps), np.zeros(steps), np.zeros(steps)])  # No change in y or z
    return np.array([x_vals, y_vals]).T, v_vals.T

# Generate a straight-line trajectory in the positive y-direction
def generate_straight_line_y_pos(start_point, length, steps, time):
    y_vals = np.linspace(start_point[1], start_point[1] + length, steps)  # Move in positive y-direction
    x_vals = np.full(steps, start_point[0])  # Keep x constant
    v_vals = np.vstack([np.zeros(steps), np.gradient(y_vals, time/steps), np.zeros(steps)])  # No change in x or z
    return np.array([x_vals, y_vals]).T, v_vals.T


def generate_combined_trajectory(radius, length_y, length_x, steps, time):
    # Generate the semi-circle trajectory
    pos_circle, vel_circle = generate_semi_circle_trajectory(radius, steps, time)
    
    # Generate the straight line in the negative Y-direction
    pos_line_y, vel_line_y = generate_straight_line_y(pos_circle[-1], length_y, steps, time)
    
    # Generate the straight line in the positive X-direction
    pos_line_x, vel_line_x = generate_straight_line_x(pos_line_y[-1], length_x, steps, time)
    
    # Generate the straight line in the positive Y-direction
    pos_line_y_pos, vel_line_y_pos = generate_straight_line_y_pos(pos_line_x[-1], length_y, steps, time)
    
    # Combine all trajectories
    positions = np.vstack([pos_circle, pos_line_y[1:], pos_line_x[1:], pos_line_y_pos[1:]])
    velocities = np.vstack([vel_circle, vel_line_y[1:], vel_line_x[1:], vel_line_y_pos[1:]])
    
    # Calculate accelerations
    dt = time / steps
    accelerations = np.zeros_like(velocities)
    accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)
    accelerations[0] = (velocities[1] - velocities[0]) / dt
    accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
    
    return positions, velocities, accelerations


#Data reference for UR3 robot :https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
masses = np.array([2.0 , 3.42 , 1.26, 0.8, 0.8, 0.35], dtype=np.float64)
com = np.array([[0, -0.02, 0], [0.13, 0, 0.1157], [0.05, 0, 0.0238], [0, 0, 0.01], [0, 0, 0.01], [0, 0, -0.02]], dtype=np.float64)
F_tool = np.array([0,0,5.0,0,0,0], dtype=np.float64)


def compute_mass_matrix(q, masses, coms):
    M = np.zeros((6, 6))
    for i in range(6):
        J_i = jacobian(q)   # Full 6x6 Jacobian
        M_i = np.zeros((6, 6))
        M_i[:3, :3] = np.eye(3) * masses[i]  # Mass for linear components
        # Simplified inertia tensor (you may want to use actual inertia tensors for more accuracy)
        M_i[3:, 3:] = np.eye(3) * (masses[i] * np.sum(coms[i]**2))  
        M += J_i.T @ M_i @ J_i
    return M

# Modify the gravity matrix calculation as well
def compute_gravity_matrix(q, masses, coms):
    g = np.array([0, 0, -9.81])  # Gravity vector
    G = np.zeros(6)
    J = jacobian(q)
    for i in range(6):
        J_i = J[:3, i]  # Linear part of the Jacobian for the i-th joint
        G[i] = masses[i] * g.dot(J_i)
    return G

# The potential energy calculation remains the same
def compute_potential_energy(q, masses, coms):
    G = compute_gravity_matrix(q, masses, coms)
    print("Gravity Matrix: \n", G)
    return -q.dot(G)  # Negative because G is defined with opposite sign


def compute_mass_matrix_derivative(q, epsilon=1e-6):
    n = len(q)
    dM = np.zeros((n, n, n))
    
    for k in range(n):
        q_plus = q.copy()
        q_plus[k] += epsilon
        q_minus = q.copy()
        q_minus[k] -= epsilon
        M_plus = compute_mass_matrix(q_plus,  masses, com)
        M_minus = compute_mass_matrix(q_minus, masses, com)
        
        dM[:, :, k] = (M_plus - M_minus) / (2 * epsilon)
    
    return dM

def compute_coriolis_matrix(q, q_dot):
    n = len(q)
    C = np.zeros((n, n))
    
    # Compute the derivative of M with respect to q
    dM = compute_mass_matrix_derivative(q)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[k, j, i]) * q_dot[k]
    
    return C


def compute_torque(q, q_dot, q_dot_dot, F):
    M = compute_mass_matrix(q, masses, com)
    g = compute_gravity_matrix(q, masses, com)  # This function needs to be implemented
    C = compute_coriolis_matrix(q, q_dot)  # This function needs to be implemented
    J = jacobian(q)
    torque = M.dot(q_dot_dot) + C.dot(q_dot) + g - J.T.dot(F)
    return torque


# Total time for trajectory
T_total = 200  # seconds
steps = 50  # Number of trajectory steps for smoothness
dt = T_total / steps  # Time step
radius = 0.05  # Radius of semicircle in cm

# q_init = np.array([-pi/4, pi/5, pi/4, pi/6, pi/4, pi/6])
q_init = np.array([pi/6, pi/6, pi/6, pi/6, pi/6, 0])

# Generate the semi-circle trajectory with center (5, 5)
target_positions, target_velocities, target_accelerations = generate_combined_trajectory(radius, 0.05, 0.10, steps, T_total)

print("dim of position",len(target_positions))
print("dim of velocity",len(target_velocities))
print("dim of velocity",len(target_accelerations))

num_points = len(target_positions)  # Total number of points in the combined trajectory
# Create a time array with 3997 points, evenly spaced from 0 to T_total
time_array = np.linspace(0, T_total*4, num_points)
# Store joint angles and end-effector positions
joint_angles_combined = [q_init]
end_effector_positions_combined = []
joint_torque_combined = []


# Loop through each time step for the combined trajectory
for t in range(len(target_positions)):
    q_current = joint_angles_combined[-1]
    v_ee = np.concatenate((target_velocities[t], [0, 0, 0]))  # 6D velocity with no rotation (angular velocity is 0)
    # Compute joint velocities using inverse kinematics
    joint_velocities = compute_joint_velocities(q_current, v_ee)
    a_ee = np.concatenate((target_accelerations[t], [0, 0, 0])) 
    # Compute joint accelerations using inverse dynamics
    joint_acceleration = compute_joint_accelerations(q_current, joint_velocities, a_ee, dt)
    #Compute the Joint Torque as per the robot dynamics model
    joint_torque =  compute_torque(q_current, joint_velocities , joint_acceleration, F_tool)
    joint_torque_combined.append(joint_torque)
    # Integrate joint velocities to get joint angles
    # s  = s0 + u *t + 1/2 * a * t**2
    q_new = q_current + joint_velocities * dt + 0.5 * joint_acceleration * dt **2
    joint_angles_combined.append(q_new)
    print("joint Torque\n", joint_torque)

    # Calculate the end-effector position for the current joint configuration
    T = T0_EE_with_tool.evalf(subs={theta1: q_new[0], theta2: q_new[1], theta3: q_new[2], theta4: q_new[3], theta5: q_new[4], theta6: q_new[5]})
    end_effector_position = T[:3, 3]
    end_effector_positions_combined.append(np.array(end_effector_position).flatten())

# Convert end-effector positions to numpy array for plotting
end_effector_positions_combined = np.array(end_effector_positions_combined)
joint_torque_combined = np.array(joint_torque_combined)
# Plot the end-effector trajectory in the XY plane
plt.figure(figsize=(12, 6))
joint_labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for i in range(6):
    plt.plot(time_array, joint_torque_combined[:, i], label=joint_labels[i], color=colors[i])

plt.xlabel('Time (sec)')
plt.ylabel('Torque (Nm)')
plt.title('Joint Torque Plot')
plt.legend(loc='center right')
plt.grid(True)
plt.show()

print("Joint Torques combined: \n", joint_torque_combined)



