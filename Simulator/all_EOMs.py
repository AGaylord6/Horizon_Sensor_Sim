'''
all_EOMs.py
Authors: Michael Paulucci, Payton Reynolds, Kris, Michael Cukzun, other controls members

Mother file for all equations of motion for our nearspace systems
Returns the derivative of the next state, doesn't actually propagate the state

Also contains helper functions for quaternion and matrix math
'''

import numpy as np
import math

def eoms(quaternion: np.ndarray, w_sat: np.ndarray, w_rw: np.ndarray, tau_sat: np.ndarray, alpha_rw: np.ndarray, dt: float, I_body: np.ndarray, rw_off: float):
        '''
        Uses the Equations of Motion (EOMs) to yield the next values of quaternion and angular velocity of satellite
        The EOMs are based upon the current state, current reaction wheel speeds, and external torque being applied by the magnetorquers
        Propogates our state vector through time using physics predictions
        
        @params:
            quaternion (np.ndarray, (1x4)): quaternion describing orientation of satellite with respect to given reference frame
            w_sat (np.ndarray, (1x3)): angular velocity of whole satellite (w/ reaction wheel) (degrees/s)
            w_rw (np.ndarray, (1x3) for 1D test only): angular velocities of wheels (in respective wheel frame)
            tau_sat (np.ndarray, (1x3)): external torque applied on the satellite, including magnetorquers and disturbances
            alpha_rw (nd.ndarray, (1x3)): angular acceleration of the reaction wheels in their respective wheel frames
            dt (float): timestep
            I_body (np.ndarray, (3x3)): inertia tensor of satellite body
            rw_off (float): set to zero if using only magnetorquers, one if using reaction wheels

        @returns:
            (1x7) state vector containing quaternion and angular velocity of satellite
                quaternion_dot (np.ndarray, (1x4)): first derivative of quaternion
                w_sat (np.ndarray, (1x3)): first derivative of angular velocity of satellite
        '''

        # Store norm of w_vect as separate variable, as w_magnitude. Also separate out components of w
        w_x = w_sat[0]
        w_y = w_sat[1]
        w_z = w_sat[2]
        
        # Quaternion product matrix for angular velocity of satellite
        ### THIS IS THE BIG OMEGA that should work for our scalar-component of quaternion first notation (based on https://ahrs.readthedocs.io/en/latest/filters/angular.html)
        w_sat_skew_mat = np.array([[0, -w_x, -w_y, -w_z],
                                    [w_x, 0, w_z, -w_y],
                                    [w_y, -w_z, 0, w_x],
                                    [w_z, w_y, -w_x, 0]])

        # First derivative of quaternion
        quaternion_dot = 0.5 * np.matmul(w_sat_skew_mat, quaternion)        

        # First derivative of angular velocity
        # subtract gyroscopic term from all external torquers: inverse_I * (tau_sat - w_sat x (I * w_sat))
        # note: equivalent to multiply by w_skew_matrix instead of taking cross product
        # w_skew_matrix = np.array([[0, -w_z, w_y],
                            # [w_z, 0, -w_x],
                            # [-w_y, w_x, 0]])
        # w_sat_dot = np.matmul( np.linalg.inv(I_body), (tau_sat - np.matmul(w_skew_matrix, np.matmul(I_body, w_sat))))
        w_sat_dot = np.matmul( np.linalg.inv(I_body), (tau_sat - np.cross(w_sat, np.matmul(I_body, w_sat))))
        # NOTE: total vel magnitude steadily increases over time due to euler's method (i think). Smaller timestep = less increase

        # or just take external torque into account
        # w_sat_dot = np.matmul( np.linalg.inv(I_body), (tau_sat))

        return quaternion_dot, w_sat_dot


def normalize(v):
        # normalizes the vector v (usuallly a quaternion)
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return np.array(v / norm)


def quaternion_rotation_matrix(Q):
    '''
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    @params
        Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    @returns
        rot_matrix: A 3x3 element matrix representing the full 3D rotation matrix. 
            This rotation matrix converts a point in the local reference 
            frame to a point in the global reference frame.
    '''
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


# converts euler angles to quaternion
def eToQ(angles):
    x,y,z=[a/2 for a in angles] #dividing by 2 cause all of the trigs need it
    Q=[0]*4
    cr=math.cos(x)
    sr=math.sin(x)
    cp=math.cos(y)
    sp=math.sin(y)
    cy=math.cos(z)
    sy=math.sin(z)
    Q[0]=cr*cp*cy+sr*sp*sy
    Q[1]=sr*cp*cy-cr*sp*sy
    Q[2]=cr*sp*cy+sr*cp*sy
    Q[3]=cr*cp*sy-sr*sp*cy
    #comes out normalized
    return Q



def euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    # switch to other quaternion notation
    # rot = Rotation.from_quat([x, y, z, w])
    # return rot.as_euler('xyz', degrees=False)

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians


def delta_q(q_actual, q_target):
    '''
    delta_q
        Returns error quaternion by taking quaternion product (x)
            between actual quaternion and conjugate of target quaternion. 
        Tells us what rotation is needed to reach target

    @params
        q_actual, q_target: normalized (unit) quaternion matrices (1 x 4) [q0, q1:3]
    @returns
        error quaternion: always normalized. equals [1, 0, 0, 0] when q_actual and q_target are equal
    '''

    # because we're using unit quaternions, inverse = conjugate
    # q_actual_inverse = np.array([q_actual[0], -q_actual[1], -q_actual[2], -q_actual[3]])
    q_target_inverse = np.array([q_target[0], -q_target[1], -q_target[2], -q_target[3]])

    
    q_error = quaternionMultiply(q_actual, q_target_inverse)
    # q_error = quaternionMultiply(q_target, q_actual_inverse)

    # since a quaternion can represent 2 relative orientations, we also want to ensure that the error quaternion is the shortest path
    # from: Quaternion Attitude Control System of Highly Maneuverable Aircraft
    if q_error[0] < 0:
        # if desired rotation is > pi away, then the actual closest rotation is the inverse
        q_error = -q_error
    
    # error_range = 0.1
    # if np.linalg.norm(q_error[1:4]) < error_range:
        # TODO: if we're close enough to the target, don't waste energy on micro movements?
        #print("close enough")
        # return np.array([1, 0, 0, 0])
    # else:
        #print("error: ", q_error)
        # return q_error

    return q_error


def quaternionMultiply(a, b):
    '''
    quaternionMultiply
        custom function to perform quaternion multiply on two passed-in matrices

    @params
        a, b: quaternion matrices (4 x 1) [q0 ; q1:3]
    @returns
        multiplied quaternion matrix [q0 ; q1:3]
    '''

    return np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])
