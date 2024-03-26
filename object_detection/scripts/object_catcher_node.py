#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_srvs.srv import Empty, EmptyResponse
import numpy as np


class BallCatcher:
    """ExampleMoveItTrajectories"""

    def __init__(self):
        try:
            # Detect gripper
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                self.gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(
                rospy.get_namespace() + 'move_group/display_planned_path',
                moveit_msgs.msg.DisplayTrajectory,
                queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            # Advertise service
            rospy.Service('catch_ball', Empty, self.catch_ball)

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())

            # Related to catching the ball
            self.z_height = 0.20
            self.delta_h = 0.17

            # Starting position
            self.go_to_tracking()

        except Exception as e:
            print(e)
            self.is_init_success = False
        else:
            self.is_init_success = True


    def reach_cartesian_pose(self, pose, tolerance, constraints):
        arm_group = self.arm_group

        # Set the tolerance
        arm_group.set_goal_position_tolerance(tolerance)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)

        # Get the current Cartesian Position
        arm_group.set_pose_target(pose)

        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        return arm_group.go(wait=True)

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        rospy.loginfo("Actual cartesian pose is : ")
        rospy.loginfo(pose.pose)

        return pose.pose

    def reach_gripper_position(self, relative_position):
        # We only have to move this joint because all others are mimic!
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        try:
            val = gripper_joint.move(
                relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos,
                True)
            return val
        except:
            return False

    def go_to_tracking(self):
        # Send arm to overhead position where tracking is done
        tracked_pose = geometry_msgs.msg.PoseStamped()
        tracked_pose.header.stamp = rospy.Time.now()
        tracked_pose.header.frame_id = "base_link"

        tracked_pose.pose.position.x = 0.38
        tracked_pose.pose.position.y = 0.0
        tracked_pose.pose.position.z = 0.34

        tracked_pose.pose.orientation.x = -0.7026480050848345
        tracked_pose.pose.orientation.y = -0.7114656958752997
        tracked_pose.pose.orientation.z = -0.00539779454456196
        tracked_pose.pose.orientation.w = 0.008556188230352993

        success = self.reach_cartesian_pose(pose=tracked_pose, tolerance=0.01, constraints=None)
        return success

    def catch_ball(self, message):
        # Initialize
        success = self.is_init_success

        # Reach the vertical on top of the ball
        success = False
        while not success:
            tracked_pose = geometry_msgs.msg.PoseStamped()
            tracked_pose.header.stamp = rospy.Time.now()
            tracked_pose.header.frame_id = "base_link"

            # Get the parameter from the server
            point = rospy.get_param("/ball_center")

            tracked_pose.pose.position.x = point[0]
            tracked_pose.pose.position.y = point[1]
            tracked_pose.pose.position.z = self.z_height

            tracked_pose.pose.orientation.x = -0.7026480050848345
            tracked_pose.pose.orientation.y = -0.7114656958752997
            tracked_pose.pose.orientation.z = -0.00539779454456196
            tracked_pose.pose.orientation.w = 0.008556188230352993

            success = self.reach_cartesian_pose(pose=tracked_pose, tolerance=0.01, constraints=None)

        # Open gripper to avoid hitting ball
        success = False
        while self.is_gripper_present and not success:
           rospy.loginfo("Opening the gripper...")
           success = self.reach_gripper_position(0)

        # Move to the ball position
        success = False
        start_value = tracked_pose.pose.position.z
        while not success:
            tracked_pose.pose.position.z = start_value - self.delta_h
            tracked_pose.header.stamp = rospy.Time.now()
            success = self.reach_cartesian_pose(pose=tracked_pose, tolerance=0.01, constraints=None)

        # Close fingers
        # This will yield an error, but no matter
        if success:
            rospy.loginfo("Closing the gripper...")
            self.reach_gripper_position(0.4)

        # Liftup the ball
        success = False
        start_value = tracked_pose.pose.position.z
        while not success:
            tracked_pose.pose.position.z = start_value + self.delta_h
            tracked_pose.header.stamp = rospy.Time.now()
            success = self.reach_cartesian_pose(pose=tracked_pose, tolerance=0.01, constraints=None)

        # Release the ball
        if success:
            rospy.loginfo("Releasing ball...")
            self.reach_gripper_position(0)

        return EmptyResponse()


def main(args):
    rospy.init_node('object_catcher')
    bc = BallCatcher()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)