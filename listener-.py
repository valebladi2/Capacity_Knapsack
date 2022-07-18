#!/usr/bin/env python
   2 import rospy
   3 from std_msgs.msg import String
   4 
   5 def callback(data):
   6     rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
   7     
   8 def listener():
   9 
  10     # In ROS, nodes are uniquely named. If two nodes with the same
  11     # name are launched, the previous one is kicked off. The
  12     # anonymous=True flag means that rospy will choose a unique
  13     # name for our 'listener' node so that multiple listeners can
  14     # run simultaneously.
  15     rospy.init_node('listener', anonymous=True)
  16 
  17     rospy.Subscriber("chatter", String, callback)
  18 
  19     # spin() simply keeps python from exiting until this node is stopped
  20     rospy.spin()

  # Path: listener-.py
   # Compare this snippet from publisher.py:
   #    1 #!/usr/bin/env python
   #    2 # license removed for brevity

   #    3 import rospy
   # cretae array to listen to multiple topics

   #    4 def callback(data):
   #    5     rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
  21 
  22 if __name__ ==