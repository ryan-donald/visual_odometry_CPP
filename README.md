This is a repository containing a visual odometry implementation written in C++ utilizing OpenCV, based on my python visual odometry pipeline.
  
The driving force behind my desire to implement visual odometry in C++, when I have already done so in Python is to improve my C++ skills.
Within Python I have a strong intuition and familiarity with the nuances of the syntax, which I would like to have the same skills with C++.
Through re-implementing the same algorithm in C++, I didn't have to worry about the entire logical framework, as I had done that previously, and could focus on how best to implement the logic in C++.

The *vo.cpp* file contains the entire visual odometry pipeline. Below is how to use the program:
- When you first run the file, it prompts for a sequence number from the KITTI dataset.
- Then, it will run through the image frames to calculate the estimated trajectory, displaying a progress bar and a number of processed frames over total frames.
- Lastly, the trajectory will save into a file in the format 'trajectory_{sequence}.txt', which is formatted the same as the ground-truth trajectories given by the KITTI dataset.
