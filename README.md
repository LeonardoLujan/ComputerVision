Billiards v2 (changes in red)
thumb_40.png

Your task in this project is to locate the balls on top of the billiards table and distinguish between solid and striped balls (green and red circles above). The entire processing of one image must occur in less than 10 seconds.

Running your solution
Your program must be named "project3.py" and should receive one command-line argument indicating the path to the image to be processed. Example:

$ python project3.py /path/to/image.png
Output format
Your program must print the result to the standard output. The output starts with an integer N indicating the number of balls found in the input image. 
Follow N lines, each of them with four integers X, Y, R, and V separated by spaces. X and Y indicate the coordinates of the center of the ball in the image (0 ≤ X < width, 0 ≤ Y < height), 
R indicates the radius of the ball, and V indicates the value of the type of the ball (1 for striped balls and 0 for solid balls). The order of the balls does not matter.
