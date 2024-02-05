import cv2

# open the video capture
cap = cv2.VideoCapture('IMG_3827.MOV') # replace with filename of your video

# create windows for both original and edge detection
# the edge detection window will be helpful in debugging
cv2.namedWindow('circumference detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('edge detection', cv2.WINDOW_NORMAL)

# set minimum and maximum radius values
min_radius = 350
max_radius = 400

# before the loop
avg_center = None
avg_radius = None

while True:
    # read a frame from the video
    ret, frame = cap.read()

    # break the loop if the video has ended
    # rets a variable we assign the video to
    if not ret:
        break

    # convert the video to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply various blurs
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 10, 50)

    # contour for circle detection
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # use more permissive criteria

    # filter contours based on area
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # sort contours by area and get the largest contour
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    # conditional to identify the largest contour
    if valid_contours:
        largest_contour = valid_contours[0]

        # fit a circle to the largest contour
        center, radius = cv2.minEnclosingCircle(largest_contour)
        center = tuple(map(int, center))

        # check if the radius is within the specified range
        if min_radius <= radius <= max_radius:
            # update the average values
            if avg_center is None:
                avg_center = center
                avg_radius = radius
            else:
                alpha = 0.7  # smoothing factor, helps for reducing flickering
                avg_center = tuple(int((1 - alpha) * avg + alpha * cur) for avg, cur in zip(avg_center, center))
                avg_radius = (1 - alpha) * avg_radius + alpha * radius

            # draw the circle on the original frame using the averaged values
            # also for thickness values
            thickness_circle = 40  
            thickness_dot = 40  
            cv2.circle(frame, tuple(map(int, avg_center)), int(avg_radius), (0, 255, 0), thickness_circle)
            cv2.circle(frame, tuple(map(int, avg_center)), 5, (0, 0, 255), thickness_dot)

    # display the frames in separate windows
    cv2.imshow('circumference detection', frame)
    cv2.imshow('edge detection', edges)

    # break the loop if 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
