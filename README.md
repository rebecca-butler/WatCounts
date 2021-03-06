# WatCounts
WatCounts is a COVID-19 occupancy tracking application designed for the University of Waterloo. It uses a computer vision object tracking and detection algorithm to monitor the number of people entering and exiting a room. This information is recorded on a web app so students can plan their trips to the university and avoid areas of high occupancy.

The system is designed to run on a Raspberry Pi with a Camera Module.

## Usage
Execute the following command from the `src` folder: `python3 counter.py [-above]`

Use the optional `-above` flag to indicate that the camera is positioned above the doorway. If the flag is not given, the camera is assumed to be beside the doorway.

## Dependencies
- OpenCV
- imutils
- numpy
- picamera