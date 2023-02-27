The bin folder contains the libraries required to run the Kalman smoothing algorithm, make sure you add it to your computer PATH.

You can find the source code here: https://github.com/antoinefalisse/opensim-core/tree/kalman_smoother.

Please take a look at the example setup file (download the KS-example folder) to find out how to use the algorithm.

A few tips and tricks:

1. You will want to use weights between 0.5 and 2 (the larger the less uncertainty, ie the more importance you give to the marker).
2. The Kalman smoothing algorithm does not handle kinematic constraints. In practice, you might see for example the patella of your model flying around. Since the patella coordinate value is fully determined by the knee coordinate value, you should be able to calculate back the right patella coordinate values.
3. In practice, the setup file is similar to the one used for running OpenSim's IK algorithm. There are a few additional parameters you can adjust though. Please take a look at the example setup file (between lines 214 and 223).
