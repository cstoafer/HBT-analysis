# HBT-analysis
Tools for analyzing and visualizing signals from plasma diagnostic data on HBT-EP

The purpose of the class shotData is to easily access signals from the HBT-EP experiment.  The class takes a shot number and start and end times as arguments to create an instance.  An instance has many attributes and methods for getting and plotting commin signals from the plasma shot.

The following is an example of using the class to plot the major radius of shot 88794:

```python
import matplotlib.pyplot as plt
from shotData import shotData

shot_data = shotData(shotno=88794, startTime=0, endTime=8)

shot_data.MR.plot()
plt.show()
```

or with axes:

```python
import matplotlib.pyplot as plt
from shotData import shotData

fig = plt.figure()
MR_ax = fig.add_subplot(111)

shot_data = shotData(shotno=88794, startTime=0, endTime=8)

shot_data.MR.plot(ax = MR_ax)
plt.show()
```

An arbitrary signal node from the MDSplus tree can also be plotted:

```python
import matplotlib.pyplot as plt
from shotData import shotData

shot_data = shotData(shotno=88794, startTime=0, endTime=8)

shot_data.getSignal('.sensors.bias_probe:voltage').plot()
plt.show()
```

The file example.py contains a full example of using shotData to plot various plasma parameters.

Overview of files
-----------------

* shotData.py - Contains shotData class to easily access and plot data from HBT shots
* signalAnalysis.py - Contains common functions for analyzing HBT data
* plotSettings.py - Easily setup matplotlib environment and adjust plot and axes settings
* BandPassFilter.py - Functions written by another student, which are dependencies for other files
* example.py - Example for using these packages
* hbtplot.py - Visualization code for plotting shot data while running HBT-EP

Planned Updates
---------------

* Make easily available on spitzer server in the system path
* Easily allow legends to be included with shot numbers
* Automated configurations for axes of common parameters (i.e. set y-limit from 88cm to 96cm for major radius)
* Include n=1 quadrature plotting for magnetic sensors
