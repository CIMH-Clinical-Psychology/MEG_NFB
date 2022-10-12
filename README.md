# MEG Neurofeedback

MEG-NFB is a project to attempt a closed loop neurofeedback system including MaxFiltering and Beamforming.

Currently, the server side is using a FieldTripBuffer (and possibly also LSL) to send data to MNE-Python for further processing. 

![MNE_NFB_UML.drawio.png](C:\Users\Simon\Desktop\MEG_NFB\assets\2c2e31928e0111eb455f7711d79bede35ebeac6a.png)

### 1. Installation

Currently, there are no extended installation instructions yet. Generally, for the FieldTripBuffer you need an active MATLAB installation. For the client, you need Python>=3.8 and a couple of packages installed.

```python
pip install mne mne_realtime pylsl
```

### 2. Getting started

To run the prototype you need to perform the following steps



Ziel: Alpha messen, 0.5s chunks are important, movement correction on 1 second chunks



### 3. Steps



1. Obtain MPRAGE / MRI image
2. Calculate BEM
3. Wait 24 hours (de-magnetization)
4. Put subject in MEG
5. calculate coregistration while participant is holding still
6. Compute forward solution
7. start task
8. maxfiltering/headpos correction -> apply beamforming -> get source