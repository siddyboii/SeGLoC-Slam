To complete the **Masking and Filtering** stage (Section IV-C) and prepare for **Tracking**, you need to implement the following logic in your `DynaTrack` node:

### 1. Temporal Data Management

* 
**Store Previous Flow:** You must store the previous linear velocity field () in a member variable, as it is required to calculate the propagation mask.


* 
**Calculate :** Ensure you have the time increment between frames (typically 0.1s for a 10Hz LiDAR) to use in the propagation equations.



### 2. Vector Field Propagation Mask ()

* 
**Warp the Velocity Field:** Calculate the predicted position of each cell by shifting the current  coordinates based on the previous linear velocities  and .


* 
**Compute Difference:** Subtract the propagated (predicted) vector field  from the actual calculated field at the current time step .


* 
**Apply Threshold :** Create a boolean mask () where cells are set to 1 only if the magnitude of the difference is below a small constant threshold .



### 3. Rigid-Body Continuity Mask ()

* 
**Linear Continuity ():** Apply a Laplacian or divergence operator to the linear velocity field to identify areas where the "rigid body" is tearing or imploding.


* 
**Angular Continuity:** Ensure all points on a single object rotate with the same angular velocity by checking the condition .


* 
**Filter Static Noise:** Use these operations to create the  mask, setting cells to zero if they do not satisfy the rigid-body motion constraints.



### 4. Mask Integration and Application

* 
**Combine Masks:** Calculate the final filtering mask by multiplying the two layers: .


* 
**Refine the Vector Field:** Multiply your original dense optical flow field by this final mask to filter out "false positive" vectors from unoccupied or static cells.



### 5. Preparation for Clustering (Next Step)

* 
**Cluster Remaining Vectors:** Once the field is masked, use Euclidean distance clustering to group the valid velocity vectors into distinct "objects".


* 
**Calculate Cluster Means:** For each cluster, compute the mean position and mean 2D velocity (linear and angular) to serve as the input "Measurement" for the EKF.
