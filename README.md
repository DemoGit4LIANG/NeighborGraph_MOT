# NeighborGraph_MOT
Codes for the paper "Enhancing the association in multi-object tracking via neighbor graph"  
使用GCN增强多目标跟踪中的外观特征建模和数据关联


##### Step 1: Pre-train the __NeighborGraph__ model  
##### Step 2: Upgrade the __FairMOT__ tracker (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking) with the files in "tracker_parts" as follows:  
        gcn => FairMOT/src/lib/models/  
        multitracker.py => FairMOT/src/lib/tracker/  
        track.py => FairMOT/src/  
        track_half.py => FairMOT/src/  
		
![Alt text](https://raw.githubusercontent.com/DemoGit4LIANG/NeighborGraph_MOT/main/Screenshots/main.png)