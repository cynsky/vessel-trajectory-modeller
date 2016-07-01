# vessel-trajectory-modeller
Trajectory Modelling for Vessel Tracking and Collision Avoidance
- Author: Xing Yifan (A0105591J)
- Vessel-Trajectory-Modeller is a python machine learning package which receives large vessel maneuvering datasets from Automated Identification System ([AIS](http://catb.org/gpsd/AIVDM.html#_types_1_2_and_3_position_report_class_a)). It makes use of Machine learning techniques for trajectory clustering and it extracts minumum maneuvering distance between pairs of vessels for modelling the interation among the different vessels. **Please cite the papers listed at the end if you use this code**

- Setup Module Requirements:
  - Python
  - Numpy
  
    ```
    pip install numpy
    ```
  - Scipy
  
    ```
    pip install scipy
    ```
  - [Scikit Learn](http://scikit-learn.org/stable/install.html)
  
    ```
    pip install -U scikit-learn    
    ```
  - matplotlib
  
    ```
    pip install matplotlib
    ```

- Usage:
  - run inputDataSearcher.py
    - To search for raw {IMO}.csv files from the dataSource/dynamic folder and create a sub folder structure with the {vessel_type} input
      - dataSource/dynamic is excluded in the repository, contact [xingyifan@u.nus.edu](xingyifan@u.nus.edu) for the complete files
    - To specifiy a maximum number of .csv files copied, type a number at the second time asking for input (or press enter for searching for all .csv files of that {vessel_type})
    - a folder structure of {vessel_type}/input/*.csv will be created
  
  - run readCSV.py
    - input the {vessel_type} from previous step as root_folder asked
    - the raw {IMO}.csv will be cleaned
    - a sub folder under {vessel_type}/cleanData/*.csv will be created
  
  - run trajectory_modeller.py
    - input the {vessel_type} from previous step as root_folder asked
    - choose 'y' to 'Need to compute min_distance_matrix for vessel interaction? (y/n)' if min distance matrix which contains information on vessel interaction is not yet computed
      - min distance matrix is a {n} by {n} matrix for {n} vessels, where each entry in the matrix is the minimum maneuvering distance that the two vessels will keep from each other
    - trajectory_modeller will continue with endpoint extraction, trajectory formation, trajectory interpolation and trajectory clustering
    - Final subfolder structure will be as the following:
      - {vessel_type}/
      
        |___ cleanedData/

            - Contains the cleaned {IMO}.csv files after cleaning errorneous points from the raw input
            |___ {IMO}.csv 

        |___ endPoints/
        
            - Contains the learned endpoints for each vessel data file
            |___ {IMO}_endpoints.csv
        
        |___ input/
        
              - Contains the raw input from AIS
              |___ {IMO}.csv 
        
        |___ trajectories/
        
              - Contains the trajectory learned for each vessel data file, from one endpoint to another.
              |___ {IMO}_trajectory_endpoint_{endpoint index}_to_{endpoint index}.csv 
       
    - Compiled learning results will be found in ./{vessel_type}LearningResult/ (Includes the following files)
      - protocol_trajectories_with_cluster_size.csv
        - A list of trajectory patterns learned stored with their corresponding width (pattern size)
      - endpoints_to_protocol_trajectories.csv
        - A dictionary of [Endpoint: Possible trajectories that start from this endpoint]
      - mmsi_list.csv
        - The list of MMSI identifier (unique indentifier) for all the vessels involved in the learning procedure
      - vessel_min_distance_matrix.csv
        - The learned minimum maneuvering distance matrix between pairs of vessels
      - vessel_speed_to_distance.csv
        - A dictionary of [A pair of vessels' maneuvering relative speed: A pair of vessels' maneuvering distance]
      

- Please cite this paper if you use this code
  ```
  Pedrielli, G., Xing, Y., Peh, J.H., and and Ng, S.H. A Real Time Simulation Optimization Framework for Vessel Collision avoidance and the case of Singapore Strait. Working paper, 2016.
  ```
  
