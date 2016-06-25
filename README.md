# vessel-trajectory-modeller
Trajectory Modelling for Vessel Tracking and Collision Avoidance
- Author: Xing Yifan (A0105591J)
- Large Datasets
- Machine learning techniques used for trajectory clustering
- Min Distance Extraction between pairs of vessels
- Result Aggregation and Saving as inputs for Agent Based Simulator

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
    - choose 'y' to 'Need to compute min_distance_matrix for ABM input? (y/n)' if min distance matrix for ABM input later is not yet computed
    - trajectory_modeller will continue with endpoint extraction, trajectory formation, trajectory interpolation and trajectory clustering
    - Final subfolder structure will be as the following:
      - {vessel_type}/
      
        |___ cleanedData/

        |___ endPoints/
        
        |___ input/
        
        |___ trajectories
       
    - Required ABM inputs for the C++ ABM module will be found in ./{vessel_type}ABMInput/
