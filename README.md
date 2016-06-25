# vessel-trajectory-modeller
Trajectory Modelling for Vessel Tracking and Collision Avoidance
- Author: A0105591J Xing Yifan
- Purely Python
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
    - To specifiy a maximum number of .csv files copied, type a number at the second time asking for input (or press enter for searching for all .csv files of that {vessel_type})
    - a folder structure of {vessel_type}/input/*.csv will be created
  
  - run readCSV.py
    - input the {vessel_type} from previous step as root_folder asked
    - the raw {IMO}.csv will be cleaned
    - a sub folder under {vessel_type}/cleanData/*.csv will be created
