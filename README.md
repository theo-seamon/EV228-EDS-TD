# EV228-EDS-TD
Repository for EV228 EDS. Theo & David.

**Summary:**

This project is analyzing the 2025 Flooding in Beijing by looking at precipitation data from 1951-2025 from Beijing, Chengde, and Shijiazhuang.

**Process to generate figures:**

For the figure of daily average with 2025 precipitation data you use the avg_precip.py and EDS_functions.py code. The function to import the data, calculate a mean and plot a graph is defined in avg_precip.py and is called daily_avg. To then run this function EDS_functions demonstrates how you can import the module and call the function with the file path of the data, file output, and name of graph being defined.
For the figure of Annual Total precipitation trends, please use the EVS_code_1.py, and check if scipy and cartopy are accessible. After click on the 'start running this code',  Wait for several minutes to have all the images/graphs jumps out. I'll send data in Chengde and Shijiazhuang to you through email.

**Code Index:**

yearly_precip_theo.py - defines yearly_avg function that imports time series data, creates a yearly average, and plots a graph.
avg_precip.py - defines daily_avg function that imports time series data, creates a day-of-year average, and plots a graph.
EDS_functions.py - this code calls the two functions defined above and is an example of how to run the functions to generate graphs from data.
EV_story_code_david.py - These codes were written when only have dataset from Beijing.
EVS_code_1.py - These codes were written after I found 2 more datasets from Shijiazhuang and Chengde, after all the dataset were imported I put them together in a zip and plot several graphs.

**Generative AI Statement:**

Theo: I did not use any generative AI for this assignment.

David: I used ChatGPT in the "EV_story_code_david.py" and "EVS_code_1.py" files to check for errors and to assist me in explaining some of the code I found online.
