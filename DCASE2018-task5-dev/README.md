# DCASE 2018 - Task 5, Development dataset
## Monitoring of domestic activities based on multi-channel acoustics

Authors:

- Gert Dekkers (<gert.dekkers@kuleuven.be>, <https://iiw.kuleuven.be/onderzoek/advise/People/Gert_Dekkers>)
- Peter Karsmakers (<peter.karsmakers@kuleuven.be>, <https://iiw.kuleuven.be/onderzoek/advise/People/PeterKarsmakers>)

[Advanced Integrated Sensing lab (ADVISE)  / Department of Electrical Engineering (ESAT) / KU Leuven](https://iiw.kuleuven.be/onderzoek/advise)

Other (Recording, supervision, ...):

- Steven Lauwereins
- Bart Thoen
- Mulu Weldegebreal Adhana,
- Henk Brouckxon
- Bertold Van den Bergh
- Toon van Waterschoot
- Bart Vanrumste

# Table of Contents
1. [Dataset](#1-dataset)
2. [Content](#2-content)
3. [Cross-validation setup](#3-cross-validation-setup)
4. [Changelog](#4-changelog)
5. [License](#5-license)

1. Dataset
=================================

The dataset is a derivative of the **SINS dataset**. The SINS dataset contains a continuous recording of one person living in a vacation home over a period of one week.  It was collected using a network of 13 microphone arrays distributed over the entire home. The microphone array consists of 4 linearly arranged microphones. For this dataset 4 microphone arrays in the combined living room and kitchen area are used. Figure 2 shows the floorplan of the recorded environment along with the position of the used sensor nodes.
<figure>
    <p align="center">
        <img src="http://d33wubrfki0l68.cloudfront.net/5cb3f5d055aa8916eaafa831fefea69c113cbc5c/c201d/images/tasks/challenge2018/task5_2dplan.png" class="img img-responsive" style="width: 300px" align="center">
    </p>
	<p align="center">
        <figcaption>Figure 2: 2D floorplan of the combined kitchen and living room with the used sensor nodes.</figcaption>
    </p>
</figure>
<br>
Approximately 200 hours of data from 4 sensor nodes are taken from the SINS dataset. The partitioning of the data was done randomly. The segments belonging to one particular consecutive activity (e.g. a full session of cooking) were kept together. The data provided for each sensor node contain recordings of the same time period. This means that the performed activities are observed from multiple microphone arrays at the same time instant. 

The recordings were split into audio segments of 10s. Each segment represents one activity.  These audio segments are provided as individual files along with the ground truth. 
The daily activities for this dataset (9) are shown in Table 1 along with the available 10s segments in the dataset and the amount of full sessions of a certain activity (e.g. a cooking session).

<div class="table table-responsive">
<table class="table table-striped">
    <thead>
        <tr>
            <th>Activity</th>
            <th class="col-md-3"># 10s segments</th>
            <th class="col-md-3"># sessions</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Absence (nobody present in the room)</td>
            <td>18860</td>
            <td>42</td>
        </tr>
        <tr>
            <td>Cooking</td>
            <td>5124</td>
            <td>13</td>
        </tr>
        <tr>
            <td>Dishwashing</td>
            <td>1424</td>
            <td>10</td>
        </tr>
        <tr>
            <td>Eating</td>
            <td>2308</td>
            <td>13</td>
        </tr>
        <tr>
            <td>Other (present but not doing any relevant activity)</td>
            <td>2060</td>
            <td>118</td>
        </tr>  
        <tr>
            <td>Social activity (visit, phone call)</td>
            <td>4944</td>
            <td>21</td>
        </tr>
        <tr>
            <td>Vacuum cleaning</td>
            <td>972</td>
            <td>9</td>
        </tr> 
        <tr>
            <td>Watching TV</td>
            <td>18648</td>
            <td>9</td>
        </tr>
        <tr>
            <td>Working (typing, mouse click, ...)</td>
            <td>18644</td>
            <td>33</td>
        </tr>
    </tbody>
    <tfoot>
        <tr>
            <td><strong>Total</strong></td>
            <td><strong>72984</strong></td>
            <td><strong>268</strong></td>
        </tr>
    </tfoot>
</table>
</div>
<div class="clearfix"></div>

### Recording and annotation procedure

The sensor node configuration used in this setup is a control board together with a linear microphone array. The control board contains an EFM32 ARM cortex M4 microcontroller from Silicon Labs (EFM32WG980) used for sampling the analog audio. The microphone array contains four Sonion N8AC03 MEMS low-power (±17µW) microphones with an inter-microphone distance of 5 cm. The sampling for each audio channel is done sequentially at a rate of 16 kHz with a bit depth of 12.
The annotation was performed in two phases. First, during the data collection a smartphone application was used to let the monitored person(s) annotate the activities while being recorded. The person could only select a fixed set of activities. The application was easy to use and did not significantly influence the transition between activities. Secondly, the start and stop timestamps of each activity were refined by using our own annotation software. Postprocessing and sharing the database involves privacy-related aspects. Besides the person(s) living there, multiple people visited the home. Moreover, during a phone call, one can partially hear the person on the other end. A written informed consent was obtained from all participants.

2. Content
=================================

The content of the dataset is structured in the following manner:

	dataset root
	│   EULA.pdf				End user license agreement
	│   meta.txt				meta data, tsv-format, [audio file (str)][tab][label (str)][tab][session (str)]
	│   readme.md				Dataset description (markdown)
	│   readme.html				Dataset description (HTML)
	│
	└───audio					72984 audio segments, 16-bit 16kHz
	│   │   DevNode1_ex1_1.wav	name format DevNode{NodeID}_ex{sessionID}_{segmentID}.wav
	│   │   DevNode2_ex1_2.wav
	│   │   ...
	│
	└───evaluation_setup		cross-validation setup, 4 folds
	    │   fold1_train.txt		training file list, tsv-format, [audio file (str)][tab][label (str)][tab][session (str)]
	    │   fold1_test.txt 		test file list, tsv-format, [audio file (str)][tab][label (str)]  
	    │   ...        

The multi-channel audio files can be found under directory `audio` and are formatted in the following manner:

	DevNode{NodeID}_ex{sessionID}_{segmentID}.wav

-	*{NodeID}* (1-4) is an identifier to indicate which segments belong to a specific node. In total 4 nodes are given (1-4). It is unknown what the location of the node is to the user.
-	*{sessionID}* indicates a full session of a certain activity.
-	*{segmentID}* indicates a segment belonging to a certain {sessionID}. A session of a certain activity (e.g. cooking) can have multiple 10s segments.

The file `meta.txt` and the content of the folder `evaluation_setup` contain filenames along with ground truth labels and an identifier of to which session the segment belongs. These are arranged in the following manner:
	
	[filename (str)][tab][activity label (str)][tab][session (str)]

The directory `evaluation_setup` provides cross-validation folds for the development dataset. More information on the usage can be read [here](#usage)

3. Cross-validation setup
=================================
The dataset includes multi-channel audio segments along with the ground truth and cross-validation folds.
Cross-validation folds are provided for the dataset in order to make results reported with this dataset uniform. The setup consists of four folds distributing the available files. 
Segments belonging to a particular session of an activity (e.g. a session of cooking collected by multiple sensor nodes) are kept together to minimize leakage between folds. The folds are provided with the dataset in the directory `evaluation setup`. For each fold a training, testing and evaluation subset is provided. 

#### Training

`evaluation setup\fold[1-4]_train.txt`
: training file list (in csv-format)

Format:
    
    [filename (str)][tab][activity label (str)][tab][session (str)]

#### Testing

`evaluation setup\fold[1-4]_test.txt`
: testing file list (in csv-format)

Format:
    
    [filename (str)][tab]

#### Evaluating

`evaluation setup\fold[1-4]_evaluate.txt`
: evaluation file list (in csv-format), same as fold[1-4]_test.txt but with additional reference information. These two files are provided separately to prevent contamination with ground truth when testing the system

Format: 

    [filename (str)][tab][activity label (str)]

4. Changelog
=================================
#### 1.0.3 / 2017-05-15
* Fixed an error. More information on the issue: [https://groups.google.com/forum/#!topic/dcase-discussions/iDh4M-RMy8U](https://groups.google.com/forum/#!topic/dcase-discussions/iDh4M-RMy8U).
#### 1.0.2 / 2017-04-12
* Added eval folds to folder 'evaluation_setup'
* Added README.md
#### 1.0.1 / 2017-03-30
* Fixed error in dataset. The error caused duplicate segments.
#### 1.0.0 / 2018-03-21
* Initial commit

5. License
=================================

See file [EULA.pdf](EULA.pdf)