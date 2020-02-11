# DCASE2019 task4: Sound event detection in domestic environments (DESED dataset and baseline)

You can find discussion about the dcase challenge here: [dcase_discussions](https://groups.google.com/forum/#!forum/dcase-discussions). For more information about the DCASE 2019 challenge please visit the challenge [website](http://dcase.community/challenge2019/).

This task follows [dcase2018 task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection), you can find an analysis of dcase2018 task4 results [here](https://turpaultn.github.io/dcase2018-results/).

Detailed information about the baseline can be found on the dedicated [baseline page](baseline).

**If you use the dataset or the baseline, please cite [this paper](https://hal.inria.fr/hal-02160855).**

## Updates
**17th January 2020: adding public evaluation set, link to Desed, and change the format from csv to tsv to match Desed dataset.**

6th march 2019: [baseline] add baseline/Logger.py, update baseline/config.py and update README to send csv files.

**2nd May 2019: Removing duplicates in dataset/validation/test_dcase2018.csv and dataset/validation/validation.csv, changing eventbased results of 0.03%**

**19th May 2019: Updated the eval_dcase2018.csv and validation.csv. Problem due to annotation export. Files with empty annotations did have annotations.**

28th May 2019: Updated evaluation dataset 2019.

**31st May 2019: Update link to evaluation dataset (tar.gz) because of compression problem on some OS.**

30th June 2019: [baseline] Update get_predictions (+refactor) to get directly predictions in seconds.

## Dependencies

Python >= 3.6, pytorch >= 1.0, cudatoolkit=9.0, pandas >= 0.24.1, scipy >= 1.2.1, pysoundfile >= 0.10.2,
librosa >= 0.6.3, youtube-dl >= 2019.4.30, tqdm >= 4.31.1, ffmpeg >= 4.1, dcase_util >= 0.2.5, sed-eval >= 0.2.1

A simplified installation procedure example is provide below for python 3.6 based Anconda distribution for Linux based system:
1. [install Ananconda](https://www.anaconda.com/download/)
2. launch conda_create_environment.sh`

**Note:** `The baseline and download script have been tested with python 3.6, on linux (CentOS 7)`

## DESED Dataset
The Domestic Environment Sound Event Detection (DESED) dataset is composed of two subset that can be downloaded independently:

1. (Real recordings) launch `python download_data.py` (in `baseline/` folder).
2. (Synthetic clips) download at : [synthetic_dataset](https://doi.org/10.5281/zenodo.2583796).
3. **(Public evaluation set: Youtube subset)** download at: [evaluation dataset](https://zenodo.org/record/3588172).
It contains 692 Youtube files. 
4. Synthetic evaluation set: Find information here to download them: [Desed repo](https://github.com/turpaultn/DESED)

**It is likely that you'll have download issues with the real recordings.
Don't hesitate to relaunch `download_data.py` once or twice.
At the end of the download, please send a mail with the CSV files
created in the `missing_files` directory.** (in priority to Nicolas Turpault and Romain Serizel)


You should have a development set structured in the following manner:

```
dataset root
└───metadata			              (directories containing the annotations files)
│   │
│   └───train			              (annotations for the training sets)
│   │     weak.tsv                    (weakly labeled training set list and annotations)
│   │     unlabel_in_domain.tsv       (unlabeled in domain training set list)
│   │     synthetic.tsv               (synthetic data training set list and annotations)
│   │
│   └───validation			          (annotations for the test set)
│   │     validation.tsv                (validation set list with strong labels)
│   │     test_2018.tsv                  (test set list with strong labels - DCASE 2018)
│   │     eval_2018.tsv                (eval set list with strong labels - DCASE 2018)
│   │
│   └───eval			              (annotations for the public eval set (Youtube in papers))
│         public.tsv  
└───audio					          (directories where the audio files will be downloaded)
    └───train			              (audio files for the training sets)
    │   └───weak                      (weakly labeled training set)
    │   └───unlabel_in_domain         (unlabeled in domain training set)
    │   └───synthetic                 (synthetic data training set)
    │
    └───validation			                 
    └───eval		
        └───public                            
```

#### Synthetic data (1.8Gb)
Freesound dataset [1,2]: A subset of [FSD](https://datasets.freesound.org/fsd/) is used as foreground sound events for the synthetic subset of the DESED dataset. FSD is a large-scale, general-purpose audio dataset composed of Freesound content annotated with labels from the AudioSet Ontology [3].

SINS dataset [4]: The derivative of the SINS dataset used for DCASE2018 task 5 is used as background for the synthetic subset of the dataset for DCASE 2019 task 4.
The SINS dataset contains a continuous recording of one person living in a vacation home over a period of one week.
It was collected using a network of 13 microphone arrays distributed over the entire home.
The microphone array consists of 4 linearly arranged microphones.

The synthetic set is composed of 10 sec audio clips generated with [Scaper](https://github.com/justinsalamon/scaper) [5].
The foreground events are obtained from FSD.
Each event audio clip was verified manually to ensure that the sound quality and the event-to-background ratio were sufficient to be used an isolated event. We also verified that the event was actually dominant in the clip and we controlled if the event onset and offset are present in the clip. Each selected clip was then segmented when needed to remove silences before and after the event and between events when the file contained multiple occurrences of the event class.


##### License
All sounds comming from FSD are released under Creative Commons licences.
**Synthetic sounds can only be used for competition purposes until the full CC license list is made available at the end of the competition.**


#### Real recordings (23.4Gb):
Subset of [Audioset](https://research.google.com/audioset/index.html) [3].
Audioset: Real recordings are extracted from Audioset. It consists of an expanding ontology of 632 sound event classes and a collection of 2 million human-labeled 10-second sound clips (less than 21% are shorter than 10-seconds) drawn from 2 million Youtube videos. The ontology is specified as a hierarchical graph of event categories, covering a wide range of human and animal sounds, musical instruments and genres, and common everyday environmental sounds.

The download/extraction process can take approximately 4 hours.
If you experience problems during the download of this subset please contact the task organizers.

### Annotation format

#### Weak annotations
The weak annotations have been verified manually for a small subset of the training set. 
The weak annotations are provided in a tab separated csv file (.tsv) under the following format:

```
[filename (string)][tab][event_labels (strings)]
```
For example:
```
Y-BJNMHMZDcU_50.000_60.000.wav	Alarm_bell_ringing,Dog
```

#### Strong annotations
Synthetic subset and validation set have strong annotations.

The minimum length for an event is 250ms. The minimum duration of the pause between two events from the same class is 150ms. When the silence between two consecutive events from the same class was less than 150ms the events have been merged to a single event.
The strong annotations are provided in a tab separated csv file (.tsv) under the following format:

```
[filename (string)][tab][event onset time in seconds (float)][tab][event offset time in seconds (float)][tab][event_label (strings)]
```
For example:

```
YOTsn73eqbfc_10.000_20.000.wav	0.163	0.665	Alarm_bell_ringing
```

# Description

This task is the follow-up to [DCASE 2018 task 4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection).
The task evaluates systems for the large-scale detection of sound events using weakly labeled data (without timestamps).
The target of the systems is to provide **not only the event class but also the event time boundaries** given that multiple events can be present in an audio recording.
The challenge of exploring the possibility to **exploit a large amount of unbalanced and unlabeled training data** together with a small weakly annotated training set to improve system performance remains but an **additional training set with strongly annotated synthetic data** is provided.
**The labels in all the annotated subsets are verified and can be considered as reliable.**  An additional scientific question this task is aiming to investigate is whether we really need real but partially and weakly annotated data or is using synthetic data sufficient? or do we need both?

Further information on [dcase_website](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments)

You can find the detailed results of dcase2018 task 4 to this [page](https://turpaultn.github.io/dcase2018-results/) and this [paper](https://hal.inria.fr/hal-02114652v2)[6].

## Authors

Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019 -- Present

## References

- [1] F. Font, G. Roma & X. Serra. Freesound technical demo. In Proceedings of the 21st ACM international conference on Multimedia. ACM, 2013.
- [2] E. Fonseca, J. Pons, X. Favory, F. Font, D. Bogdanov, A. Ferraro, S. Oramas, A. Porter & X. Serra. Freesound Datasets: A Platform for the Creation of Open Audio Datasets.
In Proceedings of the 18th International Society for Music Information Retrieval Conference, Suzhou, China, 2017.

- [3] Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter.
Audio Set: An ontology and human-labeled dataset for audio events.
In Proceedings IEEE ICASSP 2017, New Orleans, LA, 2017.

- [4] Gert Dekkers, Steven Lauwereins, Bart Thoen, Mulu Weldegebreal Adhana, Henk Brouckxon, Toon van Waterschoot, Bart Vanrumste, Marian Verhelst, and Peter Karsmakers.
The SINS database for detection of daily activities in a home environment using an acoustic sensor network.
In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2017 Workshop (DCASE2017), 32–36. November 2017.

- [5] J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. Scaper: A library for soundscape synthesis and augmentation
In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017.

- [6] Romain Serizel, Nicolas Turpault. 
Sound Event Detection from Partially Annotated Data: Trends and Challenges. 
IcETRAN conference, Srebrno Jezero, Serbia, June 2019.
