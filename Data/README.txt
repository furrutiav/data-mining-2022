Semeval 2018 Task 2 - Multilingual Emoji Prediction
https://competitions.codalab.org/competitions/17344

The task consists of detecting one of the 20 emoji labels (19 in Spanish due to a previous issue). The results of the participant systems can be found in the folder "results", or on the reference paper.
The official ranking was computed using macro f1 (sklearn implementation).

There are five subdirectories: train, test, and trial (the latter may be used as development), mapping (mapping each emoji to a number between 0 and 20) and results.
Test and Trial tweets along with their corresponding labels are already available, but, due to Twitter restrictions, you will need to download the tweets of the training set yourself. It is a smooth process and all the instructions and commands can be found in the train folder.

If you use this dataset please cite the main reference paper, where you can find more information about the task:

@InProceedings{semeval2018task2,
  title={{SemEval-2018 Task 2: Multilingual Emoji Prediction}},
  author={Barbieri, Francesco and Camacho-Collados, Jose and Ronzano, Francesco and Espinosa-Anke, Luis and Ballesteros, Miguel and Basile, Valerio and Patti, Viviana and Saggion, Horacio},
  booktitle={Proceedings of the 12th International Workshop on Semantic Evaluation (SemEval-2018)},
  year={2018}, 
  address={New Orleans, LA, United States},
  publisher = {Association for Computational Linguistics}
 }