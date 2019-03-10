## Baseline

#### System description
The baseline system is based on the idea of last year winner: JiaKai, Lu [1].
He provided his system so the parameter of this system can be closed to the one he used last year.
The implementation of Mean teacher model is based on Tarvainen & Valpola from Curious AI [work](https://github.com/CuriousAI/mean-teacher) [2]. 

In this baseline you can run 4 models, they all use the same CRNN base:
- Single CRNN, with synthetic data only.
- Single CRNN with weak labeled and synthetic data.
- Mean teacher model, using unlabeled and weak labeled data (like [1])
- Mean teacher model, using unlabeled, weak labeled and synthetic data.


Parameters of the CRNN model can be found in `config.py`.
The system is trained for 100 epochs.



Inputs are 500 frames long, each of them labeled identically following clip labels.
The model outputs a decision for each frame. 
Postrocessing (median filtering) of 5 frames is used to obtain events onset and offset for each file.
The baseline system includes evaluations of results using **event-based F-score** as metric. 

#### Script description
The baseline is a mean-teacher model

The baseline system is a semi supervised approach:
 - Download the data (only the first time)
 - First pass at clip level:
    - Train a CRNN on weak data (`train/weak`) - *20% of data used for validation*
    - Predict unlabel (in domain) data (`train/unlabel_in_domain`)
 - Second pass at frame level:
    - Train a CRNN on predicted unlabel data from the first pass (`train/unlabel_in_domain`) - *weak data (`train/weak`)
    is used for validation*
    *Note: labels are used at frames level but annotations are at clip level, so if an event is present in the 10 sec, 
    all frames contain this label during training*
    - Predict strong test labels (`test/`) *Note: predict an event with an onset and offset*
 - Evaluate the model between test annotations and second pass predictions (Metric is (macro-average) [event based](http://tut-arg.github.io/sed_eval/sound_event.html#event-based))

System performance (event-based measures with a 200ms collar on onsets and a 200ms / 20% of the events length collar on offsets):
 <table class="table table-striped">
 <thead>
 <tr>
 <td colspan="2">F-score metrics (macro averaged)</td> 
 <td>Simple CRNN (synthetic)</td>
 <td>Simple CRNN (synthetic + weak)</td>
 <td>Mean Teacher (no synthetic)</td>
 <td>Mean Teacher (with synthetic)</td>
 </tr>
 </thead>
 <tbody>
 <tr>
 <td><strong>Event-based</strong></td>
 <td>14.68 %</td>
 <td>14.68 %</td>
 <td>14.68 %</td>
 <td>14.68 %</td>
 </tr>
 <tr>
 <td>Segment-based</td>
 <td>14.68 %</td>
 <td>14.68 %</td>
 <td>14.68 %</td>
 <td>14.68 %</td>
 </tr>
 </tbody>
 </table>

**Note:** The performance might not be exactly reproducible on a GPU based system. 
That is why, you can found the weights of the experiments at this adress: (launch TestModel.py --model_path="Pathofmodel" to get the results)

### References
 - [1] JiaKai, Lu: Mean teacher convolution system for dcase 2018 task 4. DCASE 2018 Challenge report. September, 2018. 
 - [2] Tarvainen, A. and Valpola, H., 2017. 
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. 
 In Advances in neural information processing systems (pp. 1195-1204).