## Baseline

**Important update: 19th of May 2019, problem with annotations (export), the corrected labels are updated.** This affects:
- validation/eval_dcase2018.csv
- validation/validation.csv
The results table at the bottom have been updated.

Minor updates described in [previous fodler](../README.md). 
To include new functionnalities, do not hesitate to do a pull request.

If you use the baseline, please cite [this paper](https://hal.inria.fr/hal-02160855).

### System description
The baseline system is based on the idea of the best submission of DCASE 2018 task 4 [1]. The author provided his system code and most of the hyper-parameters of this year baseline close to the hyper-parameters defined by last year winner. However, the network architecture itself remains similar to last year baseline so it is much simpler that the networks used by Lu JiaKai [1]. The parameters of the CRNN model can be found in `config.py`.

The baseline using a mean-teacher model that is composed of two networks that are both the same CRNN. The implementation of Mean teacher model is based on Tarvainen & Valpola from [Curious AI](https://github.com/CuriousAI/mean-teacher) [2]. The model is trained as follows:
- The student model is trained on synthetic and weakly labeled data. The classification cost is computed at frame level on synthetic data and at clip level on weakly labeled data.
- The teacher model is not trained, its weights are a moving average of the student model (at each epoch).
- The inputs of the teacher model are the inputs of the student model + some Gaussian noise
- A cost for consistency between teacher and student model is applied (for weak and strong predictions).


The baseline exploit unlabeled, weakly labeled and synthetic data for training and is trained for 100 epochs. Inputs are 864 frames long. The CRNN model is pooling in time to have 108 frames.
Postrocessing (median filtering) of 5 frames is used to obtain events onset and offset for each file.
The baseline system includes evaluations of results using **event-based F-score** as metric.

### Evaluation



System performance are reported in term of event-based F-scores with a 200ms collar on onsets and a 200ms / 20% of the events length collar on offsets. Additionally, the performance in terms of segment-based F-scores on 1 sec segment are reported for information. Performance are reported on this year validation set (Validation 2019) and on the evaluation set from DCASE 2018 task 4.
 <table class="table table-striped">
 <thead>
 <tr>
 <td colspan="3">F-score metrics (macro averaged)</td>
 </tr>
 </thead>
 <tbody>
 <tr>
 <td></td>
 <td><strong>Public evaluation 2019 (Youtube)</strong></td>
  <td><strong>Validation 2019</strong></td>
 <td>Evaluation 2018</td>
 </tr>
 <tr>
 <td><strong>Event-based</strong></td>
 <td><strong> 29.0 %</strong></td>
 <td><strong>23.7 %</strong></td>
 <td>20.6 %</td>
 </tr>
 <tr>
 <td>Segment-based</td>
 <td> 58.54 %</td>
 <td>55.2 %</td>
  <td>51.4 %</td>
 </tr>
 </tbody>
 </table>

**Note:** The performance might not be exactly reproducible on a GPU based system.
That is why, you can download the [weights of the networks](https://mybox.inria.fr/f/1fcd41e717/) used for the experiments and
run `TestModel.py --model_path="Path_of_model" ` to reproduce the results.

### References
 - [1] JiaKai, Lu: Mean teacher convolution system for dcase 2018 task 4. DCASE 2018 Challenge report. September, 2018.
 - [2] Tarvainen, A. and Valpola, H., 2017.
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
 In Advances in neural information processing systems (pp. 1195-1204).
