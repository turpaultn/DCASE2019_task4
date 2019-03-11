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



Inputs are 864 frames long. The CRNN model is pooling in time to have 108 frames.
Postrocessing (median filtering) of 5 frames is used to obtain events onset and offset for each file.
The baseline system includes evaluations of results using **event-based F-score** as metric. 

#### Script description
The baseline using a mean-teacher model:
It is composed of two networks, both the same CRNN.
- The teacher model is trained regularly. (with synthetic or weak labels depending the data)
- The student model is not trained, its weights are a moving average of the teacher model (at each epoch).
- Inputs of the student model = inputs + gaussian noise
- A cost for consistency between teacher and student model is applied. (for weak and strong predictions)


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
That is why, you can found the weights of the experiments at this adress: 
(launch TestModel.py --model_path="Path_of_model" to get the results)

### References
 - [1] JiaKai, Lu: Mean teacher convolution system for dcase 2018 task 4. DCASE 2018 Challenge report. September, 2018. 
 - [2] Tarvainen, A. and Valpola, H., 2017. 
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. 
 In Advances in neural information processing systems (pp. 1195-1204).