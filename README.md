# 👀MobileNetV2-TSM

Implementation of [Temporal Shift Module](https://arxiv.org/abs/1811.08383) using TensorFlow library.

[Bukva: Russian Sign Language Alphabet Dataset](https://github.com/ai-forever/bukva) was used for model training.

Main feature of this architecture is that it can analyse sequences of frames using TSM layers that add zero parameters in MobileNetV2 architecture. It allows the resulting model to be used on edge devices without high computational requirements.

## 📒Contents

Repository consists of 3 notebooks that contain code for model training, model evaluation and model conversion from TF SavedModel format to TFLite with uint8 quantization as well as 3 pretrained models.

**The notebooks are:**

- [🤖model](./model.ipynb) - notebook with complete pipeline from dataset creation to model training. It creates dataset from the generator and implements TSM layers into MobileNetV2 before the 2 stage model training.
- [📊Model_Evaluation](./Model_Evaluation.ipynb) - notebook with some metrics of the resulting model. It has confusion matrix, accuracy and classification report.
- [🔀Model_Conversion](./Model_Conversion.ipynb) - notebook with code for TFLite converting. The code inside converts the original model in Float32 SavedModel format into Uint8 TFLite model with fixed batch size.

## 🤔How to use it

If you want to use the same training pipeline as in this repo, you should use ffmpeg to turn your videos to sequence of jpg frames [like in this script](https://github.com/mit-han-lab/temporal-shift-module/blob/master/tools/vid2img_sthv2.py) from the creators of original TSM paper.<br>
This will create *num of videos* folders with frames of certain videos. After this, just chnge the ROOT_DIR to the root directory, containing.<br>
Alternatively, you can use OpenCV library to move through frames of videos.

**Model training is done in 2 stages:**
- Training with frozen conv base
- Training with cosine decay

Because of how TSM works, there is a huge RAM consumption during model training (up to 28 Gb with shuffle on and batch_size = 16). 

**Memory consumption can be lowered by:**
- Lowering batch size
- Turning off or lowering shuffle batch size
- Adjusting dataset prefetch
- Lowering number of segments
- Freezing some of the MobileNet blocks (check in the notebook)

## 📊Models and Metrics

Repository contains pretrained model in saved_model and tflite formats. You might need to specify num_segments as 8 and num_classes as 34 when loading saved_model format.

Models are located inside [./models](./models) folder.

Current implementation has 54.5% accuracy on test data. The most problematic gestures are ones that resemble another gesture, like "Ш" and "Щ" or "Е" and "Ё".

Confusion matrix of current implementation:

<img alt="Confusion Matrix" src="./images/confusion_matrix.png" width="1024" />
