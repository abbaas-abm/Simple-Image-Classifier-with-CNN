# Simple Image Classifier with CNN

### Project Overview
This project implements a basic Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. It's an introductory computer vision task that demonstrates key concepts in deep learning, such as convolutional layers, pooling, and dense layers. The model achieves approximately 98% accuracy on the test set after a few epochs of training.

This is ideal for showcasing foundational ML skills in image classification, suitable for beginners or as a portfolio piece to highlight proficiency in Keras.

### Features
- Loads and preprocesses the MNIST dataset automatically.
- Builds and trains a simple CNN architecture.
- Evaluates model performance on a test set.
- Visualizes a sample prediction using Matplotlib.
- Easy to run with minimal setup.

### Requirements
- Python 3.6+
- TensorFlow (install via `pip install tensorflow`)
- Matplotlib (included with TensorFlow or install via `pip install matplotlib`)

No additional datasets needed; MNIST is fetched via Keras.

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/simple-image-classifier.git
   cd simple-image-classifier
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` with: `tensorflow` and `matplotlib`.)

### Usage
Run the main script:
```
python main.py
```
- The script will train the model for 5 epochs.
- It prints training progress, test accuracy, and displays a visualization of a predicted digit.

To customize:
- Adjust epochs, batch size, or model architecture in `main.py`.
- Use your own images by modifying the prediction section (ensure 28x28 grayscale format).

### How It Works
1. **Data Loading**: MNIST dataset (60,000 training, 10,000 test images) is loaded and normalized (0-1 range).
2. **Model Architecture**:
   - Conv2D (32 filters, 3x3 kernel) → MaxPooling (2x2)
   - Conv2D (64 filters, 3x3 kernel) → MaxPooling (2x2)
   - Flatten → Dense (64 units) → Dense (10 units, softmax)
3. **Training**: Adam optimizer, categorical cross-entropy loss, 5 epochs with 20% validation split.
4. **Evaluation**: Computes accuracy on test set.
5. **Prediction**: Infers on a test image and plots it.

### Example
Here's a snippet from `main.py` for training and prediction:

```python
# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.2f}')

# Visualize a prediction
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {model.predict(test_images[0:1]).argmax()}')
plt.show()
```

**Sample Output** (Console):
```
Epoch 1/5
750/750 [==============================] - 10s 13ms/step - loss: 0.2198 - accuracy: 0.9345 - val_loss: 0.0805 - val_accuracy: 0.9757
Epoch 2/5
750/750 [==============================] - 9s 12ms/step - loss: 0.0615 - accuracy: 0.9810 - val_loss: 0.0599 - val_accuracy: 0.9822
...
Epoch 5/5
750/750 [==============================] - 9s 12ms/step - loss: 0.0201 - accuracy: 0.9935 - val_loss: 0.0489 - val_accuracy: 0.9862
313/313 [==============================] - 1s 3ms/step - loss: 0.0412 - accuracy: 0.9874
Test accuracy: 0.99
```

**Visualization Output**: A Matplotlib window shows a digit (e.g., '7') with title "Predicted: 7".

### Performance Notes
- Accuracy: ~98-99% on test set.
- Training Time: ~1-2 minutes on a standard CPU.
- Improvements: Add dropout for regularization, experiment with more layers, or use data augmentation.

### License
MIT License. Feel free to use and modify for your projects.

---