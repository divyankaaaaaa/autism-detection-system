import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
data_dir = 'dataset/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
valid_dir = os.path.join(data_dir, 'valid')


IMAGE_SIZE = (224, 224)

def load_data(data_dir):
    """Loads images and labels, resizing them for the model."""
    images = []
    labels = []
    for category in ['autistic', 'non_autistic']:
        path = os.path.join(data_dir, category)
        label = 1 if category == 'autistic' else 0
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
                image = cv2.resize(image, IMAGE_SIZE)
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)


# Load the datasets
train_images, train_labels = load_data(train_dir)
val_images, val_labels = load_data(valid_dir)
test_images, test_labels = load_data(test_dir)

# Preprocess the data specifically for MobileNetV2
# This scales pixel values to the range [-1, 1]
train_images_preprocessed = preprocess_input(train_images.copy())
val_images_preprocessed = preprocess_input(val_images.copy())
test_images_preprocessed = preprocess_input(test_images.copy())

print(f"Training data shape: {train_images_preprocessed.shape}")
print(f"Validation data shape: {val_images_preprocessed.shape}")
print(f"Test data shape: {test_images_preprocessed.shape}")


# Load the MobileNetV2 base model, pre-trained on ImageNet
base_model = MobileNetV2(input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet')

# Freeze the layers of the base model so they are not re-trained
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

# Create our custom classification head
from tensorflow.keras.layers import BatchNormalization, Dropout

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid')(x)

# Combine the base model and our custom head
transfer_model = Model(inputs, outputs)

# Compile the model with a lower learning rate
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.00001)
transfer_model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

transfer_model.summary()


# Define callbacks
early_stopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='best_transfer_model.keras', save_best_only=True, monitor='val_loss', verbose=1)
# Define callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Early stopping
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Save best model
model_checkpoint = ModelCheckpoint(
    filepath='best_transfer_model.keras',
    save_best_only=True
)

# Reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    verbose=1
)

# Train the model
history = transfer_model.fit(
    train_images_preprocessed,
    train_labels,
    epochs=25,
    validation_data=(val_images_preprocessed, val_labels),
    callbacks=[early_stopper, model_checkpoint, reduce_lr]
)


# The best weights are already restored by EarlyStopping.
# Now, evaluate the model on the held-out test data.
print("\n--- Evaluating on Test Data ---")
test_loss, test_acc = transfer_model.evaluate(test_images_preprocessed, test_labels)
print(f'\nTest Accuracy: {test_acc:.4f}')

# Generate a classification report
predictions = (transfer_model.predict(test_images_preprocessed) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(test_labels, predictions, target_names=['Non-Autistic', 'Autistic']))

# Display a confusion matrix
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Autistic', 'Autistic'], yticklabels=['Non-Autistic', 'Autistic'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


base_model.trainable = True


print("Number of layers in the base model: ", len(base_model.layers))

for layer in base_model.layers[:-40]:
    layer.trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) 
transfer_model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


transfer_model.summary()


fine_tune_epochs = 10
total_epochs = history.epoch[-1] + 1 + fine_tune_epochs

history_fine_tune = transfer_model.fit(
    train_images_preprocessed,
    train_labels,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1, 
    validation_data=(val_images_preprocessed, val_labels),
    callbacks=[early_stopper, model_checkpoint] 
)


print("\n--- Evaluating Fine-Tuned Model on Test Data ---")
test_loss, test_acc = transfer_model.evaluate(test_images_preprocessed, test_labels)
print(f'\nFinal Test Accuracy after Fine-Tuning: {test_acc:.4f}')


predictions = (transfer_model.predict(test_images_preprocessed) > 0.5).astype(int)
print("\nNew Classification Report:")
print(classification_report(test_labels, predictions, target_names=['Non-Autistic', 'Autistic']))


cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Autistic', 'Autistic'], yticklabels=['Non-Autistic', 'Autistic'])
plt.title('Confusion Matrix (After Fine-Tuning)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

