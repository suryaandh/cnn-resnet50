# Persiapan Data: Persiapkan dataset Endek Bali dengan dua kelas ("Endek A" dan "Endek B"). 
#Pisahkan dataset menjadi bagian training set dan testing set.

# Bangun Model CNN dengan ResNet-50:

# Import library yang diperlukan.
# Muat arsitektur ResNet-50 pre-trained dan hapus lapisan terakhirnya.
# Tambahkan lapisan CNN untuk tugas klasifikasi dengan jumlah kelas yang sesuai.
# Latih model dengan menggunakan training set.
# Pruning Model:

# Gunakan teknik pruning untuk menghapus sebagian bobot yang tidak signifikan dari model.
# Evaluasi Model:

# Evaluasi model dengan menggunakan testing set.
# Hitung confusion matrix menggunakan library scikit-learn.


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Langkah 1: Persiapan Data (sebagai contoh)
# --------------------------------------------------
# Load and preprocess your dataset
# Split dataset into training and testing sets

# Langkah 2: Bangun Model CNN dengan ResNet-50
# --------------------------------------------------
num_classes = 2  # Ubah sesuai dengan jumlah kelas di dataset Anda

# Muat pre-trained ResNet-50 model tanpa lapisan terakhir
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Tambahkan lapisan global average pooling dan lapisan output untuk klasifikasi
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Gabungkan model ResNet-50 dengan lapisan klasifikasi
model = Model(inputs=base_model.input, outputs=predictions)

# Langkah 3: Latih Model CNN dengan Data Training
# --------------------------------------------------
# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model dengan data training
# Misalkan `train_data` adalah data training dan `train_labels` adalah labelnya
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Langkah 4: Pruning Model
# --------------------------------------------------
# Lakukan pruning pada model
# Simpan model sebelum pruning
model_before_pruning = model

# Lakukan pruning pada model
# Pruning dapat dilakukan dengan berbagai metode seperti
# magnitude-based pruning, unit-wise pruning, atau optimal brain damage
# Sebagai contoh, kita gunakan pruning dengan threshold pada magnitude weight
from tensorflow_model_optimization.sparsity import keras as sparsity

# Tentukan target sparsity (contoh: 50%)
target_sparsity = 0.5

# Definisikan metode pruning
pruning_params = {
    'pruning_schedule': sparsity.ConstantSparsity(target_sparsity, begin_step=0, end_step=100),
}
model = sparsity.prune_low_magnitude(model, **pruning_params)

# Kompilasi ulang model setelah pruning
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih ulang model setelah pruning (opsional)
# model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Langkah 5: Evaluasi Model dengan Data Testing dan Confusion Matrix
# --------------------------------------------------
# Prediksi kelas dengan data testing
# Misalkan `test_data` adalah data testing
y_pred = model.predict(test_data)
y_pred_labels = np.argmax(y_pred, axis=1)

# Dapatkan label sebenarnya dari data testing
# Misalkan `test_labels` adalah label yang sebenarnya
y_true_labels = np.argmax(test_labels, axis=1)

# Hitung confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Tampilkan confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Endek A", "Endek B"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (After Pruning)")
plt.show()

# Optional: Evaluasi model sebelum pruning
y_pred_before_pruning = model_before_pruning.predict(test_data)
y_pred_labels_before_pruning = np.argmax(y_pred_before_pruning, axis=1)

cm_before_pruning = confusion_matrix(y_true_labels, y_pred_labels_before_pruning)
disp_before_pruning = ConfusionMatrixDisplay(confusion_matrix=cm_before_pruning, display_labels=["Endek A", "Endek B"])
disp_before_pruning.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Before Pruning)")
plt.show()
