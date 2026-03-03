import tensorflow as tf

print("🔧 Fixing model compatibility...")

# Load with compile=False
model = tf.keras.models.load_model('plastic_classifier_final.h5', compile=False)

# Recompile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save again (overwrites old file)
model.save('plastic_classifier_final.h5')

print("✅ Model fixed and saved!")