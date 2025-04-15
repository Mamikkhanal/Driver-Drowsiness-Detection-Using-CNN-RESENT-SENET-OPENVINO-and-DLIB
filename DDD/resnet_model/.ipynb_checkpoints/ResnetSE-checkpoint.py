import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from helper import SEResNet18
from sklearn.metrics import roc_curve, auc
from helperFunctions import plot_fig, put_text

#load data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../../ddd_images_train/drowsiness-scale',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    subset='training',
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    '../../ddd_images_train/drowsiness-scale',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    subset='validation',
    class_mode='binary')


#new model
model = SEResNet18(input_shape=(128,128,3), classes = 1)
model.summary()

#optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
optimizer = tf.optimizers.Adam(learning_rate=0.001)

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True) #from_logits=True means output probabilities are not normalized
acc_metric = keras.metrics.BinaryAccuracy()

val_acc_metric = keras.metrics.BinaryAccuracy()
val_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

model.summary()

#train function.
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training = True)

        loss = loss_fn(y, y_pred)
        
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_metric.update_state(y, y_pred)
    
    return loss, y_pred

#validation function
@tf.function
def test_step(x, y):
    val_preds = model(x, training=False)
    loss = val_loss_fn(y, val_preds)
    # Update val metrics
    val_acc_metric.update_state(y, val_preds)
    return val_preds, loss

#training loop
num_epochs = 15

train_writer = tf.summary.create_file_writer('logs/SEModel/train', max_queue = 32)
valid_writer = tf.summary.create_file_writer('logs/SEModel/validation', max_queue = 32)

total_train_files = 0
total_valid_files = 0

for epoch in range(num_epochs):
    print(f"\nStart of Training Epoch {epoch}")
    #not sure if they work at last while saving because of scope, remove later
    loss_value=0
    train_acc=0

    #empty at the begining of every epoch
    fpr=[]
    tpr=[]
    
    valid_tpr=[]
    valid_fpr=[]
    
    #from here
    for batch_idx in range(train_generator.samples//32):
        recent_batch = train_generator.next()
        x_batch = recent_batch[0]
        y_batch = recent_batch[1]

        loss_value, y_pred = train_step(x_batch, y_batch)
        train_acc = acc_metric.result()

        #change (1,32) to (1,batch_number)
        y_pred_shape = y_pred.shape[0]
        y_preds = np.reshape(tf.get_static_value(y_pred),(1,y_pred_shape))[0]

        #to log scalars
        if batch_idx % 32 == 0:
            print(f"epoch {epoch}, batch {batch_idx} loss = {loss_value}, accuracy = {acc_metric.result()}")
            
            #for roc
            batch_fpr, batch_tpr, _ = roc_curve(y_batch, y_preds)
            #store in array for overall epoch roc
            for i in range(len(batch_fpr)):
                fpr.append(batch_fpr[i])
                tpr.append(batch_tpr[i])
            

            with train_writer.as_default(step=total_train_files):
                tf.summary.scalar("train_accuracy", train_acc)
                tf.summary.scalar("train_loss", loss_value)

                #train images truth vs prediction imgs
                annotated_images = put_text(x_batch, y_batch, y_preds)
                tf.summary.image('train_images', annotated_images, max_outputs=12)

                #histogram
                tf.summary.histogram("train_predicted_output", y_preds)

                #increasing after every 32 batches
                total_train_files += 32

        #for trial remove later    
        # if batch_idx%128 == 0 and batch_idx!=0:
        #     break
    
    roc_auc = auc(np.sort(fpr), np.sort(tpr))
    image = plot_fig(np.sort(fpr), np.sort(tpr), roc_auc)
    with train_writer.as_default():
        tf.summary.image('roc_train', image, step=epoch)
    
    print(f"Accuracy over epoch {train_acc}")
    acc_metric.reset_states()

    #to here
    

    #for validataion at the end of every epoch
    for val_batch_idx in range(validation_generator.samples//32):
        recent_validation_batch = validation_generator.next()
        x_validation_batch = recent_validation_batch[0]
        y_validation_batch = recent_validation_batch[1]
        
        y_valid_pred, val_loss = test_step(x_validation_batch,y_validation_batch)

        y_pred_shape = y_valid_pred.shape[0]
        val_y_pred= np.reshape(tf.get_static_value(y_valid_pred),(1,y_pred_shape))[0]
  
        val_acc = val_acc_metric.result()

        #store scalars in validataion
        if val_batch_idx % 32 == 0:
            print(f"Validation acc in valid batch {val_batch_idx}: %.4f" % (float(val_acc),))

            #for roc
            fpr, tpr, _ = roc_curve(y_validation_batch, val_y_pred)
            for i in range(len(batch_fpr)):
                valid_fpr.append(batch_fpr[i])
                valid_tpr.append(batch_tpr[i])

            with valid_writer.as_default(step=total_valid_files):
                tf.summary.scalar("validation_accuracy", val_acc)
                tf.summary.scalar("validation_loss", val_loss)

                #train images truth vs prediction imgs
                annotated_images = put_text(x_validation_batch, y_validation_batch, val_y_pred)
                tf.summary.image('validation_images', annotated_images, max_outputs=12)

                #histogram
                tf.summary.histogram("validation_predicted_output", val_y_pred)
                total_valid_files += 32
                
        #remember to undo comment for model save weights at bottom
        # if val_batch_idx%128 == 0 and val_batch_idx!=0:
        #     break
              
    roc_auc = auc(np.sort(fpr), np.sort(tpr))
    valid_roc_image = plot_fig(np.sort(valid_fpr), np.sort(valid_tpr), roc_auc)
    with train_writer.as_default():
        tf.summary.image('roc_valid', valid_roc_image, step=epoch)


    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc)))
    
    model.save_weights(f'saved_models/data_aug_on_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")} epoch {epoch}_acc_{train_acc}_loss_{loss_value}.h5')
    
train_writer.close()
valid_writer.close()