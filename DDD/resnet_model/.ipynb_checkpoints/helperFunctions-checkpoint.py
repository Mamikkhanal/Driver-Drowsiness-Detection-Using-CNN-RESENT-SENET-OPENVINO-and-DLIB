import cv2
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def put_text(imgs, actuals, preds):
    result = np.empty_like(imgs)
    
    for i in range(imgs.shape[0]):
        actual = actuals[i]
        pred = preds[i]
        pred = float("{0:.3f}".format(pred))
        if isinstance(actual, bytes):
            actual = actual.decode()
            
        if isinstance(pred, bytes):
            pred = pred.decode() 
        # You may need to adjust text size and position and size..
        # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
        
        result[i, :, :, :] = cv2.putText(imgs[i, :, :, :], f"Expected: {actual}", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        result[i, :, :, :] = cv2.putText(result[i, :, :, :], f"Predicted: {pred}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    return result


def plot_fig(fpr,tpr,roc_auc):
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Diagram')
    plt.legend(loc="lower right")
    #convert to png
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image