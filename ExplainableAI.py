#The pre-trained ResNet distinguisher is retrieved from Aron Gohr's
#Agohr. (n.d.). Agohr/deep_speck: Supplementary code and data to 
#“improving attacks on round-reduced speck32/64 using Deep Learning.” GitHub. https://github.com/agohr/deep_speck 
import lime
import lime.lime_tabular
from IPython.display import HTML
from IPython.display import display
import Speck as sp
import numpy as np
from keras.models import model_from_json

# load distinguishers
json_file = open('single_block_resnet.json', 'r')
json_model = json_file.read()

# Load the pre-trained ResNet model
net5 = model_from_json(json_model)
net5.load_weights('net5_small.h5')

# X_EVALUATE AND Y_EVALUATE since the net5 is already a pre-trained model.
X5, Y5 = sp.make_speck_train_data(10**6, 5)

# A prediction class that is made into a function to be passed into the explainer for XAI 
def prediction(X):

    Y_PREDICTED = net5.predict(X, batch_size=10000).flatten()
    Y_PROB = 1 / (1 + np.exp(-Y_PREDICTED)) #convert into probability , sigmoid function
    return (np.concatenate((1-Y_PROB.reshape(-1, 1), Y_PROB.reshape(-1, 1)), axis=1))# changing 1d array to 2d array 1 col,



explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X5, mode='classification', class_names=['0', '1'], categorical_features=None,
                                                   categorical_names=None)

print(explainer)
# # generate local explaination
explaination = explainer.explain_instance(
    data_row=X5[0], predict_fn=prediction)

# Visualisation
fig = explaination.as_pyplot_figure(label=1)
fig.savefig("lime_explanation.png")

