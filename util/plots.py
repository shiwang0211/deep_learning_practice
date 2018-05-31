import matplotlib.pyplot as plt
import numpy as np
import seaborn  as sns
import pandas as pd

def plot_decision_boundary(pred_func, X, y):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def plot_loss_acc(train_loss, train_acc, val_loss, val_acc, CAL_STEP, num_iter):
    f, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1})

    df = pd.DataFrame(columns = ['Type','Iteration', 'Value'])
    Values = train_loss, train_acc, val_loss, val_acc
    Types = ['Loss', 'Accuracy', 'Loss', 'Accuracy']
    Splits = ['Train', 'Train', 'Validation', 'Validation']

    for Value, Type, Split in zip(Values, Types, Splits):
        dff = pd.DataFrame({'Type': Type,
                           'Split': Split,
                           'Iteration': list(range(num_iter))[::CAL_STEP],
                           'Value': Value})
        df = pd.concat([dff,df])

    ax1 = sns.pointplot(x = 'Iteration', y = 'Value', markers='o', data = df[df.Type == 'Accuracy'], hue = 'Split', ax=axes[0])
    ax1.set( ylabel='Accuracy')
    ax1.xaxis.set_tick_params(rotation=90)
    ax2 = sns.pointplot(x = 'Iteration', y = 'Value', markers='o', data = df[df.Type == 'Loss'], hue = 'Split', ax=axes[1])
    ax2.set( ylabel='Loss')
    ax2.xaxis.set_tick_params(rotation=90)

