# Some plot functions
from Base.packages import *

# Figure parameters
figsize1 = (20, 10)
figsize2 = (10, 10)
dpi = 150


# Plot the prediction curve
def pred_curve(y_test, y_pred, title='Title', figsize=figsize1, dpi=dpi):
    # Compute the performance index
    r2 = 100 * r2_score(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    # Plot the prediction curve
    for i in range(y_test.shape[1]):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(y_test[:, i], label='Ground Truth')
        plt.plot(y_pred[:, i], label='Prediction')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(title + ' (QV-{}) (R2: {:.2f}%, RMSE: {:.3f})'.format(i + 1, r2[i], rmse[i]))
        plt.grid()
        plt.legend()
    plt.show()


# Plot the prediction scatter
def pred_scatter(y_test, y_pred, title='Title', figsize=figsize2, dpi=dpi):
    # Compute the performance index
    r2 = 100 * r2_score(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    # Plot the prediction scatter
    for i in range(y_test.shape[1]):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(y_test[:, i], y_pred[:, i], label='Samples')
        plt.plot(y_test[:, i], y_test[:, i], 'r', label='Isoline')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(title + ' (QV-{}) (R2: {:.2f}%, RMSE: {:.3f})'.format(i + 1, r2[i], rmse[i]))
        plt.grid()
        plt.legend()
    plt.show()


# Plot confusion matrix
def confusion(y_test, y_pred, title='Title', figsize=figsize2, dpi=dpi):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(cm, cmap='YlGnBu', annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title(title)
    plt.show()


# Plot scatter
def scatter(pc, color, title='Title', figsize=figsize2, dpi=dpi):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(pc[:, 0], pc[:, 1], c=color, cmap='tab10')
    plt.xlabel('Component_1')
    plt.ylabel('Component_2')
    plt.title(title)
    plt.show()
