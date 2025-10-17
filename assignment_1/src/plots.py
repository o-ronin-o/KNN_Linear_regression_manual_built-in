import matplotlib.pyplot as plt
import seaborn as sns

def show_heatmap(df=None):
    sns.heatmap(df.corr())

def plot_single_cat(X='category_col', df=None):
    sns.countplot(x=X, data=df)
    plt.xticks(rotation=45)
    plt.show()


import matplotlib.pyplot as plt

def plot_validation_curve(results, title="Validation Metrics vs K"):
    """
    Plot accuracy, precision, recall, and F1 score vs K from a results DataFrame.
    """
    plt.figure(figsize=(8, 6))
    
    ks = results["k"]
    plt.plot(ks, results["accuracy"], marker='o', label='Accuracy')
    plt.plot(ks, results["precision"], marker='s', label='Precision')
    plt.plot(ks, results["recall"], marker='^', label='Recall')
    plt.plot(ks, results["f1_score"], marker='d', label='F1 Score')
    
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_regularization(params, errors, label="Validation Error"):
    plt.figure(figsize=(7, 5))
    plt.plot(params, errors, marker='x', label=label)
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Validation Error")
    plt.title("Regularization Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()
