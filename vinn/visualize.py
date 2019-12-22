import matplotlib.pyplot as plt
import numpy as np

def plt_training(train_dict, figsize=(12, 4), ylim1=[], ylim2=[], legend=True):
    _, ax = plt.subplots(1, 2, figsize=figsize)

    nll = train_dict['nll']
    kl = train_dict['kl']
    accuracy_train = train_dict['accuracy_train']
    accuracy_test = train_dict['accuracy_test']
    
    n_minibatches = len(nll[0])
    epoch_loss = np.mean(nll, axis=1) + np.mean(kl, axis=1)
    nll = [item for sublist in nll for item in sublist]
    kl = [item for sublist in kl for item in sublist]
    ELBOloss = [nll[i] + kl[i] for i in range(len(nll))]    

    scaled_x = np.arange(0, len(ELBOloss), n_minibatches) + n_minibatches/2
    ax[0].plot(ELBOloss, color='green', alpha=0.5)
    ax[0].plot(scaled_x, epoch_loss, color='green')
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("ELBO loss")
    ax[0].set_ylim(*ylim1)

    ax[0] = ax[0].twinx()
    ax[0].plot(scaled_x, accuracy_train, color='blue', label='train acc.')
    if accuracy_test is not None:
        ax[0].plot(scaled_x, accuracy_test, color='red', label='val. acc.')
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlim(0.0, len(ELBOloss))
    ax[0].set_ylim(*ylim2)

    ax[1].plot(nll, color='cyan')
    ax[1].set_ylabel("Negative Log Likelihood")
    ax[1].set_xlabel("Iteration")
    ax[1] = ax[1].twinx()
    ax[1].plot(kl, color='orange', label='KL')
    ax[1].set_ylabel("KL divergence")
    ax[1].set_xlim(0.0, len(ELBOloss))
    
    if legend:
        ax[0].legend()
        ax[1].plot([], color='cyan', label='NLL')
        ax[1].legend()
        
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    
def show_images(images, labels, targets, outputs, mask, n=30):

    n = min(np.sum(mask), n)
    if n == 0:
        print('The mask is too strict, no images found.')
        return
    
    images = images[mask][:n]
    labels = labels[mask][:n]
    targets = targets[mask][:n]
    outputs = outputs[mask][:n]
    
    n_columns = 10 if n > 10 else n
    n_rows = int(np.ceil(float(n)/10))
    
    _, ax = plt.subplots(2*n_rows, n_columns, figsize=(1.6*n_columns, 3.2*n_rows), sharey='row', squeeze=False)
    for i in range(n_rows):
        for j in range(n_columns):
            if (i*n_columns+j) >= n:
                ax[2*i][j].axis('off')
                ax[2*i+1][j].axis('off')
                continue
            
            ax[2*i][j].imshow(images[i*10+j], interpolation='bilinear', cmap = 'gray')
            ax[2*i][j].axis('off')

            colors = ["blue",] * 10
            colors[labels[i*10+j]] = "red"
            colors[targets[i*10+j]] = "green"               

            ax[2*i+1][j].bar(np.arange(10), outputs[i*10+j], color=colors, align='center')
            ax[2*i+1][j].set_xlim(-0.5, 9.5)
            ax[2*i+1][j].set_ylim(0, 1)
            ax[2*i+1][j].spines['top'].set_visible(False)
            ax[2*i+1][j].spines['right'].set_visible(False)
            ax[2*i+1][j].tick_params(top=False, bottom=False, right=False, left=False)
            ax[2*i+1][j].set_xticks(np.arange(10))
            ax[2*i+1][j].set_yticks(np.arange(0, 1.5, 0.5))
    plt.show()
    
def variance_hist(variances, mask=None, bins=20, ax=None, sharey=False, range=[0, 0.25], log=True):

    def _hist(var, bins, ax, range, log, color='b'):
        ax.hist(var, bins=bins, color=color, alpha=0.7, 
                      log=log, range=range)
        ax.set_xlabel("Predictive variance")
        ax.set_ylabel("Count") 
    
    if mask is None:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))     
        _hist(variances, bins, ax, range=range, log=log)
    elif variances.ndim == 1:
        if ax is None:
            _, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=sharey)     
        _hist(variances, bins, ax[0], range=range, color='b', log=log)
        _hist(variances[mask], bins, ax[1], range=range, color='g', log=log)
        _hist(variances[~mask], bins, ax[2], range=range, color='r', log=log)
        
        ax[0].set_title("All predictions")
        ax[1].set_title("Correct predictions")
        ax[2].set_title("Wrong predictions")
        
        plt.subplots_adjust(wspace=0.3)
    else:
        if ax is None:
            _, ax = plt.subplots(3, 3, figsize=(15, 13), sharey=sharey)
        for i, var in enumerate(variances):
            _hist(var, bins, ax[i][0], range=range, color='b', log=log)
            _hist(var[mask], bins, ax[i][1], range=range, color='g', log=log)
            _hist(var[~mask], bins, ax[i][2], range=range, color='r', log=log)
           
        ax[0][0].set_title("All predictions")
        ax[0][1].set_title("Correct predictions")
        ax[0][2].set_title("Wrong predictions")

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    
def variance_analysis(images, targets, labels, variances, classes):
    
    def samples_image(mask):
        size = images.shape[2]
        img = np.zeros((2*size+2, 2*size+2))
        img[:size,:size] = images[mask][0]
        img[size+2:,:size] = images[mask][1]
        img[:size,size+2:] = images[mask][2]
        img[size+2:,size+2:] = images[mask][3]
        return img
    
    n_rows = len(classes)
    
    _, ax = plt.subplots(n_rows, 4, figsize=(4*4, n_rows*4), squeeze=False)

    low_var_mask = (variances < 0.03)
    high_var_mask = (variances > 0.2)
    for i, c in enumerate(classes):
        class_mask = (targets == c)
        
        ax[i][0].imshow(255 - samples_image((class_mask & low_var_mask)), 
                        interpolation='bilinear', cmap = 'gray')
        ax[i][0].axis('off')
        ax[i][2].imshow(255 - samples_image((class_mask & high_var_mask)), 
                        interpolation='bilinear', cmap = 'gray')
        ax[i][2].axis('off')

        ax[i][1].hist(variances[class_mask], alpha=0.7, color='blue', 
                      range=[0, 0.25], bins=10)
        ax[i][1].set_ylabel('Count')
        ax[i][1].set_yticks([])
        ax[i][1].set_xticks([0.0, 0.25])

        x, height = np.unique(labels[class_mask], return_counts=True)
        x = list(x)
        ax[i][3].bar(np.arange(10), [height[x.index(n)] if n in x else 0 for n in range(10)], 
                     tick_label=np.arange(10), align='center', color='m', alpha=0.7)
        ax[i][3].set_xlim(-1, 10)
        ax[i][3].set_ylabel('Count')
        ax[i][3].set_yticks([])

    ax[0][1].set_title('Class predictive variance')
    ax[0][3].set_title('Predictions count plot')
    ax[0][0].set_title('Low variance samples')
    ax[0][2].set_title('High variance samples')

    plt.subplots_adjust(wspace=0.3)
    plt.show()