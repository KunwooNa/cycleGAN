import matplotlib.pyplot as plt

def display_progress(real_image, fake_image, path, \
    figsize = (12, 6)) : 
    real_image = real_image.detach().cpu()
    fake_image = fake_image.detach().cpu()
    fig, ax = plt.subplot(1, 2, figsize = figsize)
    ax[0].imshow(real_image)
    ax[1].imshow(fake_image)
    ax[0].title.set_text('Input image')
    ax[1].title.set_text('Result image')
    plt.axis('off')
    plt.show()
    plt.savefig(path)