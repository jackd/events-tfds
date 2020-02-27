import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _save_or_show(anim, save_path):
    if save_path is None:
        plt.show()
    else:
        anim.save(save_path)


def animate_frames(img_data, fps=30, save_path=None):

    fig = plt.figure()
    im = plt.imshow(img_data[0])

    def animate(i):
        im.set_data(img_data[i])
        return im,

    def init():
        return animate(0)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=img_data.shape[0],
        interval=1000 // fps,  # in ms
    )
    _save_or_show(anim, save_path)


def animate_frames_multi(*img_data, fps=30, save_path=None, ax_shape=None):
    if len(img_data) == 1:
        return animate_frames(img_data[0], fps=fps, save_path=save_path)
    if ax_shape is None:
        ax_shape = (1, len(img_data))
    fig, ax = plt.subplots(*ax_shape)
    ax = ax.flatten()
    ims = [a.imshow(img[0]) for a, img in zip(ax, img_data)]

    def animate(i=0):
        out = []
        for (im, img) in zip(ims, img_data):
            out.append(im.set_data(img[i]))
        return out

    def init():
        return animate(0)

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=img_data[0].shape[0],
                                   interval=1000 // fps)

    _save_or_show(anim, save_path)
