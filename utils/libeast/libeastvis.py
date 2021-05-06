import matplotlib.pyplot as plt;
import numpy as np;
def debug_input(im, text_polys, score_map, geo_map, training_mask):
    vim=(im*(127)+127).astype(np.uint8);

    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    # axs[0].imshow(im[:, :, ::-1])
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # for poly in text_polys:
    #     poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
    #     poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
    #     axs[0].add_artist(Patches.Polygon(
    #         poly * 4, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
    #     axs[0].text(poly[0, 0] * 4, poly[0, 1] * 4, '{:.0f}-{:.0f}'.format(poly_h * 4, poly_w * 4),
    #                    color='purple')
    # axs[1].imshow(score_map)
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    axs[0, 0].imshow(vim[:, :, ::-1])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    for poly in text_polys:
        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
        # axs[0, 0].add_artist(Patches.Polygon(
        #     poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w),
                       color='purple')
    axs[0, 1].imshow(score_map[:, :,0])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].imshow(geo_map[::, ::, 0])
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(geo_map[::, ::, 1])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[2, 0].imshow(geo_map[::, ::, 2])
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    axs[2, 1].imshow(training_mask[::, ::,0])
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])
    plt.tight_layout()
    plt.show()

