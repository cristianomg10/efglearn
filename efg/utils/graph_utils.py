import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_color(x):
    if x == 0:
        return 'red'
    elif x == 1:
        return 'green'
    else:
        return 'blue'


def get_color_sigla(x):
    if x == 0:
        return 'r'
    elif x == 1:
        return 'g'
    else:
        return 'b'


def plot_granules(fbem):
    fig, ax = plt.subplots()

    for i in fbem.granules:
        for j in i.xs:
            if len(i.xs) == 1:
                plt.plot(j[0], j[1], get_color_sigla(i.output_granules[0].coef) + '*')
            else:
                plt.plot(j[0], j[1], 'k*')
        width = i.input_granules[0].L - i.input_granules[0].l if i.input_granules[0].L - i.input_granules[0].l > 0 else \
        i.input_granules[0].L - \
        i.input_granules[
            0].l  # + .1
        height = i.input_granules[1].L - i.input_granules[1].l if i.input_granules[1].L - i.input_granules[1].l > 0 else \
        i.input_granules[1].L - \
        i.input_granules[
            1].l  # + .1
        ax.add_patch(patches.Rectangle((i.input_granules[0].l, i.input_granules[1].l), width, height,
                                       color=get_color(i.output_granules[0].coef)))
    plt.show();


def plot_singular_output(fbem_instance, expected_output):
    """
    Plot singular output of predicted values from FBeM
    to be compared to the expected output
    :param fbem_instance: FBeM instance
    :param expected_output:
    :return:
    """
    # Plot singular output
    plt.figure()
    plt.plot(expected_output, 'k-', label="Expected output")
    plt.plot(fbem_instance.P, 'b-', label="Predicted output")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend(loc=2)


def plot_granular_output(fbem_instance, expected_output):
    """
    Plot granular output predicted from FBeM
    to be compared to the expected output
    :param fbem_instance:
    :param expected_output:
    :return:
    """

    # Plot granular output
    plt.figure()
    plt.plot(expected_output, 'b-', label="Expected output")
    plt.plot(fbem_instance.PLB, 'r-', label="Lower bound")
    plt.plot(fbem_instance.PUB, 'g-', label="Upper bound")
    plt.legend(loc=2)


def plot_rmse_ndei(fbem_instance):
    """
    Plot RMSE and NDEI graphs from FBeM
    :param fbem_instance:
    :return:
    """

    plt.figure()
    plt.plot(fbem_instance.rmse, 'r-', label="RMSE")
    plt.plot(fbem_instance.ndei, 'g-', label="NDEI")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend(loc=2)


def plot_rules_number(fbem_instance):
    """
    Plot the variation of number of FBeM rules
    :param fbem_instance:
    :return:
    """

    # Plot rule number
    plt.figure()
    plt.plot(fbem_instance.store_num_rules, 'r-', label="Number of rules")
    axes = plt.gca()
    axes.set_ylim([0, 30])
    plt.legend(loc=2)


def plot_acc(fbem_instance):
    """
    Plot accuraccy
    :param fbem_instance:
    :return:
    """
    plt.figure()
    plt.plot(fbem_instance.__acc, 'r-', label="Accuracy")
    axes = plt.gca()
    axes.set_ylim([0, 110])
    plt.legend(loc=2)


def plot_rho_values(fbem_instance):
    """
    Plot the variation of number of rho values
    :param fbem_instance:
    :return:
    """

    plt.figure()
    plt.plot(fbem_instance.vec_rho, 'r-', label="Rho variation")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend(loc=2)


def plot_granules_3d_space(fbem_instance, min=0, max=1, indices=[]):
    """
    Plot granules in 3D space

    :param fbem_instance:
    :param min:
    :param max:
    :param indices:
    :return:
    """

    if indices == []:
        indices = range(0, fbem_instance.c)

    colors = ["red", "blue", "black", "gray", "green", "cyan", "yellow", "pink", "fuchsia", "darkgray"]
    colors += colors
    colors += colors

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in indices:
        granule = fbem_instance.granules[i]
        gran = granule.get_granule_for_3d_plotting()
        faces = granule.get_faces_for_3d_plotting()

        # plot vertices
        ax.scatter3D(gran[:, 0], gran[:, 1], gran[:, 2], c=colors[i])

        # plot sides
        face1 = Poly3DCollection(verts=faces, facecolors=colors[i], linewidths=1, edgecolors=colors[i], alpha=.1)
        face1.set_alpha(0.25)
        face1.set_facecolor(colors=colors[i])
        ax.add_collection3d(face1)

        ax.text(gran[0, 0], gran[0, 1], gran[0, 1], s="gamma " + str(i) + " - Y=" +
                                                      str(fbem_instance.granules[i].output_granules[0].coef),
                color="black")

        # for x in granule.xs:
        #    ax.scatter(x[0], x[1], granule.output_granules[0].p(x), c=colors[i])

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_zlim(min, max)

    return ax


def plot_granule_3d_space(granule, ax, i=1):
    """
    Plot granule in 3D space

    :param fbem_instance:
    :param min:
    :param max:
    :param indices:
    :return:
    """

    colors = ["red", "blue", "black", "gray", "green", "cyan", "yellow", "pink", "fuchsia", "darkgray"]
    colors += colors

    gran = granule.get_granule_for_3d_plotting()
    faces = granule.get_faces_for_3d_plotting()

    # plot vertices
    ax.scatter3D(gran[:, 0], gran[:, 1], gran[:, 2], c=colors[i])

    # plot sides
    face1 = Poly3DCollection(verts=faces, facecolors=colors[i], linewidths=1, edgecolors=colors[i], alpha=.1)
    face1.set_alpha(0.25)
    face1.set_facecolor(colors=colors[i])
    ax.add_collection3d(face1)

    ax.text(gran[0, 0], gran[0, 1], gran[0, 1], s="gamma " + str(i), color="black")

    for x in granule.xs:
        ax.scatter(x[0], x[1], granule.output_granules[0].p(x), c=colors[i])

    return ax


def plot_show():
    """
    Plot all the graphs previously prepared
    :return:
    """
    plt.show()


def create_3d_space():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')

    return ax
